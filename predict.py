# -*- coding: utf-8 -*-
"""
3 类目标检测推理脚本（battery / board / fire）
调用方式（评测系统自动调用）：
    python predict.py <data_txt> <result_json>

整体数据流：
1. 读取 model/infer_cfg.yml，按导出模型记录的配置创建预处理算子。
2. 读取 data_txt 中的图片路径，逐张做 Resize/Normalize/Permute 等预处理。
3. 通过 Paddle Inference 加载 model.pdmodel + model.pdiparams 并执行前向推理。
4. 将 PaddleDetection 输出的 [class_id, score, x1, y1, x2, y2] 转成赛题要求的 JSON。
"""
import os
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import json
import yaml
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from PaddleDetection.deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from PaddleDetection.deploy.python.utils import argsparser, Timer, get_current_memory_mb


class PredictConfig():
    """读取 PaddleDetection 导出模型的推理配置。"""

    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf.get('min_subgraph_size', 3)
        self.labels = yml_conf['label_list']
        self.mask = yml_conf.get('mask', False)
        self.use_dynamic_shape = yml_conf.get('use_dynamic_shape', False)
        self.tracker = yml_conf.get('tracker', None)
        self.nms = yml_conf.get('NMS', None)
        self.fpn_stride = yml_conf.get('fpn_stride', None)
        self.print_config()

    def print_config(self):
        """启动时打印模型和预处理配置，便于在评测日志中排查配置是否加载正确。"""
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


def get_test_images(infer_file):
    """读取评测系统传入的图片路径列表，并转成可直接访问的绝对/相对路径。"""
    infer_dir = os.path.dirname(os.path.abspath(infer_file))
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for line in dirs:
        line = line.strip()
        if line:
            line = line.replace('\\', '/')
            if not os.path.isabs(line):
                line = os.path.join(infer_dir, line)
            images.append(line)
    return images


def write_result(result_path, result_items):
    """按赛题要求写出 JSON：顶层只有 result，result 是列表。"""
    result_dir = os.path.dirname(os.path.abspath(result_path))
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as ft:
        json.dump({"result": result_items}, ft, ensure_ascii=False)


def load_predictor(model_dir):
    """创建 Paddle Inference predictor；优先 GPU，初始化失败时退回 CPU。"""
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    if paddle.device.is_compiled_with_cuda():
        config.enable_use_gpu(500, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(2)
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.switch_use_feed_fetch_ops(False)
    try:
        predictor = create_predictor(config)
    except Exception:
        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams')
        )
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(2)
        config.switch_ir_optim(False)
        config.disable_glog_info()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
    return predictor, config


def create_inputs(imgs, im_info):
    """把预处理后的单张/多张图片组装成 PaddleDetection 模型需要的输入字典。"""
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))
    origin_scale_factor = np.concatenate(scale_factor, axis=0)
    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [float(max_shape_h) / float(im_h),
                   float(max_shape_w) / float(im_w)]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


class Detector(object):
    """封装模型加载、预处理算子构建和单次推理。"""

    def __init__(self, pred_config, model_dir):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        """根据 infer_cfg.yml 中的 Preprocess 列表动态创建预处理算子。"""
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict(self, inputs):
        """执行一次 Paddle Inference 前向推理，并取回检测框与每张图的框数量。"""
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        np_boxes = self.predictor.get_output_handle(
            output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(
            output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def predict_image(detector, image_list, result_path, threshold):
    """逐张图片推理，并按赛题指定 JSON schema 写出结果。"""
    c_results = {"result": []}
    for im_path in image_list:
        if not os.path.exists(im_path):
            continue
        input_im_lst = []
        input_im_info_lst = []
        try:
            im, im_info = preprocess(im_path, detector.preprocess_ops)
        except Exception:
            continue
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        image_id = os.path.basename(im_path).split('.')[0]
        try:
            det_results = detector.predict(inputs)
            im_bboxes_num = det_results['boxes_num'][0]
            if im_bboxes_num > 0:
                bbox_results  = det_results['boxes'][0:im_bboxes_num, 2:]
                id_results    = det_results['boxes'][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][0:im_bboxes_num, 1]
                for idx in range(im_bboxes_num):
                    if float(score_results[idx]) >= threshold:
                        c_results["result"].append({
                            "image_id": image_id,
                            "type": int(id_results[idx]) + 1,
                            "x": float(bbox_results[idx][0]),
                            "y": float(bbox_results[idx][1]),
                            "width":  float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                            "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                            "segmentation": []
                        })
        except Exception:
            continue
    write_result(result_path, c_results["result"])
    print("Results written to", result_path)


def main(infer_txt, result_path, det_model_path, threshold):
    """评测入口的主流程：加载配置和模型，读取图片列表，生成结果文件。"""
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold)


if __name__ == '__main__':
    start_time = time.time()
    det_model_path = os.path.join(SCRIPT_DIR, "model")
    threshold = 0.57
    paddle.enable_static()
    infer_txt   = sys.argv[1]
    result_path = sys.argv[2]
    try:
        main(infer_txt, result_path, det_model_path, threshold)
        print('total time:', time.time() - start_time)
    except Exception:
        write_result(result_path, [])
        print("Fallback empty result written to", result_path)
