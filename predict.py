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
# 让当前提交包中的 PaddleDetection 目录可以被 import；使用脚本绝对路径，避免评测机 cwd 不同。
sys.path.insert(0, SCRIPT_DIR)
import json

yaml = None
np = None
paddle = None
Config = None
create_predictor = None
preprocess = None
Resize = None
NormalizeImage = None
Permute = None
PadStride = None


def init_runtime():
    """把重依赖放到主流程 try 内导入，避免顶层 import 失败导致非 0 退出。"""
    global yaml, np, paddle, Config, create_predictor
    global preprocess, Resize, NormalizeImage, Permute, PadStride
    import yaml as _yaml
    import numpy as _np
    import paddle as _paddle
    from paddle.inference import Config as _Config
    from paddle.inference import create_predictor as _create_predictor
    from PaddleDetection.deploy.python.preprocess import (
        preprocess as _preprocess,
        Resize as _Resize,
        NormalizeImage as _NormalizeImage,
        Permute as _Permute,
        PadStride as _PadStride,
    )
    yaml = _yaml
    np = _np
    paddle = _paddle
    Config = _Config
    create_predictor = _create_predictor
    preprocess = _preprocess
    Resize = _Resize
    NormalizeImage = _NormalizeImage
    Permute = _Permute
    PadStride = _PadStride


class PredictConfig():
    """读取 PaddleDetection 导出模型的推理配置。"""

    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        # infer_cfg.yml 记录了模型结构、预处理流程、类别名、NMS 等导出信息。
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
    cwd = os.getcwd()
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for line in dirs:
        line = line.strip()
        if line:
            line = line.replace('\\', '/')
            if not os.path.isabs(line):
                candidates = [
                    os.path.join(cwd, line),
                    os.path.join(infer_dir, line),
                    os.path.join(SCRIPT_DIR, line),
                    os.path.join(os.path.dirname(infer_dir), line),
                ]
                line = next((p for p in candidates if os.path.exists(p)), candidates[0])
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
    """创建 Paddle Inference predictor；优先 GPU，初始化失败时退回 CPU 避免脚本非 0 退出。"""
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    # 评测要求 FPS，默认走 GPU；显存池设小一些，避免部分评测机初始化失败。
    if paddle.device.is_compiled_with_cuda():
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(2)
    # 稳定性优先：部分评测环境下 PP-YOLOE 静态图开启 IR 优化可能触发底层崩溃。
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.enable_memory_optim()
    # 使用 zero-copy API 手动拷贝输入输出张量，避免老式 feed/fetch 接口。
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
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
    return predictor, config


def create_inputs(imgs, im_info):
    """把预处理后的单张/多张图片组装成 PaddleDetection 模型需要的输入字典。"""
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        # im_shape 和 scale_factor 用于模型内部/后处理把坐标映射回原图尺度。
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
        # 如果未来启用 batch 推理，不同尺寸图片需要 pad 到同一个 H/W 才能 stack。
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
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        """根据 infer_cfg.yml 中的 Preprocess 列表动态创建预处理算子。"""
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            # op_type 例如 Resize/NormalizeImage/Permute，对应 preprocess.py 中的类名。
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
        # PaddleDetection 检测模型常见输出：boxes 和 boxes_num。
        # boxes 每行通常是 [class_id, score, x1, y1, x2, y2]。
        np_boxes = self.predictor.get_output_handle(
            output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(
            output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def iou_xyxy(box_a, box_b):
    """计算两个 xyxy 框的 IoU，用于提交端额外轻量 NMS。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def apply_extra_nms(items, nms_iou):
    """对模型内置 NMS 后的结果再做一层保守 class-wise NMS，只删除高度重叠重复框。"""
    if nms_iou is None:
        return items
    kept_all = []
    for cls_id in (1, 2, 3):
        cls_items = [item for item in items if item["type"] == cls_id]
        cls_items.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
        kept = []
        for item in cls_items:
            box = [item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"]]
            duplicate = False
            for old in kept:
                old_box = [old["x"], old["y"], old["x"] + old["width"], old["y"] + old["height"]]
                if iou_xyxy(box, old_box) >= nms_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)
        kept_all.extend(kept)
    kept_all.sort(key=lambda item: item.get("_order", 0))
    return kept_all


def append_detection_items(result_items, image_ids, det_results, thresholds, min_area, extra_nms_iou):
    """把一个 batch 的 PaddleDetection 输出转成赛题 JSON 条目。"""
    start = 0
    boxes_num = det_results['boxes_num']
    boxes = det_results['boxes']
    for image_idx, image_id in enumerate(image_ids):
        im_bboxes_num = int(boxes_num[image_idx])
        end = start + im_bboxes_num
        image_items = []
        if im_bboxes_num > 0:
            bbox_results = boxes[start:end, 2:]
            id_results = boxes[start:end, 0]
            score_results = boxes[start:end, 1]
            for idx in range(im_bboxes_num):
                score = float(score_results[idx])
                cls_id = int(id_results[idx]) + 1
                if cls_id not in (1, 2, 3):
                    continue
                cls_threshold = thresholds.get(cls_id, thresholds.get("default", 0.38))
                if score < cls_threshold:
                    continue
                x1 = float(bbox_results[idx][0])
                y1 = float(bbox_results[idx][1])
                x2 = float(bbox_results[idx][2])
                y2 = float(bbox_results[idx][3])
                if not np.isfinite([x1, y1, x2, y2]).all():
                    continue
                x = max(0.0, min(x1, x2))
                y = max(0.0, min(y1, y2))
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if width <= 0.0 or height <= 0.0:
                    continue
                if width * height < min_area.get(cls_id, 0.0):
                    continue
                image_items.append({
                    "image_id": str(image_id),
                    "type": int(cls_id),
                    "x": float(x),
                    "y": float(y),
                    "width": float(width),
                    "height": float(height),
                    "segmentation": [],
                    "_score": score,
                    "_order": idx
                })
        for item in apply_extra_nms(image_items, extra_nms_iou):
            item.pop("_score", None)
            item.pop("_order", None)
            result_items.append(item)
        start = end


def predict_image(
        detector,
        image_list,
        result_path,
        thresholds,
        min_area,
        extra_nms_iou,
        batch_size=8):
    """按 batch 推理，并按赛题指定 JSON schema 写出结果。"""
    result_items = []
    batch_images = []
    batch_infos = []
    batch_ids = []

    def flush_batch():
        if not batch_images:
            return
        inputs = create_inputs(batch_images, batch_infos)
        det_results = detector.predict(inputs)
        append_detection_items(
            result_items, batch_ids, det_results, thresholds, min_area, extra_nms_iou)
        batch_images.clear()
        batch_infos.clear()
        batch_ids.clear()

    for im_path in image_list:
        if not os.path.exists(im_path):
            # data_txt 中某张图片路径异常时跳过该图，保证脚本整体正常退出。
            continue
        # preprocess 会按 infer_cfg.yml 中的顺序执行 Resize/Normalize/Permute 等操作。
        try:
            im, im_info = preprocess(im_path, detector.preprocess_ops)
        except Exception:
            continue
        # 赛题要求 image_id 是文件名本身，不包含目录和扩展名。
        batch_images.append(im)
        batch_infos.append(im_info)
        batch_ids.append(os.path.splitext(os.path.basename(im_path))[0])
        if len(batch_images) >= batch_size:
            try:
                flush_batch()
            except Exception:
                batch_images.clear()
                batch_infos.clear()
                batch_ids.clear()
    try:
        flush_batch()
    except Exception:
        pass
    write_result(result_path, result_items)
    print("Results written to", result_path)


def main(infer_txt, result_path, det_model_path, thresholds, min_area, extra_nms_iou):
    """评测入口的主流程：加载配置和模型，读取图片列表，生成结果文件。"""
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, thresholds, min_area, extra_nms_iou)


if __name__ == '__main__':
    start_time = time.time()
    # 评测提交包约定模型固定放在根目录 model/ 下。
    det_model_path = os.path.join(SCRIPT_DIR, "model")
    # 当前 all405 继续训练模型的本地最优阈值。
    thresholds = {"default": 0.40, 1: 0.40, 2: 0.40, 3: 0.44}
    min_area = {1: 0.0, 2: 0.0, 3: 0.0}
    extra_nms_iou = None
    if len(sys.argv) != 3:
        # 评测系统会传入两个参数；本分支只兜底异常调用，仍然保证退出码为 0。
        fallback_path = sys.argv[2] if len(sys.argv) > 2 else "result.json"
        write_result(fallback_path, [])
        sys.exit(0)

    infer_txt = sys.argv[1]
    result_path = sys.argv[2]
    try:
        # Paddle Inference 静态图模型需要启用静态图模式。
        init_runtime()
        paddle.enable_static()
        main(infer_txt, result_path, det_model_path, thresholds, min_area, extra_nms_iou)
        print('total time:', time.time() - start_time)
    except Exception:
        # 任何 Python 层异常都写出合法空结果，避免评测系统因非 0 返回码直接判失败。
        write_result(result_path, [])
        print("Fallback empty result written to", result_path)
        sys.exit(0)
