# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
this code is based on https://github.com/open-mmlab/mmpose/mmpose/core/post_processing/post_transforms.py

关键点/姿态类模型常用的仿射变换工具。

在本赛题目标检测模板中，它主要被 preprocess.py 的 WarpAffine 算子引用；
如果当前 infer_cfg.yml 只使用 Resize/NormalizeImage/Permute，那么评测主流程通常不会走到这里。
"""
import cv2
import numpy as np


class EvalAffine(object):
    """按给定短边尺寸做仿射缩放，常用于关键点模型评估前处理。"""

    def __init__(self, size, stride=64):
        super(EvalAffine, self).__init__()
        self.size = size
        self.stride = stride

    def __call__(self, image, im_info):
        s = self.size
        h, w, _ = image.shape
        # 计算原图到目标评估尺寸的仿射矩阵，并执行 warpAffine。
        trans, size_resized = get_affine_mat_kernel(h, w, s, inv=False)
        image_resized = cv2.warpAffine(image, trans, size_resized)
        return image_resized, im_info


def get_affine_mat_kernel(h, w, s, inv=False):
    """根据原图高宽和目标短边尺寸，生成仿射矩阵与缩放后的输出尺寸。"""
    if w < h:
        w_ = s
        h_ = int(np.ceil((s / w * h) / 64.) * 64)
        scale_w = w
        scale_h = h_ / w_ * w

    else:
        h_ = s
        w_ = int(np.ceil((s / h * w) / 64.) * 64)
        scale_h = h
        scale_w = w_ / h_ * h

    center = np.array([np.round(w / 2.), np.round(h / 2.)])

    size_resized = (w_, h_)
    # center/scale/size_resized 共同决定从原图坐标到目标图坐标的映射。
    trans = get_affine_transform(
        center, np.array([scale_w, scale_h]), 0, size_resized, inv=inv)

    return trans, size_resized


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """根据中心点、尺度、旋转角度和输出尺寸计算 2x3 仿射矩阵。

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # 通过中心点、旋转方向点和第三点各构造一个三点坐标系。
    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    # inv=True 时计算反向映射，常用于把模型输出坐标还原回原图。
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """计算 UDP 关键点处理中的无偏仿射矩阵。

        This code is based on
        https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py

        Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * np.cos(theta) + 0.5 * size_input[1] *
        np.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * np.sin(theta) - 0.5 * size_input[1] *
        np.cos(theta) + 0.5 * size_target[1])
    return matrix


def rotate_point(pt, angle_rad):
    """把二维点绕原点旋转指定弧度。

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """由两个点构造仿射变换所需的第三个点。

    To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


class TopDownEvalAffine(object):
    """对 top-down 关键点模型输入图像执行仿射变换。

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp

    def __call__(self, image, im_info):
        rot = 0
        imshape = im_info['im_shape'][::-1]
        # 如果 im_info 没有显式给出 center/scale，就默认使用整张图的中心和尺寸。
        center = im_info['center'] if 'center' in im_info else imshape / 2.
        scale = im_info['scale'] if 'scale' in im_info else imshape
        if self.use_udp:
            trans = get_warp_matrix(
                rot, center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0], scale)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        else:
            trans = get_affine_transform(center, scale, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)

        return image, im_info


def expand_crop(images, rect, expand_ratio=0.3):
    """根据检测框裁出人体区域并适当外扩，供 top-down 关键点模型使用。"""
    imgh, imgw, c = images.shape
    label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
    # 官方模板里 label=0 通常表示 person；非 person 检测框不做关键点裁剪。
    if label != 0:
        return None, None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    if h_half > w_half * 4 / 3:
        w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    # 裁剪范围需要夹在原图边界内，避免索引越界。
    ymin = max(0, int(center[0] - h_half))
    ymax = min(imgh - 1, int(center[0] + h_half))
    xmin = max(0, int(center[1] - w_half))
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect
