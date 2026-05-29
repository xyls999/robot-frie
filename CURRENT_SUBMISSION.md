# 当前 FireDetect 提交与训练说明

本仓库当前主线是 **PP-YOLOE+ CRN-S，736 输入尺寸，原始 clean 数据训练**。

注意：当前主线不使用我之前生成的增强图，也不补标、不伪标签，目标是尽量拟合官方给出的原始标注风格。

## 当前可提交压缩包

推荐提交：

```text
submission.zip
```

同内容版本包：

```text
submission_firedetect_736_clean_epoch140_thr040_080_020_nms_area.zip
```

压缩包结构符合官方要求：

```text
submission.zip
├── predict.py
├── model/
│   ├── infer_cfg.yml
│   ├── model.pdmodel
│   └── model.pdiparams
└── PaddleDetection/
    └── deploy/python/
        ├── preprocess.py
        ├── utils.py
        └── keypoint_preprocess.py
```

## 当前模型来源

当前 `model/` 使用的是：

```text
PaddleDetection_train/output/firedetect_ppyoloe_plus_crn_s_180e_736_clean_kfold_base/best_model.pdparams
```

导出目录：

```text
PaddleDetection_train/output_inference/export_best_736_clean_kfold0_epoch140_pdmodel/ppyoloe_plus_crn_s_180e_736_clean_kfold_base
```

该模型来自 clean 5 折 fold0 训练：

```text
best epoch = 140
COCO AP = 0.7224030613
```

## predict.py 后处理策略

当前 `predict.py` 使用分类阈值和轻量后处理来拟合官方标注风格：

```text
battery threshold = 0.40
board   threshold = 0.80
fire    threshold = 0.20

extra class-wise NMS IoU = 0.70
fire min area = 600
battery / board min area = 0
batch size = 16
```

本地 405 张原始标签全量评估：

```text
mean F1 = 0.970862
battery F1 = 0.9921
board   F1 = 0.9565
fire    F1 = 0.9639
```

本地正式入口烟测：

```text
EXIT = 0
result_count = 950
405 张耗时约 14.28 秒
本机约 28 FPS
```

## 本地调参脚本

本地后处理扫描脚本：

```text
firedetect_local_tune.py
```

用途：

```text
1. 用导出模型跑全量图片，保存 raw_predictions.json
2. 离线扫描分类阈值、面积过滤、额外 NMS
3. 不直接修改正式 predict.py
4. 找到稳定参数后再手动同步到 predict.py
```

示例：

```powershell
cd "D:\baidu ruanjian"
D:\conda_envs\paddle\python.exe firedetect_local_tune.py `
  --model_dir "D:\baidu ruanjian\PaddleDetection_train\output_inference\export_best_736_clean_kfold0_epoch140_pdmodel\ppyoloe_plus_crn_s_180e_736_clean_kfold_base" `
  --anno "D:\baidu ruanjian\PaddleDetection_train\dataset\firedetect\annotations\all.json" `
  --image_dir "D:\baidu ruanjian\PaddleDetection_train\dataset\firedetect\images" `
  --out_dir "D:\baidu ruanjian\analysis\local_tune_736_clean_epoch140_all405" `
  --batch_size 16 `
  --thresholds 0.38,0.38,0.38 `
  --scan
```

## 继续训练配置

新建的继续训练配置：

```text
PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_80e_736_clean_all_continue_epoch140.yml
```

设计：

```text
init weights = clean fold0 epoch140 best_model
train data   = dataset/firedetect/annotations/all.json
epoch        = 80
base_lr      = 0.00008
batch_size   = 4
input size   = 736
empty_ratio  = 1.0
augmentation = very light RandomDistort + RandomFlip
```

启动脚本：

```text
start_continue_80e_736_clean_all_epoch140.bat
```

手动命令：

```powershell
cd "D:\baidu ruanjian\PaddleDetection_train"
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_80e_736_clean_all_continue_epoch140.yml --eval
```

## 导出当前最好模型

当前最好模型已经导出并放入根目录 `model/`。

如果需要重新导出：

```powershell
cd "D:\baidu ruanjian\PaddleDetection_train"
D:\conda_envs\paddle_export\python.exe tools\export_model.py `
  -c configs\firedetect\ppyoloe_plus_crn_s_180e_736_clean_kfold_base.yml `
  --output_dir=output_inference\export_best_736_clean_kfold0_epoch140_pdmodel `
  -o weights=output\firedetect_ppyoloe_plus_crn_s_180e_736_clean_kfold_base\best_model.pdparams use_gpu=False
```

说明：导出使用 `paddle_export` 环境，因为它能稳定导出官方要求的 `model.pdmodel + model.pdiparams`。
