# 当前 FireDetect 提交与训练说明

本仓库当前主线是 **PP-YOLOE+ CRN-S，736 输入尺寸，最新 holdout_fold0 train 最佳权重**。

注意：当前主线不使用我之前生成的增强图，也不补标、不伪标签，目标是尽量拟合官方给出的原始标注风格。

## 当前可提交压缩包

推荐提交：

```text
submission.zip
```

当前 `submission.zip` 已更新为最新 train 结束版本：

```text
holdout_fold0 best epoch40 model + batch size 8 + thresholds 0.34 / 0.76 / 0.58
```

上一版保留包：

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

当前 `model/` 使用的是最新结束的 train 输出中的最佳模型：

```text
PaddleDetection_train/output/firedetect_ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0/best_model.pdparams
```

导出目录：

```text
PaddleDetection_train/output_inference/export_latest_train_736_holdout_fold0_best_epoch40_pdmodel/ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0
```

该模型来自 holdout_fold0 train：

```text
train output = firedetect_ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0
best epoch   = 40
COCO AP      = 0.6970231121
model_final  = epoch 180, COCO AP 0.6780278354
```

## predict.py 后处理策略

当前 `predict.py` 使用分类阈值和轻量后处理来拟合官方标注风格：

```text
battery threshold = 0.34
board   threshold = 0.76
fire    threshold = 0.58

extra class-wise NMS IoU = disabled
min area = 0 for all classes
batch size = 8
```

本地 405 张原始标签全量评估：

```text
mean F1 = 0.909931
battery F1 = 0.9025
board   F1 = 0.9133
fire    F1 = 0.9140
```

本地正式入口烟测：

```text
EXIT = 0
result_count = 973
405 张耗时约 17.94 秒
本机约 22.6 FPS
永久 holdout mean F1 = 0.892466
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
