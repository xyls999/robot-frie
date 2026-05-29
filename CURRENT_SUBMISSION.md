# Current FireDetect Submission

This repository currently uses the 704 input PP-YOLOE+ CRN-S generalization model.

## Final Submission Zip

Use this file for platform evaluation:

```text
submission_firedetect_704_generalize_best_thr038_fastnms.zip
```

The zip contains:

```text
predict.py
model/infer_cfg.yml
model/model.pdmodel
model/model.pdiparams
PaddleDetection/deploy/python/preprocess.py
PaddleDetection/deploy/python/utils.py
PaddleDetection/deploy/python/keypoint_preprocess.py
```

## Inference Settings

- Algorithm: PP-YOLOE+ CRN-S
- Export input size: 704 x 704
- Prediction threshold in `predict.py`: 0.38
- Prediction batch size in `predict.py`: 16
- Export NMS config: `configs/firedetect/ppyoloe_plus_crn_s_500e_704_generalize_fastnms_firedetect.yml`
- Model files in repository: `model/`

Local smoke test on 405 images:

```text
EXIT=0
FPS ~= 27.1
result count = 925
```

Validation set result at threshold 0.38:

```text
mean F1 ~= 0.875
battery F1 ~= 0.974
board   F1 ~= 0.778
fire    F1 ~= 0.873
```

## Training Config

Main training YAML:

```text
PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_500e_704_generalize_firedetect.yml
```

Training command:

```powershell
cd "D:\baidu ruanjian\PaddleDetection_train"
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_500e_704_generalize_firedetect.yml --eval
```

Dataset expected by this YAML:

```text
PaddleDetection_train/dataset/firedetect/images
PaddleDetection_train/dataset/firedetect/annotations/train.json
PaddleDetection_train/dataset/firedetect/annotations/val.json
```

The dataset itself is intentionally not committed.

## Export Config

Fast submission export YAML:

```text
PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_500e_704_generalize_fastnms_firedetect.yml
```

Export command used:

```powershell
cd "D:\baidu ruanjian\PaddleDetection_train"
D:\conda_envs\paddle_export\python.exe tools\export_model.py -c configs\firedetect\ppyoloe_plus_crn_s_500e_704_generalize_fastnms_firedetect.yml --output_dir=output_inference -o weights=output/firedetect_ppyoloe_plus_crn_s_500e_704_generalize/best_model use_gpu=false
```
