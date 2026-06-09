# FireDetect Pre-Ignore Best Backup

This backup records the best FireDetect state before the later ignore-masked
data experiment.

## Best Submission

- Final package: `submission_09085base_noise_robust_short_best_SAFEGRAPH_20260608.zip`
- Current upload copy: `submission.zip`
- SHA256: `7C2AC73D29CC6611937A81F1766AC64A05DAE63341BB70617FEB1BE0C125FF5F`
- Inference safety rule: SAFEGRAPH package layout, using the 0.907-safe graph
  and original single-image `predict.py` inference path.
- Local verification previously passed by extracting the zip and running
  `python predict.py <data_txt> <result_json>` on the 405-image list with exit
  code 0.

## Best Training Checkpoint

- Checkpoint directory:
  `PaddleDetection_train/output/firedetect_ppyoloe_plus_crn_s_09085base_768_noise_robust_short_30e_20260608`
- Best params:
  `PaddleDetection_train/output/firedetect_ppyoloe_plus_crn_s_09085base_768_noise_robust_short_30e_20260608/best_model.pdparams`
- SHA256:
  `B0F38AA75F8D23FBB9AF1E13BC279AF6EB4BD79EA7C8ACCB64628FD6B573FB55`
- Best validation record:
  `best_model.pdstates` contains `metric=0.9320523243250892`, `epoch=15`.

## Pretrain Checkpoint

- Pretrain params:
  `PaddleDetection_train/output/firedetect_ppyoloe_plus_crn_s_0907base_768_smallfire_edgeboard_jump_holdout_rerun_20260606_002933/best_model.pdparams`
- SHA256:
  `09E0DDF0E957C83E4A67687918A5322C68E42A7E197ED697BDCCEBDFEC99AE64`

## Main Training YML

- `PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_09085base_768_smallfire_edgeboard_weakfog_300e_20260608.yml`

Despite the filename containing `weakfog_300e`, this file is the current
pre-ignore best short fine-tune config. Its `save_dir` points to:

`output/firedetect_ppyoloe_plus_crn_s_09085base_768_noise_robust_short_30e_20260608`

## Enabled Child YML Chain

- `PaddleDetection_train/configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml`
- `PaddleDetection_train/configs/datasets/coco_detection.yml`
- `PaddleDetection_train/configs/runtime.yml`
- `PaddleDetection_train/configs/ppyoloe/_base_/optimizer_80e.yml`
- `PaddleDetection_train/configs/ppyoloe/_base_/ppyoloe_plus_crn.yml`
- `PaddleDetection_train/configs/ppyoloe/_base_/ppyoloe_plus_reader.yml`

The original pre-comment backups and `.zh_comments.yml` copies are also kept
where present so the active Chinese-commented configs can be compared with the
pre-comment files.

## Inference Code And Model Files

- `predict.py`
- `model/infer_cfg.yml`
- `model/model.pdmodel`
- `model/model.pdiparams`
- `PaddleDetection/deploy/python/preprocess.py`
- `PaddleDetection/deploy/python/utils.py`
- `PaddleDetection/deploy/python/keypoint_preprocess.py`

Important hashes:

- `predict.py`:
  `2F8D1E987BAAEC2FA0B6866888BACDE02A106C309A0DE5F5D1404BA2B70AC10F`
- `model/model.pdmodel`:
  `40BC431BB236EF72F4DF461C70FA5E54A79B9D0F65F43B2BEFC9B7C5A216FFE6`
- `model/model.pdiparams`:
  `FD40D8A30EE54E607340FC7E62C96212DDB54C6C95258009DA17C7B7D3E6AAA8`

## Training Command

Run from `PaddleDetection_train`:

```powershell
D:\conda_envs\paddle\python.exe tools/train.py -c configs/firedetect/ppyoloe_plus_crn_s_09085base_768_smallfire_edgeboard_weakfog_300e_20260608.yml --eval
```

## Notes

- Do not treat the later
  `ppyoloe_plus_crn_s_09085best_768_ignoremasked_val70_16e_20260609.yml`
  experiment as this best backup.
- Do not replace the safe `predict.py` inference path with the batched variant
  for final submission unless a fresh extracted-zip test passes.
