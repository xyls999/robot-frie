@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Config: configs\firedetect\ppyoloe_plus_crn_s_80e_736_clean_all_continue_epoch140.yml
echo Init weights: output\firedetect_ppyoloe_plus_crn_s_180e_736_clean_kfold_base\best_model.pdparams
echo Dataset: dataset\firedetect\annotations\all.json only, no generated augmentation images
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_80e_736_clean_all_continue_epoch140.yml --eval
pause
