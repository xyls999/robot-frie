@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Regenerating clean original-data 5-fold COCO splits...
D:\conda_envs\paddle\python.exe tools\prepare_firedetect_clean_kfold.py
echo.
echo Config: configs\firedetect\ppyoloe_plus_crn_s_180e_736_clean_kfold_base.yml
echo Dataset: dataset\firedetect only, no generated augmentation images
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_180e_736_clean_kfold_base.yml --eval
pause
