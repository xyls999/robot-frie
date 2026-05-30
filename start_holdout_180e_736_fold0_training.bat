@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Regenerating permanent holdout + 5-fold splits...
D:\conda_envs\paddle\python.exe tools\prepare_firedetect_holdout_kfold.py
echo.
echo Config: configs\firedetect\ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0.yml
echo Permanent holdout is NOT used by this fold training.
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0.yml --eval
pause
