@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Regenerating permanent holdout + 5-fold splits...
D:\conda_envs\paddle\python.exe tools\prepare_firedetect_holdout_kfold.py
echo.
echo Config: configs\firedetect\ppyoloe_plus_crn_s_220e_736_clean_holdout_final355.yml
echo Training on 355-image train_pool and evaluating on fixed 50-image holdout.
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_220e_736_clean_holdout_final355.yml --eval
pause
