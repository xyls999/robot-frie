@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Config: configs\firedetect\ppyoloe_plus_crn_s_500e_704_generalize_firedetect.yml
echo Dataset: dataset\firedetect\annotations\train.json + val.json
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_500e_704_generalize_firedetect.yml --eval
pause
