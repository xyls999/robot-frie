@echo off
cd /d "D:\baidu ruanjian\PaddleDetection_train"
echo Config: configs\firedetect\ppyoloe_plus_crn_s_1000e_832_first_replay_firedetect.yml
echo Method: replay first best route, train.json/val.json, 832 input, 1000 epoch, lr 0.00015
echo Output: output\firedetect_ppyoloe_plus_crn_s_1000e_832_first_replay
D:\conda_envs\paddle\python.exe tools\train.py -c configs\firedetect\ppyoloe_plus_crn_s_1000e_832_first_replay_firedetect.yml --eval
pause
