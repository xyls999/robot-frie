# FireDetect 永久验证集方案

这个方案用于解决当前最大问题：`all405` 本地分数会被训练集记忆污染，不能代表线上泛化。

## 划分规则

脚本：

```text
PaddleDetection_train/tools/prepare_firedetect_holdout_kfold.py
```

固定随机种子：

```text
20260530
```

划分逻辑：

```text
1. 将 405 张图按 frame 编号顺序切成 5 个时间段
2. 每个时间段抽 10 张，合计 50 张 permanent holdout
3. 这 50 张永久不参与训练、伪标签、增强、阈值搜索
4. 剩余 355 张作为 train_pool
5. train_pool 再做 5 折交叉验证
```

输出目录：

```text
PaddleDetection_train/dataset/firedetect/annotations/holdout_kfold/
```

关键文件：

```text
holdout.json          # 永久验证集，50 张
train_pool.json       # 训练池，355 张
fold0_train.json
fold0_val.json
...
fold4_train.json
fold4_val.json
summary_holdout_kfold.json
```

## 正确使用方式

配置选择：

```text
先看 train_pool 内部 5-fold 平均效果
```

防止自欺：

```text
最后再看 permanent holdout
```

注意：

```text
已经用 all405 训练过的模型，不能拿这个 holdout 当真实泛化验证。
因为它已经见过 holdout 图片。
```

所以新的真实验证训练必须从 Obj365 预训练权重开始，不能从 all405 模型继续。

## 推荐训练配置

内部 5-fold 的 fold0：

```text
PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_180e_736_clean_holdout_fold0.yml
```

启动：

```powershell
cd "D:\baidu ruanjian"
.\start_holdout_180e_736_fold0_training.bat
```

最终保守训练：

```text
PaddleDetection_train/configs/firedetect/ppyoloe_plus_crn_s_220e_736_clean_holdout_final355.yml
```

启动：

```powershell
cd "D:\baidu ruanjian"
.\start_holdout_220e_736_final355_training.bat
```

## 决策建议

如果目标是提高线上 A 榜泛化，不要再以 `all405 F1` 作为主要指标。

更靠谱的排序是：

```text
1. 5-fold 平均 F1
2. permanent holdout F1
3. 线上 A 榜分数
4. all405 F1 只作为记忆能力参考
```
