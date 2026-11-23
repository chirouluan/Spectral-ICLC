> Historical source summary; old T3 adaptation results have been removed from this published copy.

# CNN1D与Transformer重新训练结果

完成790次模型/数据包运行，失败0次；395套包，蜂蜜及Fruit_purees排除。结果审计通过。

## 同条件与旧训练方案配对

以下按任务与数据集分别汇总各条件的Test准确率；均值跨重复划分和不同参考条件，仅用于概览，细分条件见CSV。

|任务|数据集|模型|旧准确率均值|新准确率均值|配对差值（百分点）|条件数|
|---|---|---|---:|---:|---:|---:|
| PUBLIC | beef | cnn1d | 53.83% | 67.50% | +13.67 | 20 |
| PUBLIC | beef | transformer | 37.67% | 48.67% | +11.00 | 20 |
| PUBLIC | coffee | cnn1d | 76.43% | 72.50% | -3.93 | 20 |
| PUBLIC | coffee | transformer | 62.86% | 68.57% | +5.71 | 20 |
| PUBLIC | fresh_meats | cnn1d | 64.33% | 55.17% | -9.17 | 20 |
| PUBLIC | fresh_meats | transformer | 71.00% | 73.33% | +2.33 | 20 |
| PUBLIC | olive_oils | cnn1d | 50.33% | 64.33% | +14.00 | 20 |
| PUBLIC | olive_oils | transformer | 49.33% | 55.83% | +6.50 | 20 |
| PUBLIC | turmeric_authenticity | cnn1d | 73.57% | 73.57% | +0.00 | 20 |
| PUBLIC | turmeric_authenticity | transformer | 69.52% | 75.24% | +5.71 | 20 |
| PUBLIC | turmeric_levels | cnn1d | 25.71% | 38.10% | +12.38 | 20 |
| PUBLIC | turmeric_levels | transformer | 25.24% | 40.95% | +15.71 | 20 |
| T1 | surimi9 | cnn1d | 64.81% | 80.93% | +16.11 | 60 |
| T1 | surimi9 | transformer | 33.01% | 61.71% | +28.70 | 60 |
| T2 | surimi9 | cnn1d | 18.15% | 16.85% | -1.30 | 60 |
| T2 | surimi9 | transformer | 15.93% | 13.98% | -1.94 | 60 |
| UNSEEN | surimi9 | cnn1d | 55.46% | 67.82% | +12.36 | 60 |
| UNSEEN | surimi9 | transformer | 37.55% | 52.69% | +15.14 | 60 |

## 训练与评价

Adam学习率0.001、weight decay 0.0001，batch size=min(8,训练样本数)，最后单条批次合并到前一批；最多100轮、至少20轮，验证交叉熵连续20轮未改善则早停，最小改善0.0001。随机种子使用数据包support_seed。所有特征缩放仅拟合训练参考；验证集用于选择检查点，因此验证成绩不作为无偏的独立最终评估。测试集仅作最终评估。

未见浓度任务中，训练与选检查点只使用已见类，两个闭集模型不能预测未见浓度。T2使用目标日带标签验证集早停，故该深度模型对照不是完全无目标日标签的零适应；经典模型没有使用验证标签选检查点。原始光谱已导出为监督LDA成分，K-shot条件的表示使用更大的源Train标签池。CNN在LDA坐标上卷积，不能解释为直接在连续原始波长上提取局部光谱特征。两种深度模型是有监督拟合基线，与冻结LLM上下文学习的训练预算不同。

本次固定一组训练参数，未开展系统调参；最大轮数与早停也不保证每个条件收敛。五次划分并非五套独立采集数据，样本很少的条件存在训练波动。

详细指标：[同条件结果](summary/test_all_conditions.csv)、[新旧逐条件配对](summary/paired_neural_test.csv)、[传统模型和新结果](summary/all_models_test_each_run.csv)、[训练诊断](summary/training_diagnostics.csv)。
