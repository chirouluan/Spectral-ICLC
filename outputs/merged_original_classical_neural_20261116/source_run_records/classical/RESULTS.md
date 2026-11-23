> Historical source summary; old T3 adaptation results have been removed from this published copy.

# 与网页数据包匹配的本地对比结果

本次使用当前435套网页包，完成1,800次模型运行，无失败。每次直接读取对应train.json的参考样本和已导出特征，对完整Val/Test预测；没有重新划分、重新拟合光谱降维或使用Test调参。以下均为Test准确率（%，五次划分的均值 ± 样本标准差）。标准差描述划分敏感性，不是独立重复实验的置信区间，也不把不同公共任务平均成一个总分。

两个通用模型为SVM-RBF和随机森林。第三批另运行光谱质心浓度内插。T3中30套等价source-only别名照常保存预测，但汇总时剔除别名，避免重复计数。

## 第一批：T1，完整参考

| target_day | random_forest | svm_rbf |
| --- | --- | --- |
| 0.0 | 99.44 ± 1.24 | 90.00 ± 4.21 |
| 7.0 | 98.89 ± 2.48 | 85.56 ± 4.12 |
| 14.0 | 95.56 ± 2.48 | 85.56 ± 2.32 |

## 第二批：T2，完整参考

| source_day | target_day | random_forest | svm_rbf |
| --- | --- | --- | --- |
| 0.0 | 7.0 | 20.56 ± 6.09 | 12.78 ± 3.73 |
| 0.0 | 14.0 | 12.78 ± 2.48 | 11.11 ± 0.00 |
| 7.0 | 14.0 | 33.33 ± 1.96 | 16.11 ± 3.62 |

## 第三批：未见15%/30%，完整参考

| 日期 | 模型 | 总体准确率 | 已见准确率 | 未见准确率 | H-mean |
|---|---|---|---|---|---|
| 0 | random_forest | 77.78 | 100.00 | 0.00 | 0.00 |
| 0 | svm_rbf | 75.56 | 97.14 | 0.00 | 0.00 |
| 7 | random_forest | 76.67 | 98.57 | 0.00 | 0.00 |
| 7 | svm_rbf | 70.56 | 90.71 | 0.00 | 0.00 |
| 14 | random_forest | 76.11 | 97.86 | 0.00 | 0.00 |
| 14 | svm_rbf | 69.44 | 89.29 | 0.00 | 0.00 |

四种闭集分类器无法输出训练标签之外的15%和30%，其未见准确率及H-mean必为0，仅作为闭集下限。内插模型显式知道浓度数值及候选标签，才具备输出未见浓度的机制。当前两处缺失浓度均为相邻已见等级的中点，因此顺序原型与数值内插产生相同原型，登记为等价未重复运行；这些结果不证明LLM的未见类别能力。

## 第四批：公共数据，完整参考

| dataset_id | random_forest | svm_rbf |
| --- | --- | --- |
| beef | 76.67 ± 3.33 | 82.00 ± 4.47 |
| coffee | 88.57 ± 10.83 | 94.29 ± 3.19 |
| fresh_meats | 83.33 ± 5.27 | 86.67 ± 5.27 |
| fruit_purees | 95.93 ± 1.63 | 96.50 ± 0.79 |
| honey_adulteration | 93.50 ± 1.56 | 82.54 ± 1.27 |
| olive_oils | 84.67 ± 13.25 | 85.33 ± 5.06 |
| turmeric_authenticity | 89.52 ± 3.98 | 85.71 ± 0.00 |
| turmeric_levels | 58.10 ± 6.21 | 40.00 ± 9.87 |

## 模型与评价协议

- SVM：参考样本拟合StandardScaler，RBF核，C=1、gamma=scale。
- 随机森林：200棵树，随机种子沿用数据包support_seed，单线程拟合。浓度内插：导出空间的欧氏距离，已见类质心之间按浓度线性内插，不外推。
- 所有超参数预先固定。本次估计器训练不使用Val/Test；公共数据原导出流程中已完成的Val维数选择仍属于该数据包协议，不能宣称整个上游没有Val参与。
- 拟合只使用train.json参考记录，保持原始JSON精度。经典模型仅输入features；日期用于构造参考/查询池，不作为额外特征。LLM另能看到日期元数据，这项信息利用能力不完全相同。
- 每次保存Train/Val/Test逐样本预测、准确率、宏F1、平衡准确率、每类指标、混淆矩阵、支持ID、拟合模型与哈希。Train为参考集自身的in_sample诊断，不能和LLM留一评估直接比较。
- 另保存按现有group等权准确率及1000次group bootstrap区间。区间条件于当前划分及可用group，不能扩展解释为独立制备批次泛化；五次划分也并非五套独立采集数据。
- T1/T2的K条件仍使用完整Train标签池拟合的LDA8；未见任务为七类Train拟合LDA6。K只限制参考数量，不代表整个方法仅使用K个标签。公共数据沿用各自原文处理协议及已披露实现近似，详见PUBLIC_DATASET_PROTOCOL.md。
- Honey的分组依据品牌/类别/采集，不是独立蜂蜜罐；Beef固定官方Test且完整参考与K5等价，物理ID与编码含义未确认；水果泥缺少物理母样ID；姜黄二分类与七分类共用同一批数据；Fresh meats的paper处理特征不意味着精确复现原文分类器。不能据此声称无批次泄漏风险或跨独立来源泛化。
- 完整参考与K=1/3/5的配对差值见summary/paired_full_minus_k_summary.csv；Beef full与K5不算额外独立证据。全部条件（含K）见summary/test_all_conditions.csv，逐次Test见summary/test_each_run.csv。
- 相同参考、种子、候选标签、参数和代码的模型复用拟合缓存，各条件仍单独保存输出。没有运行CNN、Transformer、LUMIR或真实LLM，model_registry.csv明确记录未运行原因；API请求为0。受测LLM的“禁止工具”要求仍保留在Prompt中，本次经典模型实验独立运行。

## 审计与复跑

审计逐条核对520,375条预测的样本ID、顺序、真实标签、分组和指标；核对1420个独立拟合模型、420份等价条件预测文件及输入输出校验和。含数值警告的运行数：1，详情保留在validation_report.json和各模型manifest.json。


配置：configs/web_matched_baselines.yaml。默认只预览；使用已有specllm环境执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& 'D:\Anaconda\envs\specllm\python.exe' scripts/run_web_matched_baselines.py --execute
& 'D:\Anaconda\envs\specllm\python.exe' scripts/audit_web_matched_baselines.py --results <新结果目录>
& 'D:\Anaconda\envs\specllm\python.exe' scripts/summarize_web_matched_baselines.py --results <新结果目录>
```

每次执行创建新的结果目录，不覆盖本次数据。离线测试41项通过；完整结果另经全量回读审计。此次汇总不做基于Test的模型选择或显著性宣称。

研究材料来源：本地四批网页实验包、固定分组划分和公共数据协议；无新增外部文献证据。当前证据支持这些固定数据与处理协议下的本地分类对照，真实LLM比较尚待网页测试。
