"""Summarize fixed web-matched runs without selecting models on Test."""
import argparse
import json
from pathlib import Path

import pandas as pd

from spectral_icl.public_validation import checksum
from spectral_icl.io import file_hash, write_json


def summarize(root):
    root = Path(root)
    audit = json.loads((root / 'validation_report.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed' or audit['failed_model_runs']:
        raise ValueError('A complete audited run is required')
    data = pd.read_csv(root / 'all_runs.csv')
    test = data[data.split.eq('test')].copy()
    # Saved accuracy_pct is rounded for display; aggregate exact count ratios.
    test['accuracy_pct'] = test['correct'] / test['total'] * 100
    canonical = test[test.equivalent_to.isna()].copy()
    out = root / 'summary'
    out.mkdir(exist_ok=True)
    keys = ['task', 'dataset_id', 'source_day', 'target_day', 'reference_mode',
            'control', 'target_k', 'k', 'model_id']
    metrics = ['accuracy_pct', 'macro_f1', 'balanced_accuracy', 'mean_group_accuracy',
               'seen_accuracy', 'unseen_accuracy', 'h_mean']
    grouped = canonical.groupby(keys, dropna=False)
    aggregate = grouped[metrics].agg(['mean', 'std', 'count'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(out / 'test_all_conditions.csv', index=False)
    test.to_csv(out / 'test_each_run.csv', index=False)
    canonical[canonical.task.eq('T3')].groupby(
        ['control', 'target_k', 'model_id'], dropna=False
    ).accuracy_pct.agg(['mean', 'std', 'count']).reset_index().to_csv(out / 't3_adaptation.csv', index=False)
    full = test[test.reference_mode.eq('full_reference')]
    full.groupby(['task', 'dataset_id', 'source_day', 'target_day', 'model_id'], dropna=False)[
        metrics].agg(['mean', 'std', 'count']).to_csv(out / 'full_reference_metrics.csv')
    pair_keys = ['task', 'dataset_id', 'source_day', 'target_day', 'repeat', 'model_id']
    pairs = test[test.reference_mode.eq('context_k')].merge(
        full[pair_keys + ['accuracy_pct']], on=pair_keys, suffixes=('_k', '_full'), validate='many_to_one')
    pairs['full_minus_k_accuracy_pp'] = pairs.accuracy_pct_full - pairs.accuracy_pct_k
    pairs.to_csv(out / 'paired_full_minus_k_each_repeat.csv', index=False)
    pairs.groupby([c for c in pair_keys if c != 'repeat'] + ['k'], dropna=False)[
        'full_minus_k_accuracy_pp'].agg(['mean', 'std', 'count']).reset_index().to_csv(
            out / 'paired_full_minus_k_summary.csv', index=False)

    def table(frame, rowkeys):
        values = frame.groupby(rowkeys + ['model_id'], dropna=False).accuracy_pct.agg(['mean', 'std'])
        values['value'] = values.apply(lambda r: f'{r["mean"]:.2f} ± {r["std"]:.2f}', axis=1)
        pivot = values.value.unstack('model_id').reset_index()
        columns = list(pivot.columns)
        lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---']*len(columns)) + '|']
        for row in pivot.itertuples(index=False, name=None):
            lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
        return '\n'.join(lines)

    text = '''# 与网页数据包匹配的本地对比结果

网页索引共有435套包，本次按用户要求完整排除蜂蜜和Fruit_purees共40套，实际运行395套。两类数据此前已有的经典模型结果仍保留在先前基线结果中。每次直接读取对应train.json参考样本和已导出特征，对完整Val/Test预测；没有重新划分、重新拟合光谱降维或使用Test调参。以下均为Test准确率（%，可用重复划分的均值 ± 样本标准差）。标准差描述划分敏感性，不是独立重复实验的置信区间，也不把不同公共任务平均成一个总分。

通用分类模型为SVM-RBF、随机森林、CNN1D和Transformer。第三批另运行光谱质心浓度内插。T3中30套等价source-only别名照常保存预测，但汇总时剔除别名，避免重复计数。

## 第一批：T1，完整参考

'''
    text += table(full[full.task.eq('T1')], ['target_day'])
    text += '\n\n## 第二批：T2，完整参考\n\n'
    text += table(full[full.task.eq('T2')], ['source_day', 'target_day'])
    text += '\n\n## 第二批：T3，Day0→Day14\n\n'
    text += 'source_only为源日每类5例；target_only只用目标日K例；mixed合并源日5例与目标日K例；size_matched_source用同总量源日参考。四组使用同一源日锚点拟合的LDA8。\n\n'
    text += table(canonical[canonical.task.eq('T3')], ['control', 'target_k'])
    text += '\n\n## 第三批：未见15%/30%，完整参考\n\n'
    unseen = full[full.task.eq('UNSEEN')]
    u = unseen.groupby(['target_day', 'model_id'])[['accuracy', 'seen_accuracy', 'unseen_accuracy', 'h_mean']].mean()*100
    u.to_csv(out / 'unseen_full_reference_means_pct.csv')
    text += '| 日期 | 模型 | 总体准确率 | 已见准确率 | 未见准确率 | H-mean |\n|---|---|---|---|---|---|\n'
    for (day, model), row in u.iterrows():
        text += f'| {day:g} | {model} | ' + ' | '.join(f'{v:.2f}' for v in row) + ' |\n'
    text += '\n所有闭集分类器（包括CNN1D和Transformer）无法输出训练标签之外的15%和30%，其未见准确率及H-mean必为0，仅作为闭集下限。内插模型显式知道浓度数值及候选标签，才具备输出未见浓度的机制。当前两处缺失浓度均为相邻已见等级的中点，因此顺序原型与数值内插产生相同原型，登记为等价未重复运行；这些结果不证明LLM的未见类别能力。\n'
    text += '\n## 第四批：公共数据，完整参考\n\n'
    text += table(full[full.task.eq('PUBLIC')], ['dataset_id'])
    text += '''

## 模型与评价协议

- SVM：参考样本拟合StandardScaler，RBF核，C=1、gamma=scale。
- 随机森林：200棵树，随机种子沿用数据包support_seed，单线程拟合。浓度内插：导出空间的欧氏距离，已见类质心之间按浓度线性内插，不外推。
- CNN1D和Transformer：使用项目中的LUMIR分类网络结构，Adam（学习率0.001）、交叉熵、批量128，最多20轮；只用Val早停（耐心6轮）并选取最佳检查点。未见浓度任务中，Val内未出现在Train的类别不参与早停。Test仅作最终评估。蜂蜜与Fruit_purees按用户要求不运行这两个模型。
- 经典模型超参数预先固定且不使用Val选模；CNN1D/Transformer用Val做早停。公共数据原导出流程中已完成的Val维数选择仍属于该数据包协议，不能宣称整个上游没有Val参与。
- 拟合只使用train.json参考记录，保持原始JSON精度。经典模型仅输入features；日期用于构造参考/查询池，不作为额外特征。LLM另能看到日期元数据，这项信息利用能力不完全相同。
- 每次保存Train/Val/Test逐样本预测、准确率、宏F1、平衡准确率、每类指标、混淆矩阵、支持ID、拟合模型与哈希。Train为参考集自身的in_sample诊断，不能和LLM留一评估直接比较。
- 另保存按现有group等权准确率及1000次group bootstrap区间。区间条件于当前划分及可用group，不能扩展解释为独立制备批次泛化；五次划分也并非五套独立采集数据。
- T1/T2的K条件仍使用完整Train标签池拟合的LDA8；未见任务为七类Train拟合LDA6。K只限制参考数量，不代表整个方法仅使用K个标签。公共数据沿用各自原文处理协议及已披露实现近似，详见PUBLIC_DATASET_PROTOCOL.md。
- Honey的分组依据品牌/类别/采集，不是独立蜂蜜罐；Beef固定官方Test且完整参考与K5等价，物理ID与编码含义未确认；水果泥缺少物理母样ID；姜黄二分类与七分类共用同一批数据；Fresh meats的paper处理特征不意味着精确复现原文分类器。不能据此声称无批次泄漏风险或跨独立来源泛化。
- 完整参考与K=1/3/5的配对差值见summary/paired_full_minus_k_summary.csv；Beef full与K5不算额外独立证据。全部条件（含K）见summary/test_all_conditions.csv，逐次Test见summary/test_each_run.csv。
- 相同参考、种子、候选标签、参数和代码的模型复用拟合缓存，各条件仍单独保存输出。没有运行真实LLM API或LUMIR提示词复现；model_registry.csv记录未运行及用户排除项。API请求为0。受测LLM的“禁止工具”要求仍保留在Prompt中，本地模型实验独立运行。

## 审计与复跑

'''
    text += f'审计逐条核对{audit["prediction_rows_verified"]:,}条预测的样本ID、顺序、真实标签、分组和指标；核对{audit["unique_fitted_estimators_verified"]}个独立拟合模型、{audit["equivalent_prediction_files_verified"]}份等价条件预测文件及输入输出校验和。含数值警告的运行数：{audit["warning_runs"]}，详情保留在validation_report.json和各模型manifest.json。\n\n'
    for warning in audit['warnings']:
        text += f'警告条件：{warning["web_directory"]} / {warning["model_id"]}，{warning["warnings"]}。拟合权重及三分区预测分数均通过有限值检查；保留原始警告，不替换结果或依据Test重训。\n\n'
    text += '''配置：configs/web_matched_baselines.yaml。默认只预览；使用已有specllm环境执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& 'D:\\Anaconda\\envs\\specllm\\python.exe' scripts/run_web_matched_baselines.py --execute
& 'D:\\Anaconda\\envs\\specllm\\python.exe' scripts/audit_web_matched_baselines.py --results <新结果目录>
& 'D:\\Anaconda\\envs\\specllm\\python.exe' scripts/summarize_web_matched_baselines.py --results <新结果目录>
```

每次执行创建新的结果目录，不覆盖本次数据。本次完整结果已通过逐样本全量回读审计。此次汇总不做基于Test的模型选择或显著性宣称。

研究材料来源：本地四批网页实验包、固定分组划分和公共数据协议；无新增外部文献证据。当前证据支持这些固定数据与处理协议下的本地分类对照，真实LLM比较尚待网页测试。
'''
    (root / 'RESULTS.md').write_text(text, encoding='utf-8')
    write_json(out / 'manifest.json', dict(source_all_runs_sha256=file_hash(root/'all_runs.csv'),
        audit_report_sha256=file_hash(root/'validation_report.json'), summary_code_sha256=file_hash(__file__),
        test_accuracy_source='exact correct/total, not rounded accuracy_pct',
        t3_aliases_excluded=True, test_used_for_model_selection=False))
    checksum(root)
    print(root / 'RESULTS.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    summarize(parser.parse_args().results)
