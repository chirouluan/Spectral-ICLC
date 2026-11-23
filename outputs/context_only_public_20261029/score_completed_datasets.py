"""Score only the three completed datasets after freezing all 60 predictions."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / 'data_json/four_batches/batch_04_public'
DATASETS = ['beef', 'coffee', 'fresh_meats']
manifest = json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
experiments = [x for x in manifest['experiments'] if x.split('/')[0] in DATASETS]
assert len(experiments) == 60
pred_paths = [OUT/rel/'predictions.json' for rel in experiments]
assert all(p.is_file() for p in pred_paths), 'All 60 outputs must exist before scoring'
frozen = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pred_paths}
prior_path = ROOT/'outputs/four_batches_strata_rerun/predictions.jsonl'
previous = {}
for line in prior_path.read_text(encoding='utf-8-sig').splitlines():
    item = json.loads(line)
    eid = item['experiment_id']
    assert eid not in previous, 'Duplicate prior experiment ID'
    previous[eid] = item
rows = []
paired = []
for rel, path in zip(experiments, pred_paths):
    pred = json.loads(path.read_text(encoding='utf-8-sig'))
    eid = pred['experiment_id']; old = previous[eid]
    assert old['response']['experiment_id'] == eid
    assert old['directory'].endswith('/'+rel), 'Directory mismatch'
    row = {'dataset': rel.split('/')[0], 'repeat': int(rel.split('/')[1].split('_')[1]), 'condition': rel.split('/')[2], 'experiment_id': eid, 'directory': rel}
    for split in ['val','test']:
        truth_rows = json.loads((SOURCE/rel/f'{split}_labels.json').read_text(encoding='utf-8-sig'))
        input_rows = json.loads((SOURCE/rel/f'{split}.json').read_text(encoding='utf-8-sig'))
        ids = [r['id'] for r in input_rows]
        assert len(set(ids)) == len(ids)
        truth = {r['id']:r['label'] for r in truth_rows}
        assert set(truth) == set(ids) and len(truth_rows) == len(ids)
        for model, response in [('current',pred),('previous',old['response'])]:
            predictions = response[split]
            assert [r['id'] for r in predictions] == ids, f'{eid} {model} {split} ID mismatch'
            correct = sum(r['label'] == truth[r['id']] for r in predictions)
            row[f'{split}_{model}_correct'] = correct
            row[f'{split}_{model}_accuracy_pct'] = 100*correct/len(ids)
            if model=='previous':
                assert old[f'{split}_metrics']['correct']==correct
                assert old[f'{split}_metrics']['total']==len(ids)
        row[f'{split}_n'] = len(ids)
        row[f'{split}_delta_pp'] = row[f'{split}_current_accuracy_pct']-row[f'{split}_previous_accuracy_pct']
        for a,b in zip(pred[split], old['response'][split]):
            paired.append({'dataset':row['dataset'],'condition':row['condition'],'split':split,'current_correct':a['label']==truth[a['id']],'previous_correct':b['label']==truth[b['id']]})
    row['combined_n']=row['val_n']+row['test_n']
    for model in ['current','previous']:
        row[f'combined_{model}_correct']=row[f'val_{model}_correct']+row[f'test_{model}_correct']
        row[f'combined_{model}_accuracy_pct']=100*row[f'combined_{model}_correct']/row['combined_n']
    row['combined_delta_pp']=row['combined_current_accuracy_pct']-row['combined_previous_accuracy_pct']
    rows.append(row)
summary=[]
for group,key_fn in [('overall',lambda x:'all'),('dataset',lambda x:x['dataset']),('condition',lambda x:x['condition']),('dataset_condition',lambda x:(x['dataset'],x['condition']))]:
    groups=defaultdict(list)
    for row in rows: groups[key_fn(row)].append(row)
    for value,items in sorted(groups.items(),key=lambda x:str(x[0])):
        for split in ['val','test','combined']:
            n=sum(x[f'{split}_n'] for x in items)
            cur=sum(x[f'{split}_current_correct'] for x in items)
            prev=sum(x[f'{split}_previous_correct'] for x in items)
            summary.append({'group':group,'value':' | '.join(value) if isinstance(value,tuple) else value,'split':split,'experiments':len(items),'n':n,'current_correct':cur,'current_accuracy_pct':100*cur/n,'previous_correct':prev,'previous_accuracy_pct':100*prev/n,'delta_pp':100*(cur-prev)/n})
for filename,data in [('accuracy_by_experiment.csv',rows),('accuracy_comparison_summary.csv',summary)]:
    with (OUT/filename).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
assert frozen=={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in pred_paths}, 'Predictions changed during scoring'
score_manifest={'datasets':DATASETS,'experiments_scored':60,'prediction_files_unchanged':True,'prediction_sha256':frozen,'prior_prediction_file':str(prior_path),'label_source':['val_labels.json','test_labels.json'],'matching':'exact experiment_id, directory and ordered input sample IDs; exact string label match','previous_scores_recomputed_and_matched_cached_counts':True}
(OUT/'scoring_manifest.json').write_text(json.dumps(score_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# Public：前三个数据集准确率与历史结果比较','','本轮使用独立上下文预测；历史结果为 four_batches_strata_rerun 中的本地 qwen3.8-flash-next-q2_0（thinking=false）。60套实验及输入样本ID完全对应；预测冻结后读取答案评分。','','## 测试集','','| 数据集 | 历史本地Qwen | 本轮 | 差值（百分点） |','|---|---:|---:|---:|']
for r in summary:
    if r['group']=='dataset' and r['split']=='test':lines.append(f"| {r['value']} | {r['previous_correct']}/{r['n']} = {r['previous_accuracy_pct']:.2f}% | {r['current_correct']}/{r['n']} = {r['current_accuracy_pct']:.2f}% | {r['delta_pp']:+.2f} |")
lines+=['','## 分条件测试准确率','','| 数据集 | 条件 | 历史本地Qwen | 本轮 | 差值（百分点） |','|---|---|---:|---:|---:|']
for r in summary:
    if r['group']=='dataset_condition' and r['split']=='test':
        ds,cond=r['value'].split(' | ')
        lines.append(f"| {ds} | {cond} | {r['previous_accuracy_pct']:.2f}% | {r['current_accuracy_pct']:.2f}% | {r['delta_pp']:+.2f} |")
lines+=['','每个数据集有20套条件（5次划分 × 4种参考条件）。相同样本可能在不同条件或划分中重复出现；这里的分母统计预测记录次数，不代表互不重叠的独立样本。','']
(OUT/'accuracy_report.md').write_text('\n'.join(lines),encoding='utf-8')
manifest['status']='waiting_for_user_after_fresh_meats'
manifest['scoring_performed']=True
manifest['scored_datasets']=DATASETS
manifest['scoring_artifacts']=['accuracy_by_experiment.csv','accuracy_comparison_summary.csv','accuracy_report.md','scoring_manifest.json']
(OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in summary:
    if r['group'] in ['overall','dataset','dataset_condition'] and r['split'] in ['val','test']:print(json.dumps(r))
