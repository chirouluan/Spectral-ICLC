import json, csv, re
from pathlib import Path
from collections import defaultdict

root = Path(r'C:\Users\35454\Desktop\llm-dl\Spectral-ICLC-local-llm_20261028\Spectral-ICLC-local-llm')
out = root / 'outputs' / 'context_only_T2_20261029'
src = root / 'data_json' / 'four_batches' / 'batch_02_T2'
manifest = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
rows=[]

def load_json(path):
    x=json.loads(path.read_text(encoding='utf-8'))
    return x

def clean_records(x, key):
    recs=x[key] if isinstance(x,dict) and key in x else x
    if not isinstance(recs,list): raise ValueError(f'{key} is not list')
    result={}
    for r in recs:
        if not isinstance(r,dict) or 'id' not in r or 'label' not in r: raise ValueError(f'bad {key} record')
        if r['id'] in result: raise ValueError(f'duplicate id {r["id"]}')
        result[r['id']]=str(r['label'])
    return result

for rel in manifest['experiments']:
    pred_path=out / Path(rel) / 'predictions.json'
    data_path=src / Path(rel)
    pred=load_json(pred_path)
    truth={split:clean_records(load_json(data_path/f'{split}_labels.json'), split) for split in ('val','test')}
    for split in ('val','test'):
        pr=clean_records(pred,split)
        if set(pr)!=set(truth[split]):
            raise ValueError(f'ID mismatch {rel} {split}: pred={len(pr)} truth={len(truth[split])}')
        n=len(pr); correct=sum(pr[i]==truth[split][i] for i in pr)
        rows.append({'experiment_id':pred['experiment_id'],'condition':rel.split('/')[-1], 'direction':rel.split('/')[1], 'repeat':int(re.search(r'repeat_(\d+)',rel).group(1)), 'split':split, 'n':n, 'correct':correct, 'accuracy':correct/n})

if len(rows)!=120: raise ValueError(f'expected 120 score rows, got {len(rows)}')
byexp=defaultdict(dict)
for r in rows: byexp[(r['direction'],r['repeat'],r['condition'])][r['split']]=r
exp_rows=[]
for (direction,repeat,condition),splits in sorted(byexp.items()):
    v,t=splits['val'],splits['test']
    exp_rows.append({'direction':direction,'repeat':repeat,'condition':condition,'val_n':v['n'],'val_correct':v['correct'],'val_accuracy':v['accuracy'],'test_n':t['n'],'test_correct':t['correct'],'test_accuracy':t['accuracy'],'combined_n':v['n']+t['n'],'combined_correct':v['correct']+t['correct'],'combined_accuracy':(v['correct']+t['correct'])/(v['n']+t['n'])})

# Pool cases and also retain the across-experiment mean. All experiments have equal split sizes.
def summarize(group_name, group_key):
    result=[]
    groups=defaultdict(list)
    for r in exp_rows: groups[group_key(r)].append(r)
    for k,items in sorted(groups.items(),key=lambda z: str(z[0])):
        for split in ('val','test','combined'):
            n=sum(x[f'{split}_n'] if split!='combined' else x['combined_n'] for x in items)
            cor=sum(x[f'{split}_correct'] if split!='combined' else x['combined_correct'] for x in items)
            acc=cor/n
            mean=sum(x[f'{split}_accuracy'] if split!='combined' else x['combined_accuracy'] for x in items)/len(items)
            keys=k if isinstance(k,tuple) else (k,)
            result.append({'group':group_name,'value':' | '.join(map(str,keys)),'split':split,'experiments':len(items),'n':n,'correct':cor,'accuracy':acc,'mean_experiment_accuracy':mean})
    return result

summaries=[]
summaries += summarize('overall',lambda x:'all')
summaries += summarize('direction',lambda x:x['direction'])
summaries += summarize('condition',lambda x:x['condition'])
summaries += summarize('direction_condition',lambda x:(x['direction'],x['condition']))

with (out/'accuracy_by_experiment.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(exp_rows[0])); w.writeheader(); w.writerows(exp_rows)
with (out/'accuracy_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
report={'n_experiments':len(exp_rows),'n_val':sum(x['val_n'] for x in exp_rows),'n_test':sum(x['test_n'] for x in exp_rows),'scoring':'exact string label match by record ID; val/test truth labels read from source JSON','overall':{},'by_direction_condition':[]}
for split in ('val','test','combined'):
    n=sum(x[f'{split}_n'] if split!='combined' else x['combined_n'] for x in exp_rows)
    c=sum(x[f'{split}_correct'] if split!='combined' else x['combined_correct'] for x in exp_rows)
    report['overall'][split]={'correct':c,'total':n,'accuracy':c/n}
for d in sorted({x['direction'] for x in exp_rows}):
  for cond in ('full_reference','k_1','k_3','k_5'):
    items=[x for x in exp_rows if x['direction']==d and x['condition']==cond]
    report['by_direction_condition'].append({'direction':d,'condition':cond,'experiments':len(items),'val_accuracy':sum(x['val_correct'] for x in items)/sum(x['val_n'] for x in items),'test_accuracy':sum(x['test_correct'] for x in items)/sum(x['test_n'] for x in items),'combined_accuracy':sum(x['combined_correct'] for x in items)/sum(x['combined_n'] for x in items),'combined_correct':sum(x['combined_correct'] for x in items),'combined_total':sum(x['combined_n'] for x in items)})
(out/'accuracy_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))


