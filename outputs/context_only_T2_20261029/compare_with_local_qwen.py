import json,csv
from pathlib import Path
from collections import defaultdict
root=Path(r'C:\Users\35454\Desktop\llm-dl\Spectral-ICLC-local-llm_20261028\Spectral-ICLC-local-llm')
out=root/'outputs'/'context_only_T2_20261029'; src=root/'data_json'/'four_batches'/'batch_02_T2'
oldfile=root/'outputs'/'four_batches_strata_rerun'/'predictions.jsonl'
def records(path):
 x=json.loads(path.read_text(encoding='utf-8'))
 if isinstance(x,list): return x
 raise ValueError(path)
old={}
for line in oldfile.open(encoding='utf-8'):
 x=json.loads(line)
 if x.get('task')!='T2': continue
 key=x['directory'].split('T2/',1)[1]
 key='T2/'+key
 if key in old: raise ValueError('duplicate old key '+key)
 old[key]=x
new={}
for rel in json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))['experiments']:
 key=rel.split('T2/',1)[1]; key='T2/'+key
 new[key]=(rel,json.loads((out/Path(rel)/'predictions.json').read_text(encoding='utf-8')))
if len(old)!=60 or len(new)!=60 or set(old)!=set(new): raise ValueError(f'key mismatch old={len(old)} new={len(new)}')
rows=[]; pcounts=defaultdict(lambda:[0,0,0,0,0])
for key in sorted(new):
 rel,pred=new[key]; prior=old[key]
 direction=rel.split('/')[1]; condition=rel.split('/')[-1]; repeat=int(rel.split('repeat_')[1].split('/')[0])
 row={'direction':direction,'condition':condition,'repeat':repeat}
 for split in ('val','test'):
  truth={r['id']:str(r['label']) for r in records(src/Path(rel)/(split+'_labels.json'))}
  current={r['id']:str(r['label']) for r in pred[split]}; previous={r['id']:str(r['label']) for r in prior['response'][split]}
  if set(truth)!=set(current) or set(truth)!=set(previous): raise ValueError(f'IDs do not match for {key} {split}')
  c=sum(current[i]==truth[i] for i in truth); o=sum(previous[i]==truth[i] for i in truth)
  row[f'{split}_n']=len(truth);row[f'{split}_current_correct']=c;row[f'{split}_previous_correct']=o
  row[f'{split}_current_accuracy']=c/len(truth);row[f'{split}_previous_accuracy']=o/len(truth);row[f'{split}_delta_pp']=(c-o)/len(truth)*100
  for i in truth:
   pc=pcounts[(direction,condition,split)]; pc[3]+=1
   if current[i]==truth[i] and previous[i]!=truth[i]:pc[0]+=1
   elif previous[i]==truth[i] and current[i]!=truth[i]:pc[1]+=1
   elif current[i]==truth[i] and previous[i]==truth[i]:pc[2]+=1
   else:pc[4]+=1
 row['combined_n']=row['val_n']+row['test_n']; row['combined_current_correct']=row['val_current_correct']+row['test_current_correct'];row['combined_previous_correct']=row['val_previous_correct']+row['test_previous_correct'];row['combined_current_accuracy']=row['combined_current_correct']/row['combined_n'];row['combined_previous_accuracy']=row['combined_previous_correct']/row['combined_n'];row['combined_delta_pp']=(row['combined_current_correct']-row['combined_previous_correct'])/row['combined_n']*100
 rows.append(row)
with (out/'paired_comparison_by_experiment.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for grp,keyfn in [('overall',lambda x:'all'),('direction',lambda x:x['direction']),('condition',lambda x:x['condition']),('direction_condition',lambda x:(x['direction'],x['condition']))]:
 groups=defaultdict(list)
 for r in rows:groups[keyfn(r)].append(r)
 for key,items in groups.items():
  for split in ('val','test','combined'):
   n=sum(x[f'{split}_n'] for x in items); c=sum(x[f'{split}_current_correct'] for x in items);o=sum(x[f'{split}_previous_correct'] for x in items)
   summary.append({'group':grp,'value':' | '.join(key) if isinstance(key,tuple) else key,'split':split,'experiments':len(items),'n':n,'current_correct':c,'current_accuracy':c/n,'previous_correct':o,'previous_accuracy':o/n,'delta_pp':(c-o)/n*100})
with (out/'paired_comparison_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
# sample-level paired counts for all, direction, condition and direction-condition
pairs=[]
for (direction,condition,split),z in pcounts.items():pairs.append({'direction':direction,'condition':condition,'split':split,'current_only_correct':z[0],'previous_only_correct':z[1],'both_correct':z[2],'n':z[3],'both_wrong':z[4]})
with (out/'paired_sample_outcomes.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
print('exact pairings',len(rows),'experiments, sample outcomes',sum(x['test_n'] for x in rows),'test +',sum(x['val_n'] for x in rows),'val')
for r in summary:
 if r['group']=='overall' and r['split'] in ('val','test','combined'):print(r)
for r in summary:
 if r['group']=='direction_condition' and r['split']=='test': print(r['value'],r['previous_correct'],r['n'],f"{r['previous_accuracy']*100:.2f}%",'->',r['current_correct'],f"{r['current_accuracy']*100:.2f}%",f"({r['delta_pp']:+.2f} pp)")
