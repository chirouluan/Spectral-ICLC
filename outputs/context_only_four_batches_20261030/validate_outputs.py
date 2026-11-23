"""Structural checks only. Never reads ground truth or computes predictions."""
from pathlib import Path
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
DATA = Path(manifest['source_directory'])
results = []
for rel in manifest['experiments']:
    path = OUT / rel / 'predictions.json'
    if not path.exists():
        results.append({'directory': rel, 'status': 'pending'})
        continue
    row = {'directory': rel, 'status': 'valid'}
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        prompt = (DATA / rel / 'prompt.txt').read_text(encoding='utf-8-sig')
        eid = re.search(r'实验编号：\s*(\S+)', prompt).group(1)
        allowed = json.loads(re.search(r'允许的label仅为以下字符串：(\[[^\n]+\])', prompt).group(1))
        assert data['experiment_id'] == eid, 'experiment_id mismatch'
        assert set(data) == {'experiment_id', 'val', 'test'}, 'unexpected top-level keys'
        row['experiment_id'] = eid
        for split in ['val', 'test']:
            inputs = json.loads((DATA / rel / f'{split}.json').read_text(encoding='utf-8-sig'))
            predictions = data[split]
            assert isinstance(predictions, list), f'{split}: not an array'
            assert [x['id'] for x in inputs] == [x['id'] for x in predictions], f'{split}: IDs/count/order mismatch'
            assert len({x['id'] for x in predictions}) == len(predictions), f'{split}: duplicate ID'
            assert all(set(x) == {'id', 'label'} and x['label'] in allowed for x in predictions), f'{split}: invalid label/schema'
            row[f'n_{split}'] = len(predictions)
        row['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        row['input_sha256'] = {
            name: hashlib.sha256((DATA / rel / name).read_bytes()).hexdigest()
            for name in manifest['allowed_source_files']
        }
    except Exception as exc:
        row['status'] = 'invalid'
        row['error'] = f'{type(exc).__name__}: {exc}'
    results.append(row)
counts = {s: sum(r['status'] == s for r in results) for s in ['valid', 'invalid', 'pending']}
report = {'checks': 'schema, experiment_id, exact input IDs/order/count, uniqueness, allowed labels; no accuracy scoring',
          'counts': counts, 'results': results}
(OUT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
manifest['status'] = 'completed' if counts['valid'] == manifest['n_expected'] else 'running'
manifest['counts'] = counts
manifest['completed_experiments'] = counts['valid']
manifest['pending_experiments'] = counts['pending']
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(counts))
