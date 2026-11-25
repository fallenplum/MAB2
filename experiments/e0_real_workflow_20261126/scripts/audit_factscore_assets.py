from pathlib import Path
import json, hashlib
root=Path('experiments/e0_real_workflow_20261126/external/factscore_official')
report={'files':{},'rows':{}}
for path in sorted((root/'data').rglob('*')) if (root/'data').exists() else []:
    if path.is_file():
        report['files'][str(path.relative_to(root)).replace('\\','/')]= {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix=='.jsonl':
            rows=[]
            with path.open(encoding='utf-8') as f:
                for idx,line in enumerate(f):
                    if idx<2:
                        rows.append(json.loads(line))
            report['rows'][str(path.relative_to(root)).replace('\\','/')]= {'count':sum(1 for _ in path.open(encoding='utf-8')),'sample_keys':sorted(rows[0].keys()) if rows else [],'sample':rows[:1]}
(root/'DATA_ASSET_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report['rows'].items()},ensure_ascii=False,indent=2))

