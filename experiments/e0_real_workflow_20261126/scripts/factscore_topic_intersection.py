from pathlib import Path
import json
root=Path('experiments/e0_real_workflow_20261126/external/factscore_official/data/data/labeled')
topics={}
for p in root.glob('*.jsonl'):
    topics[p.stem]={json.loads(l)['topic'] for l in p.open(encoding='utf-8')}
for a in topics:
  for b in topics:
    if a<b: print(a,b,len(topics[a]&topics[b]))

