from pathlib import Path
import json, hashlib
root=Path('experiments/e0_real_workflow_20261126/external/factscore_official/data/data/labeled')
for path in sorted(root.glob('*.jsonl')):
    topics=[]
    for line in path.open(encoding='utf-8'):
        obj=json.loads(line)
        topics.append(obj.get('topic'))
    print(path.name, len(topics), len(set(topics)), topics[:3])

