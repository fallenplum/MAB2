from pathlib import Path
import json,hashlib,math
root=Path('experiments/e0_real_workflow_20261126'); sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text()); salt=sel['salt']; ids={x['group_id'] for x in sel['groups']}
def gid(t): return 'e0g_'+hashlib.sha256((salt+'|'+t).encode()).hexdigest()[:16]
truth={}
for line in (root/'external/factscore_official/data/data/labeled/ChatGPT.jsonl').open(encoding='utf-8'):
 row=json.loads(line); g=gid(row['topic'])
 if g not in ids: continue
 labs=[f.get('label') for a in (row.get('annotations') or []) for f in ((a or {}).get('human-atomic-facts') or []) if f.get('label') in {'S','NS','IR'}]
 truth[g]='PASS' if labs and sum(x=='S' for x in labs)>=math.ceil(.8*len(labs)) else 'FAIL' if labs else None
run=json.loads((root/'results/E0_FULL_32.json').read_text(encoding='utf-8'))
for name in ['old_decisions','new_cold_decisions','new_warm_decisions']:
 d=run[name]; valid=[g for g in d if d[g]['status']!='INVALID' and truth.get(g) is not None]; print(name,'errors',sum(d[g]['decision']!=truth[g] for g in valid),'valid',len(valid))

