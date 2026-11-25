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
run=json.loads((root/'results/E0_FLAT_32.json').read_text(encoding='utf-8'))
rows=[r for r in run['rows'] if r['status']!='INVALID_NO_FROZEN_FACTS' and truth.get(r['group_id']) is not None]
result={'schema':'mab2_e0_flat_posthoc_v1','status':'DEVELOPMENT_ONLY','groups':len(rows),'input_tokens':sum(r['input_tokens'] for r in run['rows']),'output_tokens':sum(r['output_tokens'] for r in run['rows']),'decision_errors':sum(r['decision']!=truth[r['group_id']] for r in rows),'invalid_groups':sum(r['status']=='INVALID_NO_FROZEN_FACTS' for r in run['rows']),'truth_comparable_groups':len(rows),'warning':'Group-level flat RAG baseline is development-only and uses fixed human fact text without online labels.'}
(root/'results/E0_FLAT_32_POSTHOC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

