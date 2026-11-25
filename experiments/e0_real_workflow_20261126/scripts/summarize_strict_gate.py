from pathlib import Path
import json,hashlib,math
root=Path('experiments/e0_real_workflow_20261126'); sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text(encoding='utf-8')); ids={x['group_id'] for x in sel['groups']}; salt=sel['salt']
def gid(t): return 'e0g_'+hashlib.sha256((salt+'|'+t).encode()).hexdigest()[:16]
def labels(name):
 out={}
 for line in (root/'external/factscore_official/data/data/labeled'/name).open(encoding='utf-8'):
  row=json.loads(line); g=gid(row['topic'])
  if g not in ids: continue
  labs=[f.get('label') for a in (row.get('annotations') or []) for f in ((a or {}).get('human-atomic-facts') or []) if f.get('label') in {'S','NS','IR'}]
  out[g]=('PASS' if sum(x=='S' for x in labs)>=math.ceil(.8*len(labs)) else 'FAIL') if labs else None
 return out
truth=labels('ChatGPT.jsonl')
full=json.loads((root/'results/E0_FULL_32.json').read_text(encoding='utf-8'))
def score(path):
 o=json.loads((root/path).read_text(encoding='utf-8')); rows=o['rows']; valid=[x for x in rows if x['status']!='INVALID_NO_FROZEN_FACTS']; errors=sum(truth.get(x['group_id']) is not None and x['decision']!=truth[x['group_id']] for x in valid); return {'events':sum(len(x['events']) for x in rows),'input_tokens':sum(x['input_tokens'] for x in rows),'output_tokens':sum(x['output_tokens'] for x in rows),'invalid':sum(x['status']=='INVALID_NO_FROZEN_FACTS' for x in rows),'decision_errors':errors,'valid_groups':len(valid),'decisions':{x['group_id']:x['decision'] for x in rows}}
result={'schema':'mab2_e0_strict_gate_summary_v1','status':'DEVELOPMENT_ACTUAL_STRICT_GATE','cold':score('results/E0_STRICT_GATE_COLD_32.json'),'canonical':score('results/E0_STRICT_GATE_CANONICAL_32.json'),'full_cold_input_tokens':sum(x['input_tokens'] for x in full['new_cold_events']),'full_cold_events':len(full['new_cold_events']),'warning':'Development data only; no independent confirmation or strong flat/RAG baseline.'}
(root/'results/E0_STRICT_GATE_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['cold','canonical','full_cold_input_tokens','full_cold_events']},ensure_ascii=False))


