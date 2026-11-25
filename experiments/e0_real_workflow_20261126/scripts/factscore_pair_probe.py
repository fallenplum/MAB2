from pathlib import Path
import hashlib,json,re,itertools
root=Path('experiments/e0_real_workflow_20261126')
sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text(encoding='utf-8')); ids={x['group_id'] for x in sel['groups']}; salt=sel['salt']
def gid(t): return 'e0g_'+hashlib.sha256((salt+'|'+t).encode()).hexdigest()[:16]
def load(name):
 out={}
 for line in (root/'external/factscore_official/data/data/labeled'/name).open(encoding='utf-8'):
  row=json.loads(line); g=gid(row['topic'])
  if g not in ids: continue
  ann=row.get('annotations') or []
  out[g]=[f['text'].strip() for a in ann for f in a.get('model-atomic-facts',[])]
 return out
sets={n:load(n) for n in ['InstructGPT.jsonl','ChatGPT.jsonl','PerplexityAI.jsonl']}
for a,b in itertools.combinations(sets,2):
 raw=sum(len(set(sets[a][g])&set(sets[b][g])) for g in ids)
 old=sum(len(sets[a][g]) for g in ids); new=sum(len(sets[b][g]) for g in ids)
 print(a,b,'raw_shared',raw,'counts',old,new)

