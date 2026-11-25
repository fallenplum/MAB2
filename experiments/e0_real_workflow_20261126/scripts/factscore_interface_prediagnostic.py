from pathlib import Path
import hashlib,json,re
root=Path('experiments/e0_real_workflow_20261126')
sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text(encoding='utf-8'))
ids={x['group_id'] for x in sel['groups']}
salt=sel['salt']
def gid(topic): return 'e0g_'+hashlib.sha256((salt+'|'+topic).encode()).hexdigest()[:16]
def load(name):
 out={}
 for line in (root/'external/factscore_official/data/data/labeled'/name).open(encoding='utf-8'):
  row=json.loads(line); g=gid(row['topic'])
  if g not in ids: continue
  facts=[f['text'].strip() for a in (row.get('annotations') or []) for f in a.get('model-atomic-facts',[])]
  out[g]=facts
 return out
old,new=load('InstructGPT.jsonl'),load('ChatGPT.jsonl')
def canonical(x): return re.sub(r'\s+',' ',x.strip().lower())
rows=[]
for g in sorted(ids):
 a,b=old[g],new[g]; aset,bset=set(a),set(b); ac,bc={canonical(x) for x in a},{canonical(x) for x in b}
 rows.append({'group_id':g,'old_facts':len(a),'new_facts':len(b),'raw_shared':len(aset&bset),'raw_old_only':len(aset-bset),'raw_new_only':len(bset-aset),'canonical_shared':len(ac&bc),'canonical_old_only':len(ac-bc),'canonical_new_only':len(bc-ac)})
summary={'schema':'mab2_e0_interface_prediagnostic_v1','status':'DEVELOPMENT_PRETRACE_ONLY','groups':len(rows),'rows':rows,'totals':{k:sum(r[k] for r in rows) for k in rows[0] if k not in {'group_id'}},'warning':'No labels, retrieval outcomes, semantic judgments, or service costs used.'}
(root/'results/E0_INTERFACE_PREDIAGNOSTIC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary['totals'],ensure_ascii=False,indent=2))


