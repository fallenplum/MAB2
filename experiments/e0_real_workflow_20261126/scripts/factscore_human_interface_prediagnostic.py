from pathlib import Path
import hashlib,json,re
root=Path('experiments/e0_real_workflow_20261126'); sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text()); ids={x['group_id'] for x in sel['groups']}; salt=sel['salt']
def gid(t): return 'e0g_'+hashlib.sha256((salt+'|'+t).encode()).hexdigest()[:16]
def load(name):
 out={}
 for line in (root/'external/factscore_official/data/data/labeled'/name).open(encoding='utf-8'):
  row=json.loads(line); g=gid(row['topic'])
  if g not in ids: continue
  facts=[f['text'].strip() for a in (row.get('annotations') or []) for f in ((a or {}).get('human-atomic-facts') or []) if isinstance(f,dict) and isinstance(f.get('text'),str)]
  out[g]=facts
 return out
old,new=load('InstructGPT.jsonl'),load('ChatGPT.jsonl')
def canon(x): return re.sub(r'\s+',' ',x.strip().lower())
rows=[]
for g in sorted(ids):
 a,b=old[g],new[g]; aset,bset=set(a),set(b); ac,bc={canon(x) for x in a},{canon(x) for x in b}
 rows.append({'group_id':g,'old_facts':len(a),'new_facts':len(b),'raw_shared':len(aset&bset),'canonical_shared':len(ac&bc),'old_missing_decomposition':len(a)==0,'new_missing_decomposition':len(b)==0})
summary={'schema':'mab2_e0_interface_prediagnostic_v2','status':'DEVELOPMENT_PRETRACE_ONLY','fact_source':'official_human_atomic_fact_text_only','update_manifest':'UPDATE_MANIFEST.json','groups':len(rows),'rows':rows,'totals':{k:sum(r[k] for r in rows if isinstance(r.get(k),int)) for k in ['old_facts','new_facts','raw_shared','canonical_shared']},'missing_groups':{'old':[r['group_id'] for r in rows if r['old_missing_decomposition']],'new':[r['group_id'] for r in rows if r['new_missing_decomposition']]},'warning':'No retrieval outcomes, semantic judgments, labels, or service costs used.'}
(root/'results/E0_INTERFACE_PREDIAGNOSTIC_HUMAN_TEXT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary['totals'],ensure_ascii=False,indent=2))

