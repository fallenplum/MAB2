"""Frozen Formal10 input-only partition and visible/hidden case separation."""
from pathlib import Path
import argparse,hashlib,json,collections

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 a=argparse.ArgumentParser();a.add_argument('--eligible',type=Path,required=True);a.add_argument('--audit',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--tokenizers',nargs=2,required=True);a=a.parse_args()
 audit=json.loads(a.audit.read_bytes());assert audit['status']=='PASS_INPUT_FEASIBILITY_ONLY' and audit['eligible_independent_groups']==1496
 rows=json.loads(a.eligible.read_bytes());groups=collections.defaultdict(list)
 for r in rows:groups[r['group']].append(r)
 assert len(rows)==1501 and len(groups)==1496
 order=sorted(groups,key=lambda g:(h('MAB2-F10-function-partition-v1|'+g),g));parts={'pilot':order[:16],'development':order[16:64],'confirmation':order[64:192],'reserve':order[192:]}
 from transformers import AutoTokenizer
 toks=[AutoTokenizer.from_pretrained(p,local_files_only=True) for p in a.tokenizers]
 a.out.mkdir(parents=True,exist_ok=True);counts={};private={};inputs={};manifest={}
 for kind,gs in parts.items():
  private[kind]=[];inputs[kind]=[];manifest[kind]=[]
  for i,g in enumerate(gs):
   r=min(groups[g],key=lambda r:(h('MAB2-F10-task-v1|'+str(r['id'])),r['id']));pairs=r['pairs']
   hidden_order=sorted(r['hidden_indices'],key=lambda j:(h('MAB2-F10-hidden-v1|'+pairs[j][0]+'|'+pairs[j][1]),j));anchors=set(hidden_order[:2]);assert len(anchors)==2
   visible=sorted((j for j in range(len(pairs)) if j not in anchors),key=lambda j:(h('MAB2-F10-visible-v1|'+pairs[j][0]+'|'+pairs[j][1]),j))[:2]
   hidden=[j for j in hidden_order if j not in visible][:20];assert len(visible)==2 and 2<=len(hidden)<=20 and not set(visible)&set(hidden)
   payload=dict(id='apps:'+str(r['id']),question=r['question'],function_name=r['function_name'],visible_cases=[pairs[j] for j in visible]);assert set(payload)=={'id','question','function_name','visible_cases'}
   # Full question is never truncated. Notes/code/feedback capacity is dynamically rechecked at every request.
   lengths=[len(t.encode(r['question'],add_special_tokens=False)) for t in toks];assert max(lengths)<=4096
   inputs[kind].append(payload);private[kind].append(dict(id=payload['id'],group=g,visible_indices=visible,hidden_indices=hidden,hidden_cases=[pairs[j] for j in hidden],all_pair_count=len(pairs)))
   manifest[kind].append(dict(index=i,id=payload['id'],group=g,input_sha256=h(canonical(payload)+'\n'),removed_upstream='U' if i%2==0 else 'V',question_tokens=lengths,hidden_case_count=len(hidden)))
  counts[kind]=dict(tasks=len(gs),hidden_cases=sum(x['hidden_case_count'] for x in manifest[kind]),min_hidden=min(x['hidden_case_count'] for x in manifest[kind]),max_hidden=max(x['hidden_case_count'] for x in manifest[kind]))
  with (a.out/(kind+'_INPUTS.jsonl')).open('x',encoding='utf-8') as f:
   for x in inputs[kind]:f.write(canonical(x)+'\n')
  write(a.out/(kind+'_HIDDEN.json'),private[kind]);write(a.out/(kind+'_SELECTION.json'),dict(questions=manifest[kind]))
 write(a.out/'PARTITION.json',dict(groups=parts))
 receipt=dict(status='PASS_FROZEN_INPUT_PARTITION',eligible_sha256=sha(a.eligible),audit_sha256=sha(a.audit),source_sha256=sha(__file__),counts=counts,files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},model_calls=0,GPU_hours=0,hidden_labels_in_model_inputs=False,pretraining_independence_claimed=False)
 write(a.out/'INPUT_RECEIPT.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
if __name__=='__main__':main()
