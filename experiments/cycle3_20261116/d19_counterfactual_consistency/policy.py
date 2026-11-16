"""Gold-free policy replay: query only chosen continuations, charge all probes."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
from collections import Counter
import numpy as np

METHODS=['ADAPT_GRAPH','KEEP_CURRENT','SAME_GRAPH_SC','NO_NOTE_SC','NO_NOTE_SINGLE','FIXED','RANDOM_ORDER_GRAPH','ALL_GRAPH_VOTE']
def acquire_and_vote(schedule,metadata,query,normalize,early):
 seen=set();records=[];keys=[]
 for cell in schedule:
  job=metadata(cell)['job_id']
  if job in seen:continue
  seen.add(job);g=query(cell);answer=normalize(g['text']);key=answer if answer else 'EMPTY_REQUEST_'+job
  records.append(dict(mask=cell[0],terminal=cell[1],job_id=job,input_tokens=g['input_tokens'],output_tokens=g['output_tokens']));keys.append(key)
  if early and answer and Counter(keys)[key]>=2:break
 counts=Counter(keys);best=max(range(len(keys)),key=lambda k:counts[keys[k]])
 return records[best],records,keys[best]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--repo',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);r=Path(a.repo);out=Path(a.output);assert not out.exists();read=lambda f:json.loads(f.read_bytes());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert read(p/'d18_message_decisions_v1/AUDIT.json')['status']=='PASS_INDEPENDENT_D18_COST_CHOICE_AND_REFIT_AUDIT'
 path=r/'experiments/cycle2_20261113/hotpot_support/hotpot_support.py';spec=importlib.util.spec_from_file_location('hotpot',path);normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
 entries=read(p/'d16_crossed_input_v1/ENTRIES.json');new={v['job_id']:v['generation'] for f in (p/'d16_crossed_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};lookup={};generations={};shocks={};seedkeys=[26111671,26111672,261116161,261116162]
 for e in entries:
  task=e['task'];i=e['index'];s=[26111671,26111672].index(e['source_seed']);t=seedkeys.index(e['terminal_seed']);g=e['cached']['generation'] if e['cache_hit'] else new[e['job_id']];lookup[task,i,s,e['mask'],t]=dict(job_id=e['job_id'],input_tokens=g['input_tokens']);generations[e['job_id']]=g;shocks[task,i]=e['shock']
 features=np.load(p/'d18_features_v1/FEATURES.npz');fixed=np.load(p/'d18_message_decisions_v1/PRIVATE_PREDICTIONS.npz');records=[];equivalence=0
 for task in ['hotpot','2wiki']:
  prefix=features[task+'_prefix_tokens']
  for i in range(96):
   b=7^(1<<shocks[task,i]);edits=[b^(1<<e) for e in range(3) if b&(1<<e)]
   for s in range(2):
    for anchor in range(4):
     metadata=lambda cell:lookup[task,i,s,cell[0],cell[1]]
     query=lambda cell:generations[metadata(cell)['job_id']]
     ordered=sorted(edits,key=lambda m:(metadata((m,anchor))['input_tokens'],m));coin=int(hashlib.sha256(json.dumps(['D19',task,i,s,anchor],separators=(',',':')).encode()).hexdigest(),16)%2;random_order=edits[coin:]+edits[:coin];fixedmask=int(fixed[task+'_choices_fixed'][i,s]);schedules={'ADAPT_GRAPH':[(b,anchor)]+[(m,anchor) for m in ordered],'KEEP_CURRENT':[(b,anchor)],'SAME_GRAPH_SC':[(b,(anchor+k)%4) for k in range(3)],'NO_NOTE_SC':[(0,(anchor+k)%4) for k in range(3)],'NO_NOTE_SINGLE':[(0,anchor)],'FIXED':[(fixedmask,anchor)],'RANDOM_ORDER_GRAPH':[(b,anchor)]+[(m,anchor) for m in random_order],'ALL_GRAPH_VOTE':[(b,anchor)]+[(m,anchor) for m in ordered]};answers={}
     for method in METHODS:
      choice,queried,answer=acquire_and_vote(schedules[method],metadata,query,normalizer.normalize_answer,method!='ALL_GRAPH_VOTE');answers[method]=answer;prefix_cost=float(prefix[i,s,fixedmask]) if method=='FIXED' else (0 if method.startswith('NO_NOTE') else float(prefix[i,s,7]));total=prefix_cost+sum(q['input_tokens']+q['output_tokens'] for q in queried);assert all((q['mask']&b)==q['mask'] for q in queried);records.append(dict(task=task,index=i,source=s,anchor=anchor,method=method,choice=choice,queries=queried,prefix_tokens=prefix_cost,total_tokens=total))
     assert answers['ADAPT_GRAPH']==answers['ALL_GRAPH_VOTE'];assert answers['RANDOM_ORDER_GRAPH']==answers['ALL_GRAPH_VOTE'];equivalence+=1
 out.mkdir();(out/'PRIVATE_POLICY_TRACE.jsonl').write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in records),encoding='utf-8');receipt=dict(status='COMPLETE_GOLD_FREE_POLICY_REPLAY',service_simulations=1536,methods=METHODS,policy_records=len(records),early_full_vote_equivalence_checks=equivalence,trace_sha256=sha(out/'PRIVATE_POLICY_TRACE.jsonl'),source_sha256=sha(Path(__file__)),normalizer_sha256=sha(path),new_model_calls=0,new_GPU_hours=0,terminal_outputs_used_only_when_queried=True,service_gold_used=False)
 (out/'POLICY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
