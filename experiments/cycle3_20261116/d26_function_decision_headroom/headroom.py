"""Conditional zero-GPU oracle bounds; never an implementable method or confirmation."""
from pathlib import Path
import json,hashlib,argparse,sys,importlib.util

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--scores',required=True);p.add_argument('--scorer',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.archive);scores=Path(a.scores);out=Path(a.out);summary=read(scores/'SUMMARY.json');assert summary['status']=='STOP_FORMAL10_APPLICABILITY_ROUTE','Conditional diagnostic only after preserved failure'
 assert sha(a.scorer)==summary['source_sha256'],'Parent cost source changed'
 spec=importlib.util.spec_from_file_location('fixed_service_cost',a.scorer);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 table=read(scores/'PRIVATE_DECISION_TABLE.json')['records'];index={(r['question'],r['seed'],r['condition']):r for r in table};arms=['G1','G1_MINUS_EU','G1_MINUS_EV'];records=[];choices=[]
 for f in sorted(root.glob('worker_*/unit_*/MATRIX.json')):
  m=read(f);q=m['question_fingerprint']
  for seed in [261117101,261117102,261117103]:
   raw={v['condition']:v for v in m['conditions'] if v['execution_seed']==seed}
   for frontier in ['G1_OBSERVED_ORACLE','FREE_PRESELECTION_ORACLE']:
    candidates=[]
    for i,arm in enumerate(arms):
     rows=[raw[arm]] if frontier=='FREE_PRESELECTION_ORACLE' or arm=='G1' else [raw['G1'],raw[arm]];c=mod.cost(rows);z=index[(q,seed,arm)];u=z['quality']-c['tokens']/200000-c['visible_execution_seconds']/10000;candidates.append(dict(question=q,seed=seed,frontier=frontier,arm=arm,quality=z['quality'],utility=u,order=i,**c))
    winner=max(candidates,key=lambda z:(z['utility'],-z['order']));records.append(winner);choices.extend(candidates)
 assert len(records)==96
 bounds={(z['question'],z['seed'],z['frontier']):z['utility'] for z in records}
 assert len(bounds)==96
 for q,s,frontier in bounds:
  assert bounds[(q,s,'FREE_PRESELECTION_ORACLE')]+1e-12>=bounds[(q,s,'G1_OBSERVED_ORACLE')],'Nested oracle bound violated'
 results={}
 for frontier in ['G1_OBSERVED_ORACLE','FREE_PRESELECTION_ORACLE']:
  rs=[z for z in records if z['frontier']==frontier];comparisons={}
  for control in ['SINGLE_Q','SINGLE_C','BOK3','BOK3_EARLY','REPAIR3']:
   pairs=[(z,index[(z['question'],z['seed'],control)]) for z in rs];comparisons[control]=dict(utility_difference=sum(x['utility']-y['utility'] for x,y in pairs)/48,quality_difference=sum(x['quality']-y['quality'] for x,y in pairs)/48,improved=sum(x['quality']>y['quality'] for x,y in pairs),harmed=sum(x['quality']<y['quality'] for x,y in pairs),utility_by_seed={str(s):sum(x['utility']-y['utility'] for x,y in pairs if x['seed']==s)/16 for s in [261117101,261117102,261117103]})
  gates=dict(mean_headroom=all(v['utility_difference']>=.02 for v in comparisons.values()),quality=all(v['quality_difference']>=-.02 for v in comparisons.values()),seed_consistency=all(v>=0 for k in ['BOK3_EARLY','REPAIR3'] for v in comparisons[k]['utility_by_seed'].values()),nonzero_credit_support=summary['credit']['nonzero_current_tasks']>=4)
  results[frontier]=dict(quality=sum(z['quality'] for z in rs)/48,utility=sum(z['utility'] for z in rs)/48,tokens=sum(z['tokens'] for z in rs)/48,comparisons=comparisons,gates=gates,all_gates=all(gates.values()),selected_action_counts={arm:sum(z['arm']==arm for z in rs) for arm in arms})
 decision='FREEZE_CURRENT_CODING_ACTION_SPACE'
 if results['G1_OBSERVED_ORACLE']['all_gates']:decision='POST_FEEDBACK_HEADROOM_ONLY_REQUIRES_NEW_OBSERVABLE_MECHANISM_AND_DATA'
 elif results['FREE_PRESELECTION_ORACLE']['all_gates']:decision='ONLY_PRETERMINAL_HEADROOM_REQUIRES_DIFFERENT_INFORMATION_TIMING'
 result=dict(status=decision,diagnostic='D26',conditional_parent='FORMAL10_FAILED_UNCHANGED',exploratory=True,oracle_gold_used_for_selection=True,independent_confirmation=False,implementable_method=False,effective_method=False,tasks=16,seeds=3,frontiers=results,model_calls=0,GPU_hours=0,confidence_intervals_computed=False,source_sha256=sha(__file__),scorer_sha256=sha(a.scorer),parent_summary_sha256=sha(scores/'SUMMARY.json'),limitation='Gold oracle optimistically grants free knowledge of all counterfactual success, charges only the stated required service calls and visible executions, and excludes the extra cost of acquiring that knowledge. Positive headroom is necessary opportunity only, not evidence of an observable policy or of novelty. Parent failure and sealed independent partitions remain unchanged.')
 out.mkdir(parents=True,exist_ok=False)
 for name,value in [('SUMMARY.json',result),('PRIVATE_ORACLE_CHOICES.json',dict(records=records,all_choices=choices))]:
  with (out/name).open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k not in ['frontiers','limitation']}))
if __name__=='__main__':main()
