"""Scalar replay of every D17 choice, outcome and comparator."""
from pathlib import Path
import json,hashlib
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);args=ap.parse_args()
p=Path(args.private_root);out=p/'d17_legal_decisions_v1';summary=json.loads((out/'SUMMARY.json').read_bytes());a=np.load(p/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');checks=0
def same(x,y):
 global checks
 assert abs(x-y)<1e-10,(x,y);checks+=1
for task,reported in summary['tasks'].items():
 em=a[task+'_em'];tok=a[task+'_tokens'];shock=a[task+'_shock'];halves=[]
 for ts in [[0,1],[2,3]]:
  halves.append([[sum(float(em[i,s,t,g])-5e-6*float(tok[i,s,t,g]) for s in range(2) for t in ts)/4 for g in range(8)] for i in range(96)])
 gains=[]
 for result in reported['transfers']:
  tr=result['selection_half'];te=result['evaluation_half'];train=halves[tr];test=halves[te];fresh=[];stale=[];sub=[];policies=[];base=[]
  for i,h in enumerate(shock):
   b=7^(1<<int(h));base.append(b);edges=[e for e in range(3) if b&(1<<e)];legal=[b]+[b^(1<<e) for e in edges];old=[7]+[7^(1<<e) for e in edges];fresh.append(max(legal,key=lambda g:train[i][g]));stale.append(legal[max(range(3),key=lambda j:train[i][old[j]])]);sub.append(max([g for g in range(8) if g&b==g],key=lambda g:train[i][g]));policies.append([b]+[b&~(1<<e) for e in range(3)])
  means=[sum(test[i][policies[i][j]] for i in range(96))/96 for j in range(4)];best=max(range(4),key=lambda j:means[j]);fixed=[row[best] for row in policies];oof=[]
  for i in range(96):
   fit=[k for k in range(96) if k%2!=i%2];j=max(range(4),key=lambda j:sum(train[k][policies[k][j]] for k in fit));oof.append(policies[i][j])
  assert best==result['best_fixed_policy'];choices=dict(fresh=fresh,stale=stale,best_fixed=fixed,cross_question_fixed=oof,all_submask_oracle=sub,keep=base)
  for j,v in enumerate(means):same(v,result['fixed_policy_utilities'][j])
  for name,cs in choices.items():
   r=result['methods'][name];same(sum(test[i][cs[i]] for i in range(96))/96,r['utility']);outcomes=[];correct=destroy=0
   for i in range(96):
    assert cs[i]&base[i]==cs[i]
    for s in range(2):
     for t in ([0,1] if te==0 else [2,3]):
      v=int(em[i,s,t,cs[i]]);b=int(em[i,s,t,fixed[i]]);outcomes.append(v);correct+=v==1 and b==0;destroy+=v==0 and b==1
   same(sum(outcomes)/len(outcomes),r['EM']);same(correct,r['corrections_vs_fixed']);same(destroy,r['destructions_vs_fixed']);assert [cs.count(g) for g in range(8)]==r['mask_counts']
  dg=[sum(test[i][fresh[i]]-test[i][other[i]] for i in range(96))/96 for other in [stale,fixed]];same(dg[0],result['gain_vs_stale']);same(dg[1],result['gain_vs_best_fixed']);gains.append(dg)
 means=[sum(row[j] for row in gains)/2 for j in range(2)];same(means[0],reported['mean_gains']['gain_vs_stale']);same(means[1],reported['mean_gains']['gain_vs_best_fixed']);assert reported['necessary_headroom_passed']==all(means[j]>=.005 and min(row[j] for row in gains)>0 for j in range(2))
receipt=dict(status='PASS_INDEPENDENT_SCALAR_LEGAL_DECISION_AUDIT',numeric_checks=checks,all_choices_legal=True,summary_sha256=hashlib.sha256((out/'SUMMARY.json').read_bytes()).hexdigest(),new_model_calls=0)
with (out/'AUDIT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
