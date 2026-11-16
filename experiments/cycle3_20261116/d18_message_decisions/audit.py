"""Independent scalar decision/cost replay plus two held-out-fold model refits."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import sklearn,torch,transformers
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d18_message_decisions_v1';fdir=p/'d18_features_v1';read=lambda f:json.loads(f.read_bytes());s=read(run/'SUMMARY.json');pred=np.load(run/'PRIVATE_PREDICTIONS.npz');features=np.load(fdir/'FEATURES.npz');raw=np.load(p/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');checks=0;start=time.perf_counter();folds=np.arange(96)%6
def eq(x,y):
 global checks
 assert abs(float(x)-float(y))<1e-10,(x,y);checks+=1
groups=read(fdir/'PRIVATE_GROUPS.json');assert len(set(groups['hotpot']+groups['2wiki']))==192
for task,report in s['tasks'].items():
 em=raw[task+'_em'];tokens=raw[task+'_tokens'];shock=raw[task+'_shock'];prefix=features[task+'_prefix_tokens'];inputtokens=features[task+'_terminal_input_tokens'];outcomes={};utilities={};costs={};choices={name:pred[task+'_choices_'+name] for name in report['methods']}
 for name,choice in choices.items():
  q=np.zeros((96,2));u=q.copy();c=q.copy()
  for i in range(96):
   b=7^(1<<int(shock[i]));legal=[b]+[b^(1<<e) for e in range(3) if b&(1<<e)]
   for source in range(2):
    g=int(choice[i,source]);assert g&b==g
    if name in ['question','message_ridge','message_trees']:
     pv=pred[task+'_predictions_'+name];expected=max(legal,key=lambda k:pv[i,source,k]-5e-6*inputtokens[i,source,k]);assert g==expected
    q[i,source]=sum(float(em[i,source,t,g]) for t in range(4))/4;c[i,source]=sum(float(tokens[i,source,t,g]) for t in range(4))/4
    if name in ['question','message_ridge','message_trees','oracle_paid']:c[i,source]+=float(prefix[i,source,7]-prefix[i,source,g])
    u[i,source]=q[i,source]-5e-6*c[i,source]
  outcomes[name]=q;utilities[name]=u;costs[name]=c;rr=report['methods'][name];eq(q.mean(),rr['EM']);eq(u.mean(),rr['utility']);eq(c.mean(),rr['service_tokens']);assert np.bincount(choice.ravel(),minlength=8).tolist()==rr['mask_counts']
  for j in range(2):eq(u[:,j].mean(),rr['utility_by_source'][j])
  for f in range(6):eq(u[folds==f].mean(),rr['utility_by_fold'][f])
 for record in report['folds']:
  f=record['fold'];train=[i for i in range(96) if i%6!=f];test=[i for i in range(96) if i%6==f];assert train==record['train_indices'] and test==record['test_indices'];scores=[]
  for mask in range(8):scores.append(sum(float(em[i,source,t,mask&(7^(1<<int(shock[i])))]-5e-6*tokens[i,source,t,mask&(7^(1<<int(shock[i])))]) for i in train for source in range(2) for t in range(4))/(len(train)*8))
  priority=max(range(8),key=lambda k:scores[k]);assert priority==record['fixed_priority']
  for mask,v in enumerate(scores):eq(v,record['training_fixed_utilities'][mask])
  mean_cost=[sum(float(tokens[i,source,t,g]) for i in train for source in range(2) for t in range(4))/(len(train)*8) for g in range(8)]
  for i in test:
   b=7^(1<<int(shock[i]));legal=[b]+[b^(1<<e) for e in range(3) if b&(1<<e)]
   for source in range(2):assert choices['fixed'][i,source]==priority&b;assert choices['question_preselected'][i,source]==max(legal,key=lambda k:pred[task+'_predictions_question'][i,source,k]-5e-6*mean_cost[k])
 for name in ['message_ridge','message_trees']:
  gates=[]
  for baseline,rr in report['comparisons'][name].items():
   delta=utilities[name]-utilities[baseline];eq(delta.mean(),rr['utility_gain']);fg=[float(delta[folds==f].mean()) for f in range(6)];sg=delta.mean(0)
   for x,y in zip(fg,rr['gain_by_fold']):eq(x,y)
   for x,y in zip(sg,rr['gain_by_source']):eq(x,y)
   correct=destroy=0
   for i in range(96):
    for source in range(2):
     for t in range(4):
      new=em[i,source,t,choices[name][i,source]];old=em[i,source,t,choices[baseline][i,source]];correct+=new==1 and old==0;destroy+=new==0 and old==1
   eq(correct,rr['corrections']);eq(destroy,rr['destructions']);gate=delta.mean()>=.005 and min(sg)>0 and sum(v>0 for v in fg)>=4;assert gate==rr['comparison_passed'];gates.append(gate)
  regret=utilities['oracle_paid']-utilities[name];selected=[]
  for source in range(2):
   ranked=sorted(range(96),key=lambda i:float(pred[task+'_margins_'+name][i,source]));selected.extend(regret[i,source] for i in ranked[:24])
  ratio=float(sum(selected)/len(selected)/regret.mean()) if regret.mean()>0 else None;rr=report['screens'][name];eq(ratio,rr['risk_enrichment']);assert rr['primary_passed']==all(gates);assert rr['targeting_passed']==(ratio is not None and ratio>=1.2)
 # Refit one complete held-out question fold independently using stored input features.
 train=folds!=0;test=~train;y=np.mean(em[train],axis=2).reshape(-1,8);scaler=StandardScaler().fit(features[task+'_numeric'][train].reshape(-1,15));xt=np.column_stack((features[task+'_message'][train].reshape(-1,1536),scaler.transform(features[task+'_numeric'][train].reshape(-1,15))));xe=np.column_stack((features[task+'_message'][test].reshape(-1,1536),scaler.transform(features[task+'_numeric'][test].reshape(-1,15))))
 with threadpool_limits(limits=2):
  for name,model,x,z in [('question',Ridge(alpha=10),features[task+'_question'][train].reshape(-1,384),features[task+'_question'][test].reshape(-1,384)),('message_ridge',Ridge(alpha=10),xt,xe),('message_trees',ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,max_features=1.,bootstrap=False,random_state=26111618,n_jobs=2),xt,xe)]:
   v=np.clip(model.fit(x,y).predict(z),0,1).reshape(-1,2,8);assert np.allclose(v,pred[task+'_predictions_'+name][test],atol=1e-12,rtol=0);checks+=v.size
assert s['primary_passing_predictors_both_tasks']==[] and s['targeting_eligible_both_tasks']==[]
receipt=dict(status='PASS_INDEPENDENT_D18_COST_CHOICE_AND_REFIT_AUDIT',numeric_checks=checks,complete_scalar_decision_and_cost_replay=True,refit_folds_per_task=1,refit_models_per_fold=3,question_groups=192,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__,transformers=transformers.__version__,cpu_wall_seconds=time.perf_counter()-start,summary_sha256=hashlib.sha256((run/'SUMMARY.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (run/'AUDIT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
