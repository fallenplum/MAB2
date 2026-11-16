"""Question-grouped fixed-predictor decision screen with paid observed prefixes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

def evaluate_task(features,em,tokens,shock):
 assert em.shape==tokens.shape==(96,2,4,8)
 target=em.mean(axis=2);nominal=tokens.mean(axis=2);prefix=features['prefix_tokens'];paid=nominal+prefix[:,:,7,None]-prefix
 assert np.all(paid>=nominal);nominal_u=target-5e-6*nominal;paid_u=target-5e-6*paid
 predictions={name:np.full((96,2,8),np.nan) for name in ['question','message_ridge','message_trees']};folds=np.arange(96)%6;base=7^(1<<shock);legal=np.array([[int(b)]+[int(b)^(1<<e) for e in range(3) if b&(1<<e)] for b in base]);choices={name:np.full((96,2),-1,dtype=int) for name in ['fixed','question_preselected']};fold_log=[];fit_seconds=0
 for fold in range(6):
  tr=folds!=fold;te=~tr;assert not set(np.where(tr)[0])&set(np.where(te)[0]);qtr=features['question'][tr].reshape(-1,384);qte=features['question'][te].reshape(-1,384);scaler=StandardScaler().fit(features['numeric'][tr].reshape(-1,15));xtr=np.concatenate([features['message'][tr].reshape(-1,1536),scaler.transform(features['numeric'][tr].reshape(-1,15))],axis=1);xte=np.concatenate([features['message'][te].reshape(-1,1536),scaler.transform(features['numeric'][te].reshape(-1,15))],axis=1);y=target[tr].reshape(-1,8)
  start=time.perf_counter()
  for name,model,x,z in [('question',Ridge(alpha=10),qtr,qte),('message_ridge',Ridge(alpha=10),xtr,xte),('message_trees',ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,max_features=1.,bootstrap=False,random_state=26111618,n_jobs=2),xtr,xte)]:
   model.fit(x,y);predictions[name][te]=np.clip(model.predict(z).reshape(-1,2,8),0,1)
  fit_seconds+=time.perf_counter()-start
  static_values=[np.mean([nominal_u[i,s,int(mask)&int(base[i])] for i in np.where(tr)[0] for s in range(2)]) for mask in range(8)];best=int(np.argmax(static_values));choices['fixed'][te]=(base[te]&best)[:,None];costs=nominal[tr].mean(axis=(0,1))
  for i in np.where(te)[0]:
   for s in range(2):choices['question_preselected'][i,s]=legal[i,np.argmax(predictions['question'][i,s,legal[i]]-5e-6*costs[legal[i]])]
  fold_log.append(dict(fold=fold,train_question_count=int(tr.sum()),test_question_count=int(te.sum()),fixed_priority=best,training_fixed_utilities=[float(x) for x in static_values],train_indices=np.where(tr)[0].tolist(),test_indices=np.where(te)[0].tolist()))
 margins={}
 for name,pred in predictions.items():
  assert np.isfinite(pred).all();choice=np.empty((96,2),dtype=int);margin=np.empty((96,2))
  for i in range(96):
   for s in range(2):
    vals=pred[i,s,legal[i]]-5e-6*features['terminal_input_tokens'][i,s,legal[i]];choice[i,s]=legal[i,np.argmax(vals)];sortedvals=np.sort(vals);margin[i,s]=sortedvals[-1]-sortedvals[-2]
  choices[name]=choice;margins[name]=margin
 choices['single']=np.zeros((96,2),dtype=int);choices['oracle_paid']=np.array([[legal[i,np.argmax(paid_u[i,s,legal[i]])] for s in range(2)] for i in range(96)])
 qidx=np.arange(96)[:,None];sidx=np.arange(2)[None,:];values={};public={}
 for name,c in choices.items():
  assert np.all((c&base[:,None])==c);cost=paid if name in ['question','message_ridge','message_trees','oracle_paid'] else nominal;quality=target[qidx,sidx,c];cost_selected=cost[qidx,sidx,c];utility=quality-5e-6*cost_selected;values[name]=utility;public[name]=dict(EM=float(quality.mean()),utility=float(utility.mean()),service_tokens=float(cost_selected.mean()),utility_by_source=utility.mean(0).tolist(),utility_by_fold=[float(utility[folds==f].mean()) for f in range(6)],mask_counts=np.bincount(c.ravel(),minlength=8).tolist())
 comparisons={};screens={}
 for name in ['message_ridge','message_trees']:
  comparisons[name]={};passed=True
  for baseline in ['fixed','question_preselected','single']:
   delta=values[name]-values[baseline];source=delta.mean(0);fold=np.array([delta[folds==f].mean() for f in range(6)]);gate=delta.mean()>=.005 and np.all(source>0) and (fold>0).sum()>=4;passed=passed and gate
   new=em[qidx,sidx,:,choices[name]];old=em[qidx,sidx,:,choices[baseline]];assert new.shape==(96,2,4)
   comparisons[name][baseline]=dict(utility_gain=float(delta.mean()),gain_by_source=source.tolist(),gain_by_fold=fold.tolist(),positive_folds=int((fold>0).sum()),corrections=int(((new==1)&(old==0)).sum()),destructions=int(((new==0)&(old==1)).sum()),evaluation_cells=int(new.size),comparison_passed=bool(gate))
  regret=values['oracle_paid']-values[name];assert regret.min()>-1e-12;selected=[]
  for s in range(2):selected.extend(regret[np.argsort(margins[name][:,s],kind='stable')[:24],s].tolist())
  enrichment=float(np.mean(selected)/regret.mean()) if regret.mean()>0 else None;screens[name]=dict(primary_passed=bool(passed),risk_enrichment=enrichment,mean_oracle_regret=float(regret.mean()),targeting_passed=enrichment is not None and enrichment>=1.2,primary_and_targeting_passed=bool(passed and enrichment is not None and enrichment>=1.2),auxiliary_graph_EM_MSE=float(np.mean((predictions[name]-target)**2)))
 return dict(methods=public,comparisons=comparisons,screens=screens,folds=fold_log,model_fit_wall_seconds=fit_seconds),dict(predictions=predictions,choices=choices,margins=margins,utility=values,paid_tokens=paid,nominal_tokens=nominal)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.perf_counter();p=Path(a.private_root);fdir=Path(a.features);out=Path(a.output);assert not out.exists();read=lambda f:json.loads(f.read_bytes());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();receipt=read(fdir/'FEATURE_RECEIPT.json');assert receipt['status']=='COMPLETE_INPUT_ONLY_CPU_FEATURES' and sha(fdir/'FEATURES.npz')==receipt['features_sha256'];assert read(p/'d16_crossed_collection_v1/SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_RAW_SCALAR_CROSSED_AUDIT';data=np.load(p/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');features=np.load(fdir/'FEATURES.npz');tasks={};private={}
 with threadpool_limits(limits=2):
  for task in ['hotpot','2wiki']:
   fs={key.removeprefix(task+'_'):features[key] for key in features.files if key.startswith(task+'_')};tasks[task],details=evaluate_task(fs,data[task+'_em'],data[task+'_tokens'],data[task+'_shock'])
   for category,obj in details.items():
    if isinstance(obj,dict):
     for name,array in obj.items():private[task+'_'+category+'_'+name]=array
    else:private[task+'_'+category]=obj
 primary=[name for name in ['message_ridge','message_trees'] if all(v['screens'][name]['primary_passed'] for v in tasks.values())];eligible=[name for name in primary if all(v['screens'][name]['targeting_passed'] for v in tasks.values())];summary=dict(status='COMPLETE_EXPOSED_CROSS_QUESTION_FOUNDATION',tasks=tasks,primary_passing_predictors_both_tasks=primary,targeting_eligible_both_tasks=eligible,independent_confirmation=False,feedback_acquisition_cost_excluded=True,service_all_observed_prefixes_paid=True,new_generative_calls=0,new_GPU_hours=0,cpu_wall_seconds=time.perf_counter()-start,feature_cpu_wall_seconds=receipt['cpu_wall_seconds'],confidence_intervals=False,bootstrap_runs=0,source_sha256=sha(Path(__file__)),features_sha256=sha(fdir/'FEATURES.npz'));out.mkdir();np.savez_compressed(out/'PRIVATE_PREDICTIONS.npz',**private);(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(primary=primary,targeting=eligible,tasks={k:v['screens'] for k,v in tasks.items()},cpu_wall_seconds=summary['cpu_wall_seconds'])))
if __name__=='__main__':main()
