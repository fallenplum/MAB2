import argparse,json,hashlib,re,time,itertools
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tok(text):return set(re.findall(r'\w+',text.lower()))
def jac(a,b):return len(a&b)/max(1,len(a|b))
def features(q,messages,base):
 sets=[tok(t) for t in messages];qs=tok(q);numbers=[{t for t in s if any(c.isdigit() for c in t)} for s in sets]
 neg={'not','no','never','cannot','unclear','unknown','unsure','possibly','perhaps','however'}
 values=list(base)
 for t,s in zip(messages,sets):values += [np.log1p(len(t)),np.log1p(len(t.split())),jac(s,qs),len(s&neg)]
 for i,j in itertools.combinations(range(3),2):values += [jac(sets[i],sets[j]),jac(numbers[i],numbers[j])]
 return values
def nt(n):return n['generation']['input_tokens']+n['generation']['output_tokens']
def metrics(act,y,c):
 i=np.arange(len(y));e=y[i,act];u=e-5e-6*c[i,act]
 return {'accuracy':float(e.mean()),'prefix_paid_service_utility':float(u.mean()),'tokens_per_question':float(c[i,act].mean()),'accuracy_by_seed':e.reshape(-1,5).mean(0).tolist(),'utility_by_seed':u.reshape(-1,5).mean(0).tolist(),'action_counts':np.bincount(act,minlength=3).tolist()}
def run(a):
 start=time.time();d6=np.load(a.d6);y=d6['em'][:,:,:3].reshape(-1,3);yo=d6['old_em'].reshape(-1,3);x=[];xo=[];cs=[];co=[];qs=[]
 collectcost=0
 for i in range(640):
  fs=list(a.archive.glob(f'worker_*/unit_{i:03d}/MATRIX.json'));assert len(fs)==1
  m=read(fs[0]);p=read(fs[0].parent/'INPUT.json')['payload'];e=m['removed_upstream'];other='V' if e=='U' else 'U'
  for seed in range(26111571,26111576):
   rows={c['condition']:c for c in m['conditions'] if c['execution_seed']==seed}
   for stem,xx,cc in [('G0',xo,co),('G1',x,cs)]:
    row=rows[stem];texts=[row['nodes'][v]['wire_message'] for v in [e,other,'R']]
    xx.append(features(p['question'],texts,d6['features'][i]))
    prefix=sum(nt(row['nodes'][v]) for v in ['U','V','R'])
    cc.append([prefix+nt(rows[g]['nodes']['M']) for g in [stem,stem+'_MINUS_E'+e,stem+'_MINUS_E'+other]])
   unique={v['request_fingerprint']:v for g in [*['G0','G0_MINUS_EU','G0_MINUS_EV'],*['G1','G1_MINUS_EU','G1_MINUS_EV']] for v in rows[g]['nodes'].values()}
   collectcost+=sum(nt(v) for v in unique.values())
  if(i+1)%160==0:print('prefixes',i+1,flush=True)
 x=np.asarray(x);xo=np.asarray(xo);c=np.asarray(cs);co=np.asarray(co);groups=np.repeat(np.arange(640)//80,5)
 gains=y[:,1:]-y[:,0,None];ogains=yo[:,1:]-yo[:,0,None]
 acts={};folds=[];equiv=[]
 for ep in range(8):
  tr=np.flatnonzero(groups!=ep);te=np.flatnonzero(groups==ep)
  def save(name,pred):
   if name not in acts:acts[name]=np.zeros(3200,dtype=int)
   act=pred.argmax(1);acts[name][te]=act;folds.append({'fold':ep,'method':name,**metrics(act,y[te],c[te])})
  save('train_best_constant',np.repeat((y[tr]-5e-6*c[tr]).mean(0)[None,:],len(te),axis=0))
  sc=StandardScaler().fit(np.concatenate([x[tr],xo[tr]]));xt=sc.transform(x[tr]);xv=sc.transform(x[te]);xot=sc.transform(xo[tr])
  cp=c[tr].mean(0)
  for kind in ['ridge','tree']:
   def fit(z,t):return (Ridge(alpha=10,solver='svd') if kind=='ridge' else DecisionTreeRegressor(max_depth=3,min_samples_leaf=80,random_state=261116)).fit(z,t)
   current=fit(xt,gains[tr]);old=fit(xot,ogains[tr])
   direct=current.predict(xv);transfer=old.predict(xv)
   save(kind+'_current_prefix',np.column_stack([np.zeros(len(te)),direct])-5e-6*cp)
   save(kind+'_old_prefix_transfer',np.column_stack([np.zeros(len(te)),transfer])-5e-6*cp)
  # Exact same-design linear direct-vs-residual identity; this is an audit, not a new method.
  old_at_current=Ridge(alpha=10,solver='svd').fit(xt,ogains[tr]).predict(xv)
  residual=Ridge(alpha=10,solver='svd').fit(xt,gains[tr]-ogains[tr]).predict(xv)
  direct=Ridge(alpha=10,solver='svd').fit(xt,gains[tr]).predict(xv)
  equiv.append(float(np.max(abs(direct-old_at_current-residual))))
 results={k:metrics(act,y,c) for k,act in acts.items()}
 for i in range(3):results['fixed_'+str(i)]=metrics(np.full(3200,i),y,c)
 oracle=(y-5e-6*c).argmax(1);results['same_seed_oracle_non_deployable']=metrics(oracle,y,c)
 gates={}
 for k,v in results.items():
  if not k.startswith(('ridge','tree')):continue
  delta=v['prefix_paid_service_utility']-results['train_best_constant']['prefix_paid_service_utility']
  pos=sum(next(z['prefix_paid_service_utility'] for z in folds if z['fold']==e and z['method']==k)>next(z['prefix_paid_service_utility'] for z in folds if z['fold']==e and z['method']=='train_best_constant') for e in range(8))
  spos=sum(a>b for a,b in zip(v['utility_by_seed'],results['train_best_constant']['utility_by_seed']))
  gates[k]={'difference':delta,'positive_folds':pos,'positive_seeds':spos,'pass':delta>=.005 and pos>=6 and spos>=4}
 assert max(equiv)<1e-10
 out={'status':'COMPLETE_EXPLORATORY_PREFIX_STATE_DIAGNOSTIC','new_GPU_hours':0,'new_model_calls':0,'no_terminal_answer_in_features':True,'questions':640,'question_seed_rows':3200,'groups_per_fold':80,'results':results,'folds':folds,'gates':gates,'same_design_linear_identity_max_error':max(equiv),'all_old_new_training_graph_collection_tokens':collectcost,'training_feedback_cost_included_in_reported_service_utility':False,'service_prefix_even_dead_nodes_charged':True,'unseen_dataset_confirmation':False,'source_sha256':sha(__file__),'wall_seconds':time.time()-start}
 assert not a.output.exists() and not a.private_output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
 np.savez_compressed(a.private_output,x=x,old_x=xo,y=y,old_y=yo,cost=c,old_cost=co,groups=groups)
 print(json.dumps({'gates':gates,'results':results,'linear_identity':max(equiv)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--d6',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--private-output',type=Path,required=True);run(p.parse_args())
