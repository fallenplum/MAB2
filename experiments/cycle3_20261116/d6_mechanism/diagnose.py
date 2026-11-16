"""Exposed-data diagnostic only. No new generation and no new test access."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def read(p):return json.loads(Path(p).read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel()
 return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def cost(r):return sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for n in r['nodes'].values())
def feature(payload):
 q=payload['question'];context=payload['context']
 return [float(payload['type']=='comparison'),float(payload['level']=='hard'),float(payload['level']=='medium'),np.log1p(len(q)),np.log1p(len(json.dumps(context))),float(q.lower().startswith(('who','which'))),float(q.lower().startswith(('are','is','was','were')))]
def evaluate(actions,y,tokens):
 n=len(y);a=np.asarray(actions);v=y[np.arange(n),:,a];c=tokens[np.arange(n),:,a]
 return {'accuracy':float(v.mean()),'service_utility':float((v-5e-6*c).mean()),'tokens_per_service_question':float(c.mean()),'accuracy_by_seed':v.mean(0).tolist(),'utility_by_seed':(v-5e-6*c).mean(0).tolist(),'action_counts':np.bincount(a,minlength=y.shape[2]).tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--private-output',type=Path,required=True);p.add_argument('--questions',type=Path,required=True);a=p.parse_args()
 assert digest(a.private)=='dea10c767ff0588d9774fc294863649fe2c012a6f090a9d1b5749145793bcb5e'
 assert not a.output.exists() and not a.private_output.exists()
 start=time.time();d=read(a.private);n=640;assert len(d['ids'])==n
 assert digest(a.questions)=='db51007a53dfc878f807c8ff4b03363e2185d65d4d6761f942629f377d61114e'
 metadata={q['id']:q for q in [json.loads(line) for line in a.questions.read_text(encoding='utf-8').splitlines()]};assert set(metadata)==set(d['ids'])
 new=np.zeros((n,5,5));old=np.zeros((n,5,3));nt=np.zeros_like(new);ot=np.zeros_like(old)
 x=[];questions=[];cert=0;cert_bad=0;rsame=[];msg_distance=[];group=[]
 for i in range(n):
  fs=list(a.archive.glob(f'worker_*/unit_{i:03d}/MATRIX.json'));assert len(fs)==1
  m=read(fs[0]);inp=read(fs[0].parent/'INPUT.json');q=inp['payload'];assert q['id']==d['ids'][i]
  shock=m['removed_upstream'];e=shock;other='V' if e=='U' else 'U'
  ng=['G1','G1_MINUS_E'+e,'G1_MINUS_E'+other,'SINGLE_Q','INDEPENDENT3_Q'];og=['G0','G0_MINUS_E'+e,'G0_MINUS_E'+other]
  assert metadata[q['id']]['question']==q['question']
  x.append(feature(metadata[q['id']]));questions.append(q['question']);group.append(i//80)
  for s in range(5):
   new[i,s]=[d['em'][i][s][g] for g in ng];old[i,s]=[d['em'][i][s][g] for g in og]
   nt[i,s]=[cost(d['records'][i][s][g]) for g in ng];ot[i,s]=[cost(d['records'][i][s][g]) for g in og]
   rows={c['condition']:c for c in m['conditions'] if c['execution_seed']==26111571+s}
   ra=rows['G0']['nodes']['R']['wire_message'];rb=rows['G1']['nodes']['R']['wire_message']
   same=ra==rb;rsame.append(same)
   aa=set(ra.lower().split());bb=set(rb.lower().split());msg_distance.append(1-len(aa&bb)/max(1,len(aa|bb)))
   if same:
    cert+=1
    cert_bad+=any(rows[g0]['nodes']['M']['request_fingerprint']!=rows[g1]['nodes']['M']['request_fingerprint'] for g0,g1 in zip(og,ng[:3]))
  if (i+1)%160==0:print('loaded',i+1,flush=True)
 x=np.asarray(x);group=np.asarray(group);mean_y=new.mean(1);mean_cost=nt.mean(1);mean_u=mean_y-5e-6*mean_cost
 old_y=old.mean(1);old_cost=ot.mean(1);old_u=old_y-5e-6*old_cost
 method_actions={};foldrows=[];linearity=[]
 for ep in range(8):
  tr=np.flatnonzero(group!=ep);te=np.flatnonzero(group==ep);assert len(tr)==560 and len(te)==80
  def save(name,pred,k=3):
   if name not in method_actions:method_actions[name]=np.zeros(n,dtype=int)
   actions=pred.argmax(1);method_actions[name][te]=actions
   result=evaluate(actions,new[te,:,:k],nt[te,:,:k]);foldrows.append({'fold':ep,'method':name,**result})
  for k,tag in [(3,'graph'),(5,'system')]:
   base=np.repeat(mean_u[tr,:k].mean(0)[None,:],len(te),0);save('train_best_constant_'+tag,base,k)
  save('old_train_best_constant_graph',np.repeat(old_u[tr].mean(0)[None,:],len(te),0))
  for kind in ['metadata','tfidf']:
   if kind=='metadata':
    sc=StandardScaler();xt=sc.fit_transform(x[tr]);xv=sc.transform(x[te])
   else:
    sc=TfidfVectorizer(min_df=5,max_features=512,ngram_range=(1,2));xt=sc.fit_transform([questions[t] for t in tr]);xv=sc.transform([questions[t] for t in te])
   pred=Ridge(alpha=10,solver='lsqr').fit(xt,mean_u[tr]).predict(xv)
   save(kind+'_direct_graph',pred[:,:3]);save(kind+'_direct_system',pred,5)
   po=Ridge(alpha=10,solver='lsqr').fit(xt,old_u[tr]).predict(xv)
   # Transfer old quality but use current per-graph TRAINING mean prices (no future actual costs).
   py=Ridge(alpha=10,solver='lsqr').fit(xt,old_y[tr]).predict(xv)
   save(kind+'_old_credit_transfer_graph',py-5e-6*mean_cost[tr,:3].mean(0))
   residual=Ridge(alpha=10,solver='lsqr').fit(xt,mean_u[tr,:3]-old_u[tr]).predict(xv)
   save(kind+'_old_plus_shift_graph',po+residual)
   linearity.append(float(abs(po+residual-pred[:,:3]).max()))
 results={m:evaluate(acts,new,nt) for m,acts in method_actions.items()}
 for k in range(5):results['fixed_'+str(k)]=evaluate(np.full(n,k),new,nt)
 results['oracle_question_mean_graph_non_deployable']=evaluate(mean_u[:,:3].argmax(1),new,nt)
 results['oracle_question_mean_system_non_deployable']=evaluate(mean_u.argmax(1),new,nt)
 # A seed held out from the oracle decision exposes its optimistic selection component.
 z=[];cz=[]
 for s in range(5):
  others=[j for j in range(5) if j!=s];act=(new[:,others,:3].mean(1)-5e-6*nt[:,others,:3].mean(1)).argmax(1)
  z.append(new[np.arange(n),s,act]);cz.append(nt[np.arange(n),s,act])
 results['oracle_other_four_seeds_graph_non_deployable']={'accuracy':float(np.mean(z)),'service_utility':float((np.asarray(z)-5e-6*np.asarray(cz)).mean())}
 gains=new[:,:,:3][:,:,1:]-new[:,:,0,None];ogains=old[:,:,1:]-old[:,:,0,None];drift=gains-ogains
 diag={'credit_nonzero_seed_fraction':float((gains!=0).mean()),'credit_any_nonzero_question_fraction':float(np.any(gains!=0,axis=(1,2)).mean()),'old_new_mean_credit_correlation':corr(ogains.mean(1),gains.mean(1)),
 'credit_cross_seed_0_vs_remaining':corr(gains[:,0],gains[:,1:].mean(1)), 'shift_cross_seed_0_vs_remaining':corr(drift[:,0],drift[:,1:].mean(1)),
 'credit_mean_absolute':float(abs(gains.mean(1)).mean()),'shift_mean_absolute':float(abs(drift.mean(1)).mean()),
 'receiver_message_exact_equal_count':cert,'receiver_message_pairs':3200,'equal_message_fingerprint_violation_count':cert_bad,
 'message_token_set_distance_vs_absolute_credit_shift':corr(np.asarray(msg_distance),abs(drift).mean(2).ravel()),'linear_direct_vs_residual_max_abs':max(linearity)}
 gates={}
 for name in results:
  if not name.endswith('_graph') or name.startswith(('train_','old_train_')):continue
  diff=results[name]['service_utility']-results['train_best_constant_graph']['service_utility']
  positives=sum(next(t['service_utility'] for t in foldrows if t['fold']==ep and t['method']==name)>next(t['service_utility'] for t in foldrows if t['fold']==ep and t['method']=='train_best_constant_graph') for ep in range(8))
  gates[name]={'difference':diff,'positive_folds':positives,'pass':diff>=.005 and positives>=6}
 summary={'status':'COMPLETE_EXPLORATORY_D6','questions':640,'seeds':5,'new_model_calls':0,'new_GPU_hours':0,'bootstrap_runs':0,'all_data_previously_exposed':True,'not_independent_confirmation':True,'service_only_metrics_not_full_training_cost_gain':True,'action_names':['keep','delete_affected','delete_other','single_Q','independent3_Q'],'results':results,'fold_results':foldrows,'diagnostics':diag,'cpu_screen_gates':gates,'wall_seconds':time.time()-start,'source_sha256':digest(__file__),'input_sha256':digest(a.private)}
 assert cert_bad==0
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
 a.private_output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.private_output,em=new,old_em=old,tokens=nt,old_tokens=ot,features=x,groups=group,actions=np.stack(list(method_actions.values())),method_names=list(method_actions))
 print(json.dumps({'diagnostics':diag,'gates':gates,'best_constant':results['train_best_constant_graph'],'oracle_held_seed':results['oracle_other_four_seeds_graph_non_deployable']}),flush=True)
if __name__=='__main__':main()
