import json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(r'E:\图结构演化\artifacts\private\mechanism_20261122')
TASKS = ('qasc','sciq','commonsense')
DENSITIES = ('','_rows2','_rows4')
TAU = 0.01

def entropy(p):
    p=np.asarray(p,dtype=np.float32)
    return float(-(p*np.log(np.clip(p,1e-8,1))).sum())

def load_rows():
    rows=[]
    for task in TASKS:
      for suffix in DENSITIES:
        d=ROOT/(f'm60_conditional_server_{task}' if suffix=='' else f'm60_conditional_{task}{suffix}')
        if not d.exists() or not (d/'SEALED.json').exists(): continue
        gold=json.loads((d/'SEALED.json').read_text())
        for line in (d/'WORKER0.jsonl').read_text().splitlines():
          if not line.strip(): continue
          x=json.loads(line); full=x['full']; vec=lambda i: np.asarray(full[str(i)],dtype=np.float32)
          rows.append({'key':x['key'],'task':task,'density':suffix or '_rows1',
            'old_source':vec(0),'new_source':vec(1),'base':vec(2),
            'old_relay':np.stack([vec(i) for i in range(3,7)]),
            'new_relay':np.stack([vec(i) for i in range(7,11)]),
            'labels':np.asarray([int(full[str(i)]) for i in range(11,27)],dtype=np.int64).reshape(4,4),
            'tokens':x['tokens'],'changed':tuple(int(v) for v in x.get('changed_rows',[])),
            'gold':int(gold[x['key']])})
    return rows

def values(source,base,relay,labels):
    w=np.asarray(source)[:,None]*(np.asarray(relay)-np.asarray(base)[None,:])
    return np.asarray([w[np.asarray(labels)==y].sum() for y in range(int(np.max(labels))+1)])

def truth(source,base,relay,labels): return values(source,base,relay,labels)>TAU

def row_targets(r):
    target=truth(r['new_source'],r['base'],r['new_relay'],r['labels']); y=np.zeros(4,dtype=np.float32)
    for i in range(4):
      cand=r['new_relay'].copy(); cand[i]=r['old_relay'][i]
      y[i]=float(np.any(truth(r['new_source'],r['base'],cand,r['labels'])!=target))
    return y,target

def feat(r,i):
    old=r['old_relay'][i]
    oldv=values(r['old_source'],r['base'],r['old_relay'],r['labels'])
    sort=np.sort(oldv)[::-1]
    margin=float(sort[0]-sort[1]) if len(sort)>1 else 0.0
    near=float(np.min(np.abs(oldv-TAU)))
    return np.concatenate([r['old_source'],r['new_source'],r['base'],r['new_source']-r['old_source'],old,
      np.asarray([entropy(old),float(old.max()-old.min()),float(r['tokens'][7+i])/200.,margin,near],dtype=np.float32),
      np.eye(4,dtype=np.float32)[i],np.asarray([float(i in r['changed'])],dtype=np.float32)]).astype(np.float32)

class Policy(nn.Module):
  def __init__(self,d):
    super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
  def forward(self,x): return self.net(x).squeeze(-1)

def train_model(rows,seed):
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  X=[];Y=[]
  for r in rows:
    y,_=row_targets(r)
    for i in range(4): X.append(feat(r,i));Y.append(y[i])
  X=torch.tensor(np.asarray(X),dtype=torch.float32);Y=torch.tensor(np.asarray(Y),dtype=torch.float32)
  m=Policy(X.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
  pos=(len(Y)-Y.sum())/max(float(Y.sum()),1.0); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
  for _ in range(100):
    loss=lossfn(m(X),Y);opt.zero_grad();loss.backward();opt.step()
  return m,float(Y.mean())

def old_margin(r):
  v=values(r['old_source'],r['base'],r['old_relay'],r['labels']); s=np.sort(v)[::-1]
  return float(s[0]-s[1]) if len(s)>1 else 0.0

def chooser(model,r,margin_gate,drift_gate,alpha,threshold):
  c=list(r['changed'])
  if not c: return []
  # The gate uses only old observations. A closed gate keeps the warm certificate.
  near=float(np.min(np.abs(values(r['old_source'],r['base'],r['old_relay'],r['labels'])-TAU)))
  drift=float(np.sum(np.abs(r['new_source']-r['old_source'])))
  if near>margin_gate and drift<drift_gate: return []
  with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(np.stack([feat(r,i) for i in c])))).numpy()
  h=np.asarray([entropy(r['old_relay'][i]) for i in c],dtype=np.float32)
  hn=(h-h.min())/(h.max()-h.min()+1e-8) if len(h)>1 else np.zeros_like(h)
  score=alpha*p+(1-alpha)*hn
  if float(score.max())<threshold: return []
  return [c[int(np.argmax(score))]]

def metrics(rows,choose):
  exact=[];acc=[];cost=[];nq=[]
  for r in rows:
    selected=choose(r);relay=r['old_relay'].copy()
    for i in selected: relay[i]=r['new_relay'][i]
    v=values(r['new_source'],r['base'],relay,r['labels']); pred=v>TAU
    exact.append(float(np.array_equal(pred,truth(r['new_source'],r['base'],r['new_relay'],r['labels']))))
    acc.append(float(int(np.argmax(v))==r['gold']))
    cost.append(float(sum(r['tokens'][7+i] for i in selected))); nq.append(len(selected))
  return {'n':len(rows),'decision_exact':float(np.mean(exact)),'answer_accuracy':float(np.mean(acc)),'tokens':float(np.mean(cost)),'queries':float(np.mean(nq))}

def baseline(rows,name):
  if name=='warm': return metrics(rows,lambda r:[])
  if name=='uncertainty': return metrics(rows,lambda r: ([max(r['changed'],key=lambda i:entropy(r['old_relay'][i]))] if r['changed'] else []))
  if name=='manifest': return metrics(rows,lambda r:list(r['changed']))
  if name=='cold': return metrics(rows,lambda r:list(range(4)))

def fit_params(model,train):
  # Tune only on development rows. The model is fit on all non-target tasks.
  best=None
  for mg in [0.01,0.05]:
    for dg in [0.0,0.1,0.4]:
      for alpha in [0.5,1.]:
        for threshold in [0.2,0.5]:
          z=metrics(train,lambda r,mm=model,mg=mg,dg=dg,a=alpha,th=threshold: chooser(mm,r,mg,dg,a,th))
          u=baseline(train,'uncertainty')
          score=z['answer_accuracy']-0.0005*z['tokens']-0.25*max(0.,(u['decision_exact']-0.03)-z['decision_exact'])
          if best is None or score>best[0]: best=(score,mg,dg,alpha,threshold)
  return {'margin_gate':best[1],'drift_gate':best[2],'alpha':best[3],'threshold':best[4]}

def run(seed,target):
  rows=load_rows();train=[r for r in rows if r['task']!=target and int(r['key'][:8],16)%5<4];test=[r for r in rows if r['task']==target and int(r['key'][:8],16)%5==4]
  model,pos=train_model(train,seed); pars=fit_params(model,train)
  out={'seed':seed,'target_task':target,'train':len(train),'test':len(test),'positive_rate':pos,'params':pars}
  for n in ('warm','uncertainty','manifest','cold'): out[n]=baseline(test,n)
  out['safe_gated']=metrics(test,lambda r:chooser(model,r,pars['margin_gate'],pars['drift_gate'],pars['alpha'],pars['threshold']))
  return out

if __name__=='__main__':
  out={'schema':'m60_safe_gated_budget_policy_v1','results':[run(s,t) for t in TASKS for s in (7,19,43)],'interpretation':'A trainable row-impact score is permitted only when the old graph is close to a decision boundary. Hyperparameters are selected in task-held-out development splits. Test rows are question-disjoint target-task rows. This is a development experiment using private sealed answer labels.'}
  p=Path('MAB2_recovered/experiments/mechanism_20261122/M60_SAFE_GATED_POLICY.json');p.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

