import importlib.util,numpy as np,json
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).with_name('score.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
shock=np.arange(96)%3;zeros=np.zeros((96,2,4,8));a=m.analyze(zeros,zeros,shock);assert not a['necessary_headroom_passed'];assert a['mean_gains']=={'gain_vs_stale':0.,'gain_vs_best_fixed':0.}
# Shocked-edge-only rewards are inaccessible: an illegal oracle must not pass.
illegal=zeros.copy()
for i,h in enumerate(shock):
 for mask in range(8):illegal[i,:,:,mask]=bool(mask&(1<<h))
b=m.analyze(illegal,zeros,shock);assert not b['necessary_headroom_passed'];assert all(v['methods']['fresh']['EM']==0 for v in b['transfers'])
# Independent scalar evaluation of random binary arrays catches advanced-index errors.
rng=np.random.default_rng(71);em=rng.integers(0,2,size=zeros.shape);tokens=rng.integers(100,4000,size=zeros.shape);c=m.analyze(em,tokens,shock)
for direction,(train,test) in enumerate([([0,1],[2,3]),([2,3],[0,1])]):
 vals=[]
 for i,h in enumerate(shock):
  base=7^(1<<h);masks=[base]+[base^(1<<e) for e in range(3) if base&(1<<e)]
  values=[sum(float(em[i,s,t,k])-5e-6*float(tokens[i,s,t,k]) for s in range(2) for t in train)/4 for k in masks]
  chosen=masks[max(range(3),key=lambda x:values[x])]
  assert chosen&base==chosen
  vals.append(sum(float(em[i,s,t,chosen])-5e-6*float(tokens[i,s,t,chosen]) for s in range(2) for t in test)/4)
 assert abs(sum(vals)/96-c['transfers'][direction]['methods']['fresh']['utility'])<1e-12
print(json.dumps(dict(status='PASS_ARTIFICIAL_LEGAL_DECISION_TESTS',zero_rejected=True,unavailable_edge_reward_rejected=True,scalar_random_array_check=True,new_model_calls=0)))
