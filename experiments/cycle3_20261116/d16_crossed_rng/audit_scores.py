"""Independent scalar audit of crossed raw outcomes, SS components and repeat-mean gates."""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_bytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d16_crossed_collection_v1';summary=read(run/'scoring_v1/SUMMARY.json');arrays=np.load(run/'scoring_v1/PRIVATE_ARRAYS.npz');entries=read(p/'d16_crossed_input_v1/ENTRIES.json');new={v['job_id']:v['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};checks=0;seeds=[26111671,26111672];terminal=[26111671,26111672,261116161,261116162]
 def near(a,b):
  nonlocal checks
  assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10),(a,b);checks+=1
 def corr(x,y):return statistics.correlation(x,y) if len(set(x))>1 and len(set(y))>1 else None
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  formal=p/folder;approval=read(formal/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics
  vault=read(approval['pins']['vault']['path']);ids={read(u/'INPUT.json')['index']:read(u/'INPUT.json')['payload']['id'] for u in (formal/'monitor/archive').glob('worker_*/unit_*')};raw={}
  for e in (v for v in entries if v['task']==task):
   i=e['index'];s=seeds.index(e['source_seed']);t=terminal.index(e['terminal_seed']);m=e['mask'];g=e['cached']['generation'] if e['cache_hit'] else new[e['job_id']];gold=vault[ids[i]];scores=[answer_metrics(g['text'].strip(),v) for v in (gold if task=='2wiki' else [gold])];y=max(v['em'] for v in scores);f=max(v['f1'] for v in scores);cost=e['prefix_tokens']+g['input_tokens']+g['output_tokens'];raw[i,s,t,m]=(y,f,cost,y-5e-6*cost)
   for field,value in [('em',y),('f1',f),('tokens',cost)]:near(value,float(arrays[task+'_'+field][i,s,t,m]))
  result=summary['tasks'][task];credits={name:{} for name in ['old','current','shift']}
  for i in range(96):
   h=int(arrays[task+'_shock'][i]);cur=7^(1<<h)
   for j,e in enumerate(e for e in range(3) if e!=h):
    for s in range(2):
     for t in range(4):
      c0=raw[i,s,t,7][0]-raw[i,s,t,7^(1<<e)][0];c1=raw[i,s,t,cur][0]-raw[i,s,t,cur^(1<<e)][0]
      for name,value in [('old',c0),('current',c1),('shift',c1-c0)]:credits[name][i,j,s,t]=value
  measured={}
  for name,values in credits.items():
   means_terminal=[[sum(values[i,j,s,t] for s in range(2) for t in range(2*h,2*h+2))/4 for i in range(96) for j in range(2)] for h in range(2)];means_source=[[sum(values[i,j,s,t] for t in range(4))/4 for i in range(96) for j in range(2)] for s in range(2)];measured[name]={'terminal_half':corr(*means_terminal),'source_half':corr(*means_source)}
   for key,value in measured[name].items():
    if value is None:assert result['correlations'][name][key] is None
    else:near(value,result['correlations'][name][key])
   ss=[0.,0.,0.,0.]
   for i in range(96):
    for j in range(2):
     grid=[[values[i,j,s,t] for t in range(4)] for s in range(2)];grand=sum(sum(row) for row in grid)/8;sm=[sum(row)/4 for row in grid];tm=[sum(grid[s][t] for s in range(2))/2 for t in range(4)];ss[0]+=4*sum((v-grand)**2 for v in sm);ss[1]+=2*sum((v-grand)**2 for v in tm);ss[2]+=sum((grid[s][t]-sm[s]-tm[t]+grand)**2 for s in range(2) for t in range(4));ss[3]+=sum((v-grand)**2 for row in grid for v in row)
   for value,key in zip(ss,['source_SS','terminal_SS','interaction_SS','total_within_grid_SS']):near(value,result['descriptive_noise_components'][name][key])
   near(sum(ss[:3]),ss[3])
  halves=[{(i,m):sum(raw[i,s,t,m][3] for s in range(2) for t in range(2*h,2*h+2))/4 for i in range(96) for m in range(8)} for h in range(2)];gains=[]
  for n,(train,test) in enumerate([(0,1),(1,0)]):
   choice={i:max(range(8),key=lambda m:halves[train][i,m]) for i in range(96)};fixed=max(range(8),key=lambda m:sum(halves[test][i,m] for i in range(96)));gain=sum(halves[test][i,choice[i]]-halves[test][i,fixed] for i in range(96))/96;near(gain,result['transfer_oracles'][n]['utility_gain']);assert fixed==result['transfer_oracles'][n]['best_fixed_mask'];gains.append(gain)
  credit_gate=all(measured[name][kind] is not None and measured[name][kind]>=threshold for name,threshold in [('current',.30),('shift',.20)] for kind in ['terminal_half','source_half']);headroom=sum(gains)/2>=.005 and min(gains)>0;assert credit_gate==result['credit_gate'];assert headroom==result['headroom_gate'];assert (credit_gate and headroom)==result['necessary_screen_passed']
 assert summary['necessary_screen_both_tasks']==all(v['necessary_screen_passed'] for v in summary['tasks'].values())
 result=dict(status='PASS_INDEPENDENT_RAW_SCALAR_CROSSED_AUDIT',numeric_checks=checks,raw_logical_outcomes=12288,shared_official_normalizer=True,summary_sha256=hashlib.sha256((run/'scoring_v1/SUMMARY.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
