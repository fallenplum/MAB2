"""Anonymous graph-index structure audit; quality and question identifiers never loaded."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists();protocol=json.loads(a.protocol.read_bytes());start=time.process_time()
 assert hashlib.sha256(a.inputs.read_bytes()).hexdigest()==protocol['fixed_input_sha256']
 with np.load(a.inputs,allow_pickle=False) as z:
  part=z['partition'];selected=np.flatnonzero(np.isin(part,['train','tune']));assert len(selected)==280
  left=z['left'].reshape(400,2,12)[selected];right=z['right'].reshape(400,2,12)[selected]
 hist={s:Counter() for s in ['train','tune']};counts={s:Counter() for s in hist}
 for ix,qi in enumerate(selected):
  split=str(part[qi]);edges=[(int(l),int(r)) for l,r in zip(left[ix].ravel(),right[ix].ravel())];assert len(edges)==24 and all(u!=v for u,v in edges)
  vertices={v for e in edges for v in e};parents={v:v for v in vertices}
  def find(v):
   while parents[v]!=v:v=parents[v]
   return v
  for u,v in edges:parents[find(u)]=find(v)
  components=len({find(v) for v in vertices});unique_edges={tuple(sorted(e)) for e in edges}
  stars=[];centers=[]
  for anchor in range(2):
   es=edges[12*anchor:12*(anchor+1)];common=set(es[0])
   for e in es[1:]:common &= set(e)
   vs={v for e in es for v in e};assert len(common)==1 and len(vs)==13
   stars.append(vs);centers.append(next(iter(common)))
  shared=len(stars[0]&stars[1]);ls=set(left[ix].ravel().tolist());ri=right[ix].ravel().tolist()
  shared_r=sum(v in ls for v in ri)
  shared_center=sum(v in ls and v in centers for v in ri)
  key=json.dumps({'states':len(vertices),'unique_contrasts':len(unique_edges),'components':components,'incidence_rank':len(vertices)-components,'star_overlap_states':shared},sort_keys=True)
  hist[split][key]+=1;counts[split].update(questions=1,contrasts=24,independent_endpoint_state_count=48,shared_anchor_state_count=26,globally_unique_state_count=len(vertices),right_in_left=shared_r,right_in_left_center=shared_center,right_in_left_noncenter=shared_r-shared_center,duplicate_contrasts=24-len(unique_edges))
 result={'status':'COMPLETE_STRUCTURE_ONLY_NO_QUALITY_READ','protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),'input_sha256':protocol['fixed_input_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'splits':{s:{'counts':dict(counts[s]),'structure_histogram':[{'structure':json.loads(k),'questions':v} for k,v in sorted(hist[s].items())]} for s in hist},'CPU_seconds':time.process_time()-start,'new_gpu_hours':0,'formal_round_increment':0,'quality_array_accessed':False,'R1_accessed':False,'raw_question_identifiers_accessed':False}
 a.output.mkdir(parents=True);(a.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':result['status'],'splits':result['splits'],'CPU_seconds':result['CPU_seconds']}))
if __name__=='__main__':main()
