from pathlib import Path
from collections import Counter
import argparse,json,hashlib
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 expected=json.loads(a.result.read_bytes());assert hashlib.sha256(a.inputs.read_bytes()).hexdigest()==expected['input_sha256']
 with np.load(a.inputs,allow_pickle=False) as z:
  part=z['partition'];left=z['left'].reshape(400,2,12);right=z['right'].reshape(400,2,12)
 hist={s:Counter() for s in ['train','tune']};count={s:Counter() for s in hist};checks=0
 for qi in np.flatnonzero(np.isin(part,['train','tune'])):
  s=str(part[qi]);l=left[qi].ravel();r=right[qi].ravel();v=sorted(set(l)|set(r));idx={x:i for i,x in enumerate(v)}
  D=np.zeros((24,len(v)))
  for j,(u,w) in enumerate(zip(l,r)):D[j,idx[u]]=-1;D[j,idx[w]]=1
  rank=int(np.linalg.matrix_rank(D,tol=1e-8));degree=np.abs(D).sum(axis=0);centers={v[j] for j in np.flatnonzero(degree==12)};assert len(centers)==2
  star_sets=[set(left[qi,k])|set(right[qi,k]) for k in range(2)];assert all(len(x)==13 for x in star_sets)
  dup=24-len({tuple(sorted([int(u),int(w)])) for u,w in zip(l,r)});c=int(np.isin(r,l).sum());cc=sum(x in set(l) and x in centers for x in r)
  key=json.dumps({'states':len(v),'unique_contrasts':24-dup,'components':len(v)-rank,'incidence_rank':rank,'star_overlap_states':len(star_sets[0]&star_sets[1])},sort_keys=True)
  hist[s][key]+=1;count[s].update(questions=1,contrasts=24,independent_endpoint_state_count=48,shared_anchor_state_count=26,globally_unique_state_count=len(v),right_in_left=c,right_in_left_center=cc,right_in_left_noncenter=c-cc,duplicate_contrasts=dup);checks+=1
 for s in hist:
  assert dict(count[s])==expected['splits'][s]['counts']
  target=Counter({json.dumps(x['structure'],sort_keys=True):x['questions'] for x in expected['splits'][s]['structure_histogram']});assert hist[s]==target
 result={'status':'PASS_SEPARATE_MATRIX_RANK_AND_DEGREE_REPLAY','question_structures_checked':checks,'source_result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'review_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'quality_read':False,'new_GPU_calls':0,'reviewer':'root separate implementation, not external human reviewer'}
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
