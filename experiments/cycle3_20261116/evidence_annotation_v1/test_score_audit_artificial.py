"""Artificial qualification traces only: test independent ancillary reconstruction."""
from pathlib import Path
import json,sys,copy,importlib.util
import numpy as np
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';p=root/'artifacts/private/cycle3_20261116'
d=r/'experiments/cycle3_20261116/evidence_annotation_v1'
def module(name,file):
 s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
audit=module('ind',d/'audit_raw_scoring.py');ev=module('ev',d/'evaluate.py')
sys.path.insert(0,str(r/'experiments/cycle2_20261113/hotpot_support'));from hotpot_support import answer_metrics
src=next((p/'annotation_qualification_v1/monitor/archive').glob('worker_*/unit_*/MATRIX.json'));original=json.loads(src.read_bytes())
out=p/'annotation_score_audit_artificial_v1';out.mkdir(exist_ok=False);arc=out/'fixture';seeds=[26111671,26111672];oldseeds=sorted(set(c['execution_seed'] for c in original['conditions']))
names=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_ANNOTATION','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
em=np.zeros((96,2,11));f1=em.copy();tok=em.copy();cert=np.zeros((96,2,2,2),bool);valid=[];proj=[];rank=[];vault={};questions=[];viol=0
for i in range(96):
 mat=copy.deepcopy(original);qid=f'artificial_{i}';gold=mat['conditions'][0]['answer'] if i%2 else 'Deliberately unmatched artificial reference';vault[qid]=gold;questions.append({'id':qid,'type':'synthetic_a' if i%2 else 'synthetic_b'})
 for c in mat['conditions']:c['execution_seed']=seeds[oldseeds.index(c['execution_seed'])]
 wd=arc/'worker_0'/f'unit_{i:03d}';wd.mkdir(parents=True);(wd/'INPUT.json').write_text(json.dumps({'index':i,'payload':{'id':qid}}));(wd/'MATRIX.json').write_text(json.dumps(mat))
 rows={(c['execution_seed'],c['condition']):c for c in mat['conditions']}
 for j,seed in enumerate(seeds):
  for k,name in enumerate(names):
   c=rows[seed,name];s=answer_metrics(c['answer'],gold);em[i,j,k]=s['em'];f1[i,j,k]=s['f1'];tok[i,j,k]=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for n in c['nodes'].values())
   for n in c['nodes'].values():
    if 'selection_valid' in n:valid.append(n['selection_valid']);proj.append(n['budget_projection_applied']);rank.append(n['ranking_length'])
  for t in range(2):
   for e,edge in enumerate(['U','V']):
    b=rows[seed,f'G{t}'];a=rows[seed,f'G{t}_MINUS_E{edge}'];same=b['terminal_evidence_ids']==a['terminal_evidence_ids'];cert[i,j,t,e]=same
    if same:viol+=int(b['answer']!=a['answer'] or b['nodes']['M']['request_fingerprint']!=a['nodes']['M']['request_fingerprint'])
s=ev.summarize(em,f1,tok,names,seeds);s['type_breakdown']={}
for group in ['synthetic_a','synthetic_b']:
 mask=np.array([q['type']==group for q in questions]);s['type_breakdown'][group]=dict(questions=int(mask.sum()),conditions={name:dict(EM=float(em[mask,:,k].mean()),F1=float(f1[mask,:,k].mean()),service_tokens=float(tok[mask,:,k].mean())) for k,name in enumerate(names)})
result=audit.audit(arc,vault,s,questions)
broken=copy.deepcopy(s);broken['type_breakdown']['synthetic_a']['conditions']['G1']['F1']+=.1
try:audit.audit(arc,vault,broken,questions)
except AssertionError:rejected=True
else:rejected=False
assert rejected
receipt=dict(status='PASS_ARTIFICIAL_RECONSTRUCTION_AND_TYPE_CORRUPTION_TEST',checks=result['checks'],benchmark_data=False,new_GPU_hours=0,corrupted_type_metric_rejected=rejected,fixture='Repeated engineering-only traces with synthetic references; not a scientific dataset')
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
