"""Post-outcome exploratory attribution; support labels never enter a policy."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');p=root/'artifacts/private/cycle3_20261116';f=p/'evidence_formal6_v1'
assert json.loads((f/'SCORING_INDEPENDENT_AUDIT.json').read_bytes())['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
rows=[json.loads(x) for x in (p/'evidence_input_v1/QUESTIONS.jsonl').read_bytes().splitlines()];ids=[x['id'] for x in rows];wanted=pa.array(ids,type=pa.string())
desc=json.loads((root/'artifacts/private/cycle2_20261113/hotpot_selection_preparation/SOURCE_DESCRIPTOR_FROZEN.json').read_bytes());sf={}
for file in desc['parquet_files']:
 for batch in pq.ParquetFile(file['path']).iter_batches(batch_size=4096,columns=['id','supporting_facts']):
  t=pa.Table.from_batches([batch]);t=t.filter(pc.is_in(t['id'],value_set=wanted))
  for row in t.to_pylist():assert row['id'] not in sf;sf[row['id']]=row['supporting_facts']
assert set(sf)==set(ids)
out=p/'d10_support_coverage_v1';out.mkdir(exist_ok=False);(out/'PRIVATE_SUPPORT_LABELS.json').write_text(json.dumps(sf,ensure_ascii=False)+'\n',encoding='utf-8')
arrays=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz');em=arrays['em'];names=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_EVIDENCE','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3'];coverage=np.zeros((96,2,7),bool);selected_counts=np.zeros_like(coverage,int);loss=Counter();source_miss=Counter();unmatched=[]
for unit in (f/'monitor/archive').glob('worker_*/unit_*'):
 inp=json.loads((unit/'INPUT.json').read_bytes());i=inp['index'];row=rows[i];truth=set(sf[row['id']]['title']);titles=row['context']['title'];goldids={f'D{j:02d}' for j,t in enumerate(titles) if t in truth}
 if len({titles[int(d[1:])] for d in goldids})!=len(truth):unmatched.append(i)
 mat=json.loads((unit/'MATRIX.json').read_bytes());by={(c['execution_seed'],c['condition']):c for c in mat['conditions']}
 for j,seed in enumerate([26111661,26111662]):
  for k,name in enumerate(names[:7]):
   c=by[seed,name];present=set(c['terminal_evidence_ids']);coverage[i,j,k]=goldids<=present;selected_counts[i,j,k]=len(present)
  c=by[seed,'G1'];cover=bool(coverage[i,j,3]);key='all_support_retained' if cover else 'support_missing'
  loss[key+'_observations']+=1
  if em[i,j,7]==1 and em[i,j,3]==0:loss[key+'_destructions_vs_single_Q']+=1
  if em[i,j,7]==0 and em[i,j,3]==1:loss[key+'_corrections_vs_single_Q']+=1
  for node in ['U','V','R']:
   n=c['nodes'][node];available=set(n['available_evidence_ids']);chosen=set(n['evidence_ids']);source_miss[node+'_available_gold_docs']+=len(goldids&available);source_miss[node+'_dropped_available_gold_docs']+=len((goldids&available)-chosen)
summary=dict(status='COMPLETE_EXPOSED_POST_OUTCOME_SUPPORT_DIAGNOSTIC',preregistered_before_outcomes=False,question_groups=96,seeds=2,unmatched_support_title_questions=len(unmatched),conditions={name:dict(all_support_title_coverage=float(coverage[:,:,k].mean()),selected_document_count=float(selected_counts[:,:,k].mean()),EM_when_all_support_retained=float(em[:,:,k][coverage[:,:,k]].mean()) if coverage[:,:,k].any() else None,EM_when_support_missing=float(em[:,:,k][~coverage[:,:,k]].mean()) if (~coverage[:,:,k]).any() else None) for k,name in enumerate(names[:7])},paired_error_decomposition=dict(loss),selector_support_drops=dict(source_miss),new_GPU_hours=0,policy_access_to_support_labels=False,limitations=['Post-outcome exploratory diagnostic, not causal mediation or preregistered confirmation','Gold support title coverage is an evaluator-only upper bound and unavailable to a deployed method','Question groups and outputs already exposed; no gate changes or reuse as independent evidence','Raw support labels persisted privately only'])
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(out/'PRIVATE_COVERAGE.npz',coverage=coverage,selected_counts=selected_counts);print(json.dumps(summary))
