"""Independent raw hidden-return, coverage and aggregate arithmetic audit."""
import argparse,json,hashlib,math,collections
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(v):
 tag,value=v
 if tag=='atom':return value
 if tag=='list':return [decode(x) for x in value]
 if tag=='tuple':return tuple(decode(x) for x in value)
 if tag=='dict':return {decode(k):decode(x) for k,x in value}
 raise AssertionError('Unknown typed return')
def reference(result,expected):
 # Literal author comparison reconstructed separately from the execution adapter.
 try:
  if isinstance(expected,dict):expected=[dict((int(k),v) for k,v in expected.items())]
 except (ValueError,TypeError):pass
 try:
  if isinstance(expected[0],dict):expected=[dict((int(k),v) for k,v in expected[0].items())]
 except (ValueError,TypeError,IndexError,KeyError):pass
 if isinstance(result,tuple):result=list(result)
 correct=result==expected
 if isinstance(expected,list) and len(expected)>0:correct=correct or result==expected[0]
 try:
  if isinstance(result[0],tuple):correct=correct or [list(x) for x in result]==expected[0]
 except (ValueError,TypeError,IndexError,KeyError):pass
 return bool(correct)
def audit(archive,hidden_dir,labels,score_dir):
 archive=Path(archive);hidden_dir=Path(hidden_dir);score_dir=Path(score_dir);checks=0
 for root in [archive,hidden_dir]:
  for name,digest in read(root/'ARCHIVE_MANIFEST.json').items():assert sha(root/name)==digest;checks+=1
 data=read(hidden_dir/'HIDDEN_RESULTS.json');assert data['metadata']['hidden_sha256']==sha(labels);assert data['metadata']['archive_sha256']==sha(archive/'ARCHIVE_MANIFEST.json');assert data['metadata']['model_calls']==data['metadata']['GPU_hours']==0
 labels={x['id']:x for x in read(labels)};assert len(labels)==16
 table=read(score_dir/'PRIVATE_DECISION_TABLE.json');summary=read(score_dir/'SUMMARY.json');records=table['records'];assert len(records)==624
 index={(x['question'],x['seed'],x['condition']):x for x in records};assert len(index)==624
 statuses=collections.Counter();seen=set();case_count=0;program_count=0;seconds=0
 for f in sorted(archive.glob('worker_*/unit_*/MATRIX.json')):
  m=read(f);payload=read(f.parent/'INPUT.json')['payload'];identity=payload['id'];assert identity not in seen;seen.add(identity);q=m['question_fingerprint'];expected=labels[identity]['hidden_cases'];raw=data['results'][q]
  programs={pr['program_sha256']:pr for row in m['conditions'] for pr in row['programs']};assert set(raw)==set(programs)
  quality={}
  for key,program in programs.items():
   assert key==hashlib.sha256(program['code'].encode()).hexdigest();cases=raw[key]['cases'];assert len(cases)==len(expected)>=2
   for case,(_,gold) in zip(cases,expected):
    passed=case['status']=='OK' and case.get('returned') and reference(decode(case['typed_return']),json.loads(gold));assert bool(case['passed'])==bool(passed);statuses[case['status']]+=1;assert math.isfinite(case['elapsed_seconds']) and case['elapsed_seconds']>=0;seconds+=case['elapsed_seconds'];case_count+=1;checks+=3
   quality[key]=int(all(c['passed'] for c in cases));program_count+=1
  for row in m['conditions']:
   got=index[(q,row['execution_seed'],row['condition'])];selected=row['programs'][row['selected_program']]['program_sha256'];assert got['quality']==quality[selected];assert got['hidden_cases']==len(expected);checks+=2
 assert seen==set(labels) and len(data['results'])==16
 assert case_count==data['metadata']['case_executions'] and program_count==data['metadata']['unique_programs'];assert math.isclose(seconds,data['metadata']['actual_execution_seconds'],abs_tol=1e-8)
 for row in records:
  assert math.isclose(row['utility'],row['quality']-row['tokens']/200000-row['visible_execution_seconds']/10000,abs_tol=1e-12);assert row['tokens']==sum(v['input_tokens']+v['output_tokens'] for v in row['by_model'].values());checks+=2
 for condition,aggregate in summary['conditions'].items():
  rs=[v for v in records if v['condition']==condition];assert len(rs)==48
  for key in ['quality','case_accuracy','visible','utility','tokens','visible_execution_seconds','unique_generation_calls']:
   assert math.isclose(aggregate[key],sum(v[key] for v in rs)/48,abs_tol=1e-10);checks+=1
  for seed in [261117101,261117102,261117103]:
   ss=[v for v in rs if v['seed']==seed];assert len(ss)==16
   for key in ['quality','utility','tokens']:assert math.isclose(aggregate['by_seed'][str(seed)][key],sum(v[key] for v in ss)/16,abs_tol=1e-10);checks+=1
 for control,pair in summary['paired_comparisons'].items():
  pairs=[(v,index[(v['question'],v['seed'],control)]) for v in records if v['condition']=='VS_GRAPH'];assert pair['improved']==sum(a['quality']>b['quality'] for a,b in pairs);assert pair['harmed']==sum(a['quality']<b['quality'] for a,b in pairs)
  for key in ['quality','utility']:assert math.isclose(pair[key+'_difference'],sum(a[key]-b[key] for a,b in pairs)/48,abs_tol=1e-12);checks+=1
 return dict(status='PASS_INDEPENDENT_HIDDEN_RETURN_AND_AGGREGATE_AUDIT',tasks=16,case_executions=case_count,unique_programs=program_count,scalar_checks=checks,execution_status_counts=dict(statuses),actual_hidden_execution_seconds=seconds,hidden_result_sha256=sha(hidden_dir/'HIDDEN_RESULTS.json'),summary_sha256=sha(score_dir/'SUMMARY.json'),source_sha256=sha(__file__),confirmation_labels_accessed=False,confidence_intervals_computed=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--hidden-dir',required=True);p.add_argument('--labels',required=True);p.add_argument('--score-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();result=audit(a.archive,a.hidden_dir,a.labels,a.score_dir)
 with open(a.out,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
