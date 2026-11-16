"""Scalar re-score of raw D14 generations and frozen baseline comparisons."""
from pathlib import Path
import argparse,json,sys,hashlib,math
def read(p):return json.loads(Path(p).read_bytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'d14_collection_v1';summary=read(run/'scoring_v1/SUMMARY.json');entries=read(p/'d14_selective_input_v1/ENTRIES.json');generated={x['job_id']:x['generation'] for f in (run/'archive').glob('worker_*/RESULTS.jsonl') for x in map(json.loads,f.read_bytes().splitlines())};checks=0
 for task,folder,d13folder in [('hotpot','annotation_formal7_v2','d13_hotpot_v1'),('2wiki','wiki_annotation_formal8_v2','d13_2wiki_v1')]:
  formal=p/folder;approval=read(formal/'SCORING_RELEASE.json')
  for path in reversed(approval['import_paths']):sys.path.insert(0,path)
  from hotpot_support import answer_metrics
  vault=read(approval['pins']['vault']['path']);units={read(u/'INPUT.json')['index']:u for u in (formal/'monitor/archive').glob('worker_*/unit_*')};values={};seeds=[26111671,26111672]
  for e in entries:
   if e['task']!=task:continue
   unit=units[e['index']];gold=vault[read(unit/'INPUT.json')['payload']['id']];g=e['cached']['generation'] if e['cache_hit'] else generated[e['generation_equivalence_sha256']];scores=[answer_metrics(g['text'].strip(),gold1) for gold1 in (gold if task=='2wiki' else [gold])];em=max(v['em'] for v in scores);f1=max(v['f1'] for v in scores);cost=e['prefix_tokens']+g['input_tokens']+g['output_tokens'];values.setdefault(e['variant'],{})[e['index'],e['seed']]=(em,f1,cost,em-5e-6*cost)
  def near(x,y):
   nonlocal checks
   assert math.isclose(x,y,rel_tol=0,abs_tol=1e-12);checks+=1
  for name,rows in values.items():
   reported=summary['tasks'][task]['conditions'][name];assert len(rows)==192
   for k,field in enumerate(['EM','F1','service_tokens','service_utility']):near(sum(v[k] for v in rows.values())/192,reported[field])
   for j,seed in enumerate(seeds):
    for k,field in [(0,'EM_by_seed'),(3,'utility_by_seed')]:near(sum(v[k] for key,v in rows.items() if key[1]==seed)/96,reported[field][j])
  masks=read(p/d13folder/'PRIVATE_MASKS.json')
  for i,unit in units.items():
   gold=vault[read(unit/'INPUT.json')['payload']['id']];matrix=read(unit/'MATRIX.json');rows={(v['execution_seed'],v['condition']):v for v in matrix['conditions']}
   for j,seed in enumerate(seeds):
    for name in ['SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q','DROP_ALL_IF_ANY_ABSTAINS','IGNORE_ALL_NOTES']:
     d13=name in masks;selected='SINGLE_Q' if d13 and masks[name][i][j] else 'G1' if d13 else name;row=rows[seed,selected];scores=[answer_metrics(row['answer'],ans) for ans in (gold if task=='2wiki' else [gold])];em=max(v['em'] for v in scores);tokens=sum(v['input_tokens']+v['output_tokens'] for v in row['logical_service_cost_by_model'].values())
     if d13 and selected=='SINGLE_Q':tokens+=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for k,n in rows[seed,'G1']['nodes'].items() if k!='M')
     values.setdefault(('D13_' if d13 else '')+name,{})[i,seed]=(em,0,tokens,em-5e-6*tokens)
  nominee=values['G1_SELECTIVE'];passes=[]
  for control,reported in summary['tasks'][task]['comparisons'].items():
   other=values[control];delta={k:nominee[k][3]-other[k][3] for k in nominee};overall=sum(delta.values())/192;byseed=[sum(v for k,v in delta.items() if k[1]==seed)/96 for seed in seeds];near(overall,reported['utility_delta'])
   for j in range(2):near(byseed[j],reported['utility_delta_by_seed'][j])
   assert sum(nominee[k][0]>other[k][0] for k in nominee)==reported['corrections'];assert sum(nominee[k][0]<other[k][0] for k in nominee)==reported['destructions'];passed=overall>=.005 and min(byseed)>0;assert passed==reported['passed'];passes.append(passed)
  assert all(passes)==summary['tasks'][task]['passed']
 assert summary['passed_both_tasks']==all(v['passed'] for v in summary['tasks'].values())
 result=dict(status='PASS_INDEPENDENT_SCALAR_RAW_SCORING',numeric_checks=checks,question_seed_variant_observations=1536,tasks=2,scoring_summary_sha256=hashlib.sha256((run/'scoring_v1/SUMMARY.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_official_answer_normalizer=True)
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
