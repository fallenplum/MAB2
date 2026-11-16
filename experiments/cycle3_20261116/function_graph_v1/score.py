"""Formal10 fixed aggregate scoring from a complete audited hidden execution table."""
import argparse,hashlib,json,collections,statistics
from pathlib import Path

def mean(x):return sum(x)/len(x)
def cost(rows):
 calls={};programs={}
 for row in rows:
  for rec in row['nodes'].values():calls.setdefault(rec['request_fingerprint'],rec)
  for pr in row['programs']:programs.setdefault(pr['program_sha256'],pr)
 by_model={}
 for model in ['qwen','mistral']:
  gs=[r['generation'] for r in calls.values() if r['model']==model]
  by_model[model]=dict(calls=len(gs),input_tokens=sum(g['input_tokens'] for g in gs),output_tokens=sum(g['output_tokens'] for g in gs),nominal_gpu_seconds=sum(g['gpu_seconds'] for g in gs),nominal_elapsed_seconds=sum(g['elapsed_seconds'] for g in gs))
 visible_seconds=0 if len(rows)==1 and rows[0]['condition'].startswith('SINGLE_') else sum(x['visible']['seconds'] for x in programs.values())
 return dict(tokens=sum(v['input_tokens']+v['output_tokens'] for v in by_model.values()),visible_execution_seconds=visible_seconds,unique_generation_calls=len(calls),unique_visible_programs=0 if len(rows)==1 and rows[0]['condition'].startswith('SINGLE_') else len(programs),by_model=by_model)
def selected_program(row):return row['programs'][row['selected_program']]
def choice(rows):return max(range(len(rows)),key=lambda i:(selected_program(rows[i])['visible']['passed'],-i))
def score(root,hidden):
 records=[];credits=[];sourcefiles={};matrices=sorted(Path(root).glob('worker_*/unit_*/MATRIX.json'));assert len(matrices)==16
 for matrix in matrices:
  raw=matrix.read_bytes();m=json.loads(raw);sourcefiles[str(matrix.relative_to(root))]=hashlib.sha256(raw).hexdigest();assert len(m['conditions'])==30 and m['hidden_test_access'] is False
  question=m['question_fingerprint'];unit_records={}
  for seed in m['execution_seeds']:
   rows={x['condition']:x for x in m['conditions'] if x['execution_seed']==seed};assert len(rows)==10
   for condition,row in rows.items():
    pr=selected_program(row);e=hidden[question][pr['program_sha256']];c=cost([row]);q=int(all(x['passed'] for x in e['cases']));u=q-5e-6*c['tokens']-1e-4*c['visible_execution_seconds']
    z=dict(question=question,seed=seed,condition=condition,quality=q,case_accuracy=mean([int(x['passed']) for x in e['cases']]),hidden_cases=len(e['cases']),visible=pr['visible']['passed']/2,utility=u,**c);records.append(z);unit_records[(seed,condition)]=z
   bok=rows['BOK3'];prefix=[]
   for program in bok['programs']:
    prefix.append(program)
    if program['visible']['passed']==2:break
   fps={v['source_request_fingerprint'] for v in prefix};partial={**bok,'programs':prefix,'nodes':{n:v for n,v in bok['nodes'].items() if v['request_fingerprint'] in fps}}
   c=cost([partial]);z={**unit_records[(seed,'BOK3')],**c,'condition':'BOK3_EARLY'};z['utility']=z['quality']-5e-6*c['tokens']-1e-4*c['visible_execution_seconds'];records.append(z);unit_records[(seed,'BOK3_EARLY')]=z
   for name,arms in [('VS_GRAPH',['G1','G1_MINUS_EU','G1_MINUS_EV']),('REUSE_HISTORY6',['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV'])]:
    rs=[rows[c] for c in arms];j=choice(rs);source=unit_records[(seed,arms[j])];c=cost(rs);z={**source,**c,'condition':name,'selected_condition':arms[j]};z['utility']=z['quality']-5e-6*c['tokens']-1e-4*c['visible_execution_seconds'];records.append(z);unit_records[(seed,name)]=z
   for edge in ['U','V']:
    vals={g:unit_records[(seed,g)]['quality']-unit_records[(seed,g+'_MINUS_E'+edge)]['quality'] for g in ['G0','G1']};v=unit_records[(seed,'G1')]['visible']-unit_records[(seed,'G1_MINUS_E'+edge)]['visible'];credits.append(dict(question=question,seed=seed,edge=edge,old=vals['G0'],current=vals['G1'],shift=vals['G1']-vals['G0'],visible=v,sign_agree=vals['G1']*v>0))
 seeds=sorted({z['seed'] for z in records});conditions=sorted({z['condition'] for z in records});summary={}
 for name in conditions:
  rs=[z for z in records if z['condition']==name];assert len(rs)==48
  summary[name]={k:mean([z[k] for z in rs]) for k in ['quality','case_accuracy','visible','utility','tokens','visible_execution_seconds','unique_generation_calls']}
  summary[name]['by_seed']={str(s):{k:mean([z[k] for z in rs if z['seed']==s]) for k in ['quality','utility','tokens']} for s in seeds}
  summary[name]['model_cost_totals']={model:{k:sum(z['by_model'][model][k] for z in rs) for k in ['calls','input_tokens','output_tokens','nominal_gpu_seconds','nominal_elapsed_seconds']} for model in ['qwen','mistral']}
 index={(z['question'],z['seed'],z['condition']):z for z in records};comparison={}
 for control in ['SINGLE_Q','SINGLE_C','BOK3','BOK3_EARLY','REPAIR3']:
  pairs=[(z,index[(z['question'],z['seed'],control)]) for z in records if z['condition']=='VS_GRAPH'];comparison[control]=dict(quality_difference=mean([a['quality']-b['quality'] for a,b in pairs]),utility_difference=mean([a['utility']-b['utility'] for a,b in pairs]),improved=sum(a['quality']>b['quality'] for a,b in pairs),harmed=sum(a['quality']<b['quality'] for a,b in pairs),equal=sum(a['quality']==b['quality'] for a,b in pairs),utility_by_seed={str(s):mean([a['utility']-b['utility'] for a,b in pairs if a['seed']==s]) for s in seeds})
 nonzero=[c for c in credits if c['current']];agreement=mean([int(c['sign_agree']) for c in nonzero]) if nonzero else None;current_tasks=len({c['question'] for c in credits if c['current']});shift_tasks=len({c['question'] for c in credits if c['shift']})
 gates=dict(coverage=len(records)==16*3*13,all_strong_utility=all(v['utility_difference']>=.01 for v in comparison.values()),raw_quality=all(v['quality_difference']>=-.02 for v in comparison.values()),seed_consistency=all(v>=0 for k in ['BOK3','BOK3_EARLY','REPAIR3'] for v in comparison[k]['utility_by_seed'].values()),current_credit_support=current_tasks>=4,credit_shift_support=shift_tasks>=4,visible_credit_decision_signal=agreement is not None and agreement>=.60 and len(nonzero)>=12 and current_tasks>=4)
 result=dict(status='PASS_EXPLORATORY_APPLICABILITY_ONLY' if all(gates.values()) else 'STOP_FORMAL10_APPLICABILITY_ROUTE',gates=gates,passed=all(gates.values()),tasks=16,seeds=3,conditions=summary,paired_comparisons=comparison,credit=dict(nonzero_current_events=len(nonzero),nonzero_current_tasks=current_tasks,nonzero_shift_tasks=shift_tasks,visible_hidden_sign_agreement=agreement),independent_confirmation=False,confirmation_labels_accessed=False,confidence_intervals_computed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_matrices_sha256=sourcefiles,limitation='Controlled APPS TRAIN function subset. Limited hidden suites, possible pretraining contamination, model-specific token utility and cached timing limitations apply. All physical collection and qualification occupancy reported separately.')
 return result,dict(records=records,credits=credits)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--hidden-results',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.out);d.mkdir(parents=True,exist_ok=True);result,private=score(a.archive,json.loads(Path(a.hidden_results).read_bytes())['results'])
 for name,value in [('SUMMARY.json',result),('PRIVATE_DECISION_TABLE.json',private)]:
  with (d/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k not in ['conditions','source_matrices_sha256']}))
