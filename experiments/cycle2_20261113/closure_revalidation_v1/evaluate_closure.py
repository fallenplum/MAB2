"""Fixed raw-output scoring and hidden I/C/S replay, only after complete archive release."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(selection,root):
 from closure_operator import SCHEDULE,live_nodes
 from exogenous_path_operator import digest
 from factorial_operator import vote_choice
 rows=selection['questions'];seeds=selection['generation_seeds']
 assert len(rows)==640 and seeds==[26111571,26111572,26111573,26111574,26111575]
 assert len({r['id'] for r in rows})==640 and len({r['question_group_sha256'] for r in rows})==640
 records=[];predictions=[];finish={'eos':0,'length':0};blank=0
 for i,row in enumerate(rows):
  paths=list(root.glob(f'worker_*/unit_{i:03d}/MATRIX.json'));assert len(paths)==1
  m=read(paths[0]);inp=read(paths[0].parent/'INPUT.json')
  assert row['index']==inp['index']==i and row['episode']==i//80 and row['role']==('I' if i%80<8 else 'C' if i%80<16 else 'S')
  assert row['removed_upstream']==selection['episode_shocks'][i//80]==inp['shock']==m['removed_upstream']
  assert inp['payload']['id']==row['id'] and digest(inp['payload'])==m['question_fingerprint']
  assert m['status']=='COMPLETE_UNSCORED' and m['complete_terminal_conditions']==45
  assert m['execution_seeds']==seeds and sum(v['calls'] for v in m['logical_completed_node_cost_by_model'].values())==155
  assert [c['key'] for c in m['conditions']]==[f'{s}/{n}' for s in seeds for n in SCHEDULE]
  perseed=[];answers=[]
  for s in seeds:
   rs={};ps={}
   for c in [c for c in m['conditions'] if c['execution_seed']==s]:
    assert c['status']=='COMPLETE';n=c['condition'];node='M'
    if n=='VOTE_Q3':
     order=['M','U','V'];node=order[vote_choice([c['nodes'][v]['generation']['text'] for v in order])];assert node==c['selected_node']
    if n.startswith('G'):
     assert list(c['nodes'])==list(live_nodes(c['edges'])) or set(c['nodes'])==set(live_nodes(c['edges']))
     assert c['executed_nodes']==list(live_nodes(c['edges']))
    g=c['nodes'][node]['generation'];assert c['answer']==g['text'].strip() and g['finish_reason'] in finish
    finish[g['finish_reason']]+=1;blank+=not bool(c['answer']);ps[n]=c['answer']
    rs[n]={'key':c['key'],'condition':n,'nodes':{node:{'request_fingerprint':v['request_fingerprint'],'model':v['model'],'generation':{k:v['generation'][k] for k in ['input_tokens','output_tokens','gpu_seconds']}} for node,v in c['nodes'].items()}}
   perseed.append(rs);answers.append(ps)
  records.append(perseed);predictions.append(answers)
 return records,predictions,{'finish_counts':finish,'blank_final_outputs':blank}

def evaluate(selection,records,predictions,vault):
 from hotpot_support import answer_metrics
 from closure_operator import SCHEDULE
 from closure_replay_policies import choose_episode,evaluate_choices,summarize,condition_cost
 assert set(vault)=={r['id'] for r in selection['questions']}
 em=[];f1=[]
 for row,preds in zip(selection['questions'],predictions):
  es=[];fs=[]
  for perseed in preds:
   scores={n:answer_metrics(perseed[n],vault[row['id']]) for n in SCHEDULE}
   es.append({n:scores[n]['em'] for n in SCHEDULE});fs.append({n:scores[n]['f1'] for n in SCHEDULE})
  em.append(es);f1.append(fs)
 allrows=[];traces=[]
 for ep in range(8):
  start=ep*80;changed='UV'.index(selection['episode_shocks'][ep])
  for s,seed in enumerate(selection['generation_seeds']):
   ir=[records[i][s] for i in range(start,start+8)];cr=[records[i][s] for i in range(start+8,start+16)]
   ie=[em[i][s] for i in range(start,start+8)];ce=[em[i][s] for i in range(start+8,start+16)]
   for change,ci in [('change',changed),('no_change',None)]:
    choices=choose_episode(ir,cr,ie,ce,ci,seed+1000)
    # Every method's action is now frozen; S values enter only the evaluator.
    values=evaluate_choices(choices,[records[i][s] for i in range(start+16,start+80)],
      [em[i][s] for i in range(start+16,start+80)],[f1[i][s] for i in range(start+16,start+80)],ci)
    for method,v in values.items():allrows.append({'episode':ep,'seed_index':s,'generation_seed':seed,'condition':change,'method':method,**v})
    traces.append({'episode':ep,'seed_index':s,'condition':change,'choices':choices})
 result=summarize(allrows)
 result['all_condition_seed_accuracy']={n:[float(np.mean([em[i][s][n] for i in range(640)])) for s in range(5)] for n in SCHEDULE}
 result['service_condition_seed_accuracy']={n:[float(np.mean([em[i][s][n] for i in range(640) if i%80>=16])) for s in range(5)] for n in SCHEDULE}
 result['task']='HotpotQA distractor train; all active nodes full context; eight disjoint I8/C8/S64 episodes'
 return result,{'ids':[r['id'] for r in selection['questions']],'em':em,'f1':f1,'records':records,'rows':allrows,'policy_traces':traces}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--approval',required=True);ap.add_argument('--approval-sha',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert sha(a.approval)==a.approval_sha;p=read(a.approval);assert p['decision']=='APPROVED_FROZEN_CLOSURE_ICS_SCORING'
 for path,pin in p['source_pins'].items():assert sha(path)==pin
 assert p['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 for d in reversed(p['import_paths']):sys.path.insert(0,d)
 for name in ['selection','archive_manifest','replay_receipt','cost_closure','vault']:
  spec=p[name];assert sha(spec['path'])==spec['sha256']
 assert read(p['replay_receipt']['path'])['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY'
 assert read(p['replay_receipt']['path'])['completed_units']==640 and read(p['cost_closure']['path'])['actual_cost_closed']
 root=Path(p['archive_root']);manifest=read(p['archive_manifest']['path'])
 for rel,pin in manifest.items():
  f=(root/rel).resolve();assert f.is_relative_to(root.resolve()) and sha(f)==pin
 selection=read(p['selection']['path']);records,predictions,meta=collect(selection,root)
 vault=read(p['vault']['path']);result,private=evaluate(selection,records,predictions,vault);result.update(meta)
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 for n,obj in [('SUMMARY.json',result),('PRIVATE_REPLAY.json',private)]:
  with (out/n).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 receipt={'status':'COMPLETE_FIXED_ICS_POINT_EVALUATION','approval_sha256':a.approval_sha,'summary_sha256':sha(out/'SUMMARY.json'),'private_replay_sha256':sha(out/'PRIVATE_REPLAY.json'),'new_GPU_calls':0,'bootstrap_runs':0}
 (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':receipt['status'],'screen_action':result['screen_action']}))
if __name__=='__main__':main()
