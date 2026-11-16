"""Reproduce D15 inputs from audited parent archives; no labels or inference."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--repo',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();p=Path(args.private_root);r=Path(args.repo);out=Path(args.output)
read=lambda f:json.loads(f.read_bytes());canon=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'));digest=lambda v:hashlib.sha256(canon(v).encode()).hexdigest();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('d13',r/'experiments/cycle3_20261116/d13_abstention_replay/diagnose.py');d13=importlib.util.module_from_spec(spec);spec.loader.exec_module(d13)
out.mkdir(exist_ok=False);fields=('question_fingerprint','node','model','model_identity_sha256','messages','seed','max_new_tokens','config');ident=lambda req:{k:req[k] for k in fields};cache={};matrices={};jobs={};entries=[];sources=('U','V','R')
assert read(p/'d14_collection_v1/SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_SCALAR_RAW_SCORING'
for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
 for unit in (p/folder/'monitor/archive').glob('worker_*/unit_*'):
  inp=read(unit/'INPUT.json');mat=read(unit/'MATRIX.json');matrices[task,inp['index']]=(inp,mat)
  for row in mat['conditions']:
   node=row['nodes']['M'];req=ident(node['request']);key=digest(req);cache[key]=dict(request=req,generation=node['generation'],origin='PARENT',parent_folder=folder,unit=str(unit.relative_to(p/folder/'monitor/archive')),condition=row['condition'],seed=row['execution_seed'])
requests={j['job_id']:j['request'] for j in map(json.loads,(p/'d14_selective_input_v1/REQUESTS.jsonl').read_bytes().splitlines())}
for f in (p/'d14_collection_v1/archive').glob('worker_*/RESULTS.jsonl'):
 for row in map(json.loads,f.read_bytes().splitlines()):cache[row['job_id']]=dict(request=requests[row['job_id']],generation=row['generation'],origin='D14',result_file=str(f.relative_to(p/'d14_collection_v1/archive')))
for (task,index),(inp,mat) in sorted(matrices.items()):
 shock=int(hashlib.sha256(canon(['D15_SHOCK',inp['payload']['id']]).encode()).hexdigest(),16)%3
 for row in mat['conditions']:
  if row['condition']!='INDEPENDENT3_Q':continue
  original=row['nodes']['M']['request'];content0=json.loads(original['messages'][1]['content']);assert [v['source'] for v in content0['evidence_notes']]==list(sources)
  for mode in ['RAW','FILTERED']:
   for mask in range(8):
    req=copy.deepcopy(ident(original));content=copy.deepcopy(content0);selected=[n for i,n in enumerate(content['evidence_notes']) if mask&(1<<i)];content['evidence_notes']=[n for n in selected if mode=='RAW' or not d13.abstains(n['text'])];req['messages'][1]['content']=canon(content);key=digest(req);cached=cache.get(key)
    prefix=sum(row['nodes'][node]['generation']['input_tokens']+row['nodes'][node]['generation']['output_tokens'] for i,node in enumerate(sources) if mask&(1<<i))
    entry=dict(task=task,index=index,seed=row['execution_seed'],mode=mode,mask=mask,shock=shock,prefix_tokens=prefix,kept_sources=[n['source'] for n in content['evidence_notes']],job_id=key,cache_hit=cached is not None)
    if cached:assert cached['request']==req;entry['cached']=cached
    else:jobs[key]=dict(job_id=key,task=task,index=index,seed=row['execution_seed'],request=req)
    entries.append(entry)
assert len(entries)==6144;new=sorted(jobs.values(),key=lambda j:j['job_id']);(out/'REQUESTS.jsonl').write_text(''.join(canon(j)+'\n' for j in new),encoding='utf-8');(out/'ENTRIES.json').write_text(canon(entries)+'\n',encoding='utf-8')
receipt=dict(status='INPUT_ONLY_NO_NEW_INFERENCE',logical_entries=len(entries),unique_new_requests=len(new),cache_entries=sum(e['cache_hit'] for e in entries),questions_per_task=96,tasks=2,seeds=[26111671,26111672],masks=list(range(8)),modes=['RAW','FILTERED'],entries_sha256=sha(out/'ENTRIES.json'),requests_sha256=sha(out/'REQUESTS.jsonl'),planner_sha256=sha(Path(__file__)),uses_labels=False)
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

print(json.dumps(receipt))
