"""Reproduce frozen crossed-RNG input from audited D15 archives; no labels/inference."""
from pathlib import Path
import argparse,copy,hashlib,json
ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();p=Path(args.private_root);inp=Path(args.output)
read=lambda f:json.loads(f.read_bytes());canon=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'));digest=lambda v:hashlib.sha256(canon(v).encode()).hexdigest();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();source_seeds=[26111671,26111672];terminal_seeds=[26111671,26111672,261116161,261116162]
assert read(p/'d15_star_collection_v1/SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_RAW_SCALAR_FACTORIAL_AUDIT';inp.mkdir(exist_ok=False)
oldentries=read(p/'d15_star_input_v1/ENTRIES.json');oldjobs={v['job_id']:v['request'] for v in map(json.loads,(p/'d15_star_input_v1/REQUESTS.jsonl').read_bytes().splitlines())};oldgens={v['job_id']:v['generation'] for f in (p/'d15_star_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};cache={};ids={}
for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
 for unit in (p/folder/'monitor/archive').glob('worker_*/unit_*'):
  x=read(unit/'INPUT.json');ids[task,x['index']]=x['payload']['id']
for e in oldentries:
 req=e['cached']['request'] if e['cache_hit'] else oldjobs[e['job_id']];g=e['cached']['generation'] if e['cache_hit'] else oldgens[e['job_id']];assert digest(req)==e['job_id'];cache[e['job_id']]=dict(request=req,generation=g)
entries=[];jobs={}
for e in oldentries:
 if e['mode']!='FILTERED':continue
 original=cache[e['job_id']]['request'];qid=ids[e['task'],e['index']];derive=lambda seed:int(digest(['exogenous_path_rng_v1',qid,seed,'M'])[:16],16)%(2**31-1);assert original['seed']==derive(e['seed'])
 for terminal in terminal_seeds:
  req=copy.deepcopy(original);req['seed']=derive(terminal);key=digest(req);cached=cache.get(key);entry=dict(task=e['task'],index=e['index'],source_seed=e['seed'],terminal_seed=terminal,mask=e['mask'],shock=e['shock'],prefix_tokens=e['prefix_tokens'],job_id=key,cache_hit=cached is not None,source_D15_job_id=e['job_id'])
  if cached:entry['cached']=cached
  else:jobs[key]=dict(job_id=key,task=e['task'],index=e['index'],seed=terminal,request=req)
  entries.append(entry)
assert len(entries)==12288;new=sorted(jobs.values(),key=lambda j:j['job_id']);(inp/'ENTRIES.json').write_text(canon(entries)+'\n',encoding='utf-8');(inp/'REQUESTS.jsonl').write_text(''.join(canon(j)+'\n' for j in new),encoding='utf-8')
receipt=dict(status='INPUT_ONLY_CROSSED_RNG_NO_LABELS',logical_outcomes=len(entries),unique_new_requests=len(new),cache_entries=sum(v['cache_hit'] for v in entries),questions_per_task=96,tasks=2,source_seeds=source_seeds,terminal_seeds=terminal_seeds,entries_sha256=sha(inp/'ENTRIES.json'),requests_sha256=sha(inp/'REQUESTS.jsonl'),source_sha256=sha(Path(__file__)))
for f in [inp/'RECEIPT.json']:f.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

print(json.dumps(receipt))
