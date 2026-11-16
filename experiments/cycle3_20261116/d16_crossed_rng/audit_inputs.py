"""Audit crossing: only terminal RNG changes; source messages, graph and costs remain fixed."""
from pathlib import Path
import argparse,hashlib,json
def read(p):return json.loads(Path(p).read_bytes())
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);old=read(p/'d15_star_input_v1/ENTRIES.json');oldjobs={v['job_id']:v['request'] for v in map(json.loads,(p/'d15_star_input_v1/REQUESTS.jsonl').read_bytes().splitlines())};oldgens={v['job_id']:v['generation'] for f in (p/'d15_star_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};cache={};parents={};ids={}
 for e in old:
  req=e['cached']['request'] if e['cache_hit'] else oldjobs[e['job_id']];g=e['cached']['generation'] if e['cache_hit'] else oldgens[e['job_id']];cache[e['job_id']]=(req,g)
  if e['mode']=='FILTERED':parents[e['task'],e['index'],e['seed'],e['mask']]=e
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  for unit in (p/folder/'monitor/archive').glob('worker_*/unit_*'):
   inp=read(unit/'INPUT.json');ids[task,inp['index']]=inp['payload']['id']
 inp=p/'d16_crossed_input_v1';entries=read(inp/'ENTRIES.json');jobs={v['job_id']:v for v in map(json.loads,(inp/'REQUESTS.jsonl').read_bytes().splitlines())};seen=set();new=set();hits=0;terminal=[26111671,26111672,261116161,261116162]
 for e in entries:
  key=(e['task'],e['index'],e['source_seed'],e['terminal_seed'],e['mask']);assert key not in seen;seen.add(key);parent=parents[e['task'],e['index'],e['source_seed'],e['mask']];assert e['source_D15_job_id']==parent['job_id'] and e['prefix_tokens']==parent['prefix_tokens'] and e['shock']==parent['shock'];req=dict(cache[parent['job_id']][0]);qid=ids[e['task'],e['index']];derived=[int(digest(['exogenous_path_rng_v1',qid,seed,'M'])[:16],16)%(2**31-1) for seed in terminal];assert len(set(derived))==4;req['seed']=derived[terminal.index(e['terminal_seed'])];h=digest(req);assert h==e['job_id']
  assert e['cache_hit']==(h in cache)
  if e['cache_hit']:assert e['cached']['request']==req and e['cached']['generation']==cache[h][1];hits+=1
  else:assert jobs[h]['request']==req and jobs[h]['seed']==e['terminal_seed'];new.add(h)
 assert len(seen)==12288 and new==set(jobs) and len(new)==6694
 result=dict(status='PASS_INDEPENDENT_CROSSED_INPUT_AUDIT',logical_outcomes=len(seen),unique_new_requests=len(new),exact_cache_entries=hits,only_terminal_seed_changed=True,all_included_producers_charged=True,gold_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
