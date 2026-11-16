"""Rebuild each star intervention and verify every reused raw generation's provenance."""
import argparse,copy,hashlib,json,re
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);inputs=p/'d15_star_input_v1';entries=read(inputs/'ENTRIES.json');jobs={v['job_id']:v for v in map(json.loads,(inputs/'REQUESTS.jsonl').read_bytes().splitlines())};d14={v['job_id']:v['generation'] for f in (p/'d14_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())};fields=('question_fingerprint','node','model','model_identity_sha256','messages','seed','max_new_tokens','config');sources=['U','V','R'];seen=set();new=set();hits=0;matrices={};oldcache={}
 for task,folder,df in [('hotpot','annotation_formal7_v2','d13_hotpot_v1'),('2wiki','wiki_annotation_formal8_v2','d13_2wiki_v1')]:
  lexicon=read(p/df/'SUMMARY.json')['lexicon'];arc=p/folder/'monitor/archive';data={}
  for unit in arc.glob('worker_*/unit_*'):
   inp=read(unit/'INPUT.json');mat=read(unit/'MATRIX.json');rows={(v['execution_seed'],v['condition']):v for v in mat['conditions']};data[inp['index']]=(inp,rows);matrices[folder,str(unit.relative_to(arc))]=rows
   for row in rows.values():
    node=row['nodes']['M'];oldcache[digest({k:node['request'][k] for k in fields})]=node['generation']
  for e in (e for e in entries if e['task']==task):
   key=(task,e['index'],e['seed'],e['mode'],e['mask']);assert key not in seen;seen.add(key);inp,rows=data[e['index']];row=rows[e['seed'],'INDEPENDENT3_Q'];req={k:copy.deepcopy(row['nodes']['M']['request'][k]) for k in fields};content=json.loads(req['messages'][1]['content']);notes=content['evidence_notes'];selected=[]
   for bit,note in enumerate(notes):
    if not e['mask']&(1<<bit):continue
    text=' '.join(note['text'].casefold().replace('\u2019',"'").split());abstain=any(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',text) for w in lexicon)
    if e['mode']=='RAW' or not abstain:selected.append(note)
   content['evidence_notes']=selected;req['messages'][1]['content']=canon(content);h=digest(req);assert h==e['job_id'];assert e['kept_sources']==[n['source'] for n in selected];assert e['shock']==int(hashlib.sha256(canon(['D15_SHOCK',inp['payload']['id']]).encode()).hexdigest(),16)%3
   prefix=sum(row['nodes'][node]['generation']['input_tokens']+row['nodes'][node]['generation']['output_tokens'] for bit,node in enumerate(sources) if e['mask']&(1<<bit));assert prefix==e['prefix_tokens']
   if e['cache_hit']:
    cached=e['cached'];assert cached['request']==req
    if cached['origin']=='PARENT':source=matrices[cached['parent_folder'],cached['unit']][cached['seed'],cached['condition']]['nodes']['M'];assert source['generation']==cached['generation'] and all(source['request'][k]==req[k] for k in fields)
    else:assert cached['origin']=='D14' and d14[h]==cached['generation']
    hits+=1
   else:assert jobs[h]['request']==req;new.add(h)
 assert len(seen)==6144 and new==set(jobs) and len(new)==2181
 result=dict(status='PASS_INDEPENDENT_STAR_INPUT_AND_CACHE_AUDIT',logical_entries=len(seen),cache_entries=hits,unique_new_requests=len(new),producer_cost_charged_before_filter=True,gold_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
