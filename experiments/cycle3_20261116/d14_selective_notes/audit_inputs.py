"""Independent reconstruction of selective-note requests, cache lineage and full prefix cost."""
import argparse,copy,hashlib,json,re
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.private_root);inp=p/'d14_selective_input_v1';entries=read(inp/'ENTRIES.json');jobs={v['job_id']:v for v in map(json.loads,(inp/'REQUESTS.jsonl').read_bytes().splitlines())}
 fields=('question_fingerprint','node','model','model_identity_sha256','messages','seed','max_new_tokens','config');parents={'G1_SELECTIVE':'G1','G1_RANDOM_MATCH':'G1','FLAT_SELECTIVE':'FLAT_ANNOTATION','INDEPENDENT_SELECTIVE':'INDEPENDENT3_Q'};new=set();seen=set();cache=0
 for task,folder,d13 in [('hotpot','annotation_formal7_v2','d13_hotpot_v1'),('2wiki','wiki_annotation_formal8_v2','d13_2wiki_v1')]:
  lexicon=read(p/d13/'SUMMARY.json')['lexicon'];arc=p/folder/'monitor/archive';units={read(u/'INPUT.json')['index']:u for u in arc.glob('worker_*/unit_*')}
  for e in [v for v in entries if v['task']==task]:
   key=(task,e['index'],e['seed'],e['variant']);assert key not in seen;seen.add(key);unit=units[e['index']];assert unit==arc/e['original_archive_unit'];data=read(unit/'INPUT.json');rows=read(unit/'MATRIX.json')['conditions'];row=next(v for v in rows if v['condition']==parents[e['variant']] and v['execution_seed']==e['seed']);node=row['nodes']['M'];original=node['request'];req={k:copy.deepcopy(original[k]) for k in fields};content=json.loads(req['messages'][1]['content']);notes=content['evidence_notes']
   bad=[]
   for n,note in enumerate(notes):
    normalized=' '.join(note['text'].casefold().replace('\u2019',"'").split())
    if any(re.search(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',normalized) is not None for phrase in lexicon):bad.append(n)
   if e['variant']=='G1_RANDOM_MATCH':bad=sorted(range(3),key=lambda i:hashlib.sha256(canon(['D14_RANDOM',261116141,data['payload']['id'],e['seed'],notes[i]['source']]).encode()).digest())[:len(bad)]
   content['evidence_notes']=[note for n,note in enumerate(notes) if n not in bad];req['messages'][1]['content']=canon(content);h=digest(req)
   assert h==e['generation_equivalence_sha256'] and e['removed_note_count']==len(bad);assert e['kept_sources']==[n['source'] for n in content['evidence_notes']];assert e['original_request_fingerprint']==node['request_fingerprint']
   assert e['prefix_tokens']==sum(v['generation']['input_tokens']+v['generation']['output_tokens'] for k,v in row['nodes'].items() if k!='M')
   possible=[n for row2 in rows for n in row2['nodes'].values() if all(n['request'].get(f)==req[f] for f in fields)]
   assert bool(possible)==e['cache_hit']
   if possible:assert any(n['generation']==e['cached']['generation'] and n['request_fingerprint']==e['cached']['source_request_fingerprint'] for n in possible);cache+=1
   else:assert jobs[h]['request']==req;new.add(h)
 assert len(seen)==1536 and new==set(jobs) and len(new)==669
 result=dict(status='PASS_INDEPENDENT_INPUT_CACHE_PREFIX_RECONSTRUCTION',logical_entries=len(seen),exact_cache_entries=cache,unique_new_requests=len(new),uses_labels=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
