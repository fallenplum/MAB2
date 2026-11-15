"""Pinned train input projection; no benchmark label, score, or GPU access."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha',required=True);a=ap.parse_args()
 assert sha(a.plan)==a.plan_sha;plan=read(a.plan);assert plan['source_sha256']==sha(__file__) and plan['action']=='INPUT_ONLY_1024_FOR_CLOSURE'
 for item in plan['pins'].values():assert sha(item['path'])==item['sha256']
 sys.path.insert(0,plan['import_path']);import select_hotpot as h
 contract=read(plan['pins']['contract']['path']);assert contract['status']=='FROZEN_INPUT_ONLY' and contract['pool_questions']==1024 and contract['selected_questions']==640
 exp=read(plan['pins']['prior_exposures']['path']);ids=set(exp['ids']);groups=set(exp['question_group_hashes']);contents=set(exp['content_hashes'])
 for name in ['formal3','formal4']:
  for r in read(plan['pins'][name]['path'])['questions']:ids.add(r['id']);groups.add(r['question_group_sha256']);contents.add(r['content_sha256'])
 sourcepath=Path(plan['pins']['source_descriptor']['path']);paths=h._source_preflight(read(sourcepath),sourcepath)
 by_group={};seen=set();excluded=set();start=time.monotonic()
 for raw in h.iter_allowed_rows(paths):
  r=h.validate_row(raw);assert r['id'] not in seen;seen.add(r['id']);g=h.question_group_sha256(r['question']);by_group.setdefault(g,[]).append(r)
  if r['id'] in ids or g in groups or h.content_sha256(r) in contents:excluded.add(g)
 eligible=sorted([min(v,key=lambda r:r['id'].encode()) for g,v in by_group.items() if g not in excluded],key=lambda r:r['id'].encode())
 order=np.random.Generator(np.random.PCG64(261115701)).permutation(len(eligible))[:1024];pool=[eligible[int(i)] for i in order]
 assert len(pool)==1024
 out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
 with (out/'POOL.jsonl').open('xb') as f:
  for row in pool:f.write(h.canonical_bytes(row))
 union={'ids':sorted(ids),'question_group_hashes':sorted(groups),'content_hashes':sorted(contents),'known_records_scope':'Prior union plus complete formal3 and formal4 selection; all previous scores remain exposed','unknown_cross_project_exposure':'UNKNOWN'}
 put(out/'EXPOSURES_UNION.json',union)
 receipt={'status':'COMPLETE_INPUT_ONLY_POOL','source_rows':len(seen),'source_question_groups':len(by_group),'excluded_source_question_groups':len(excluded),'known_exposure_union_groups':len(groups),
  'eligible_unexposed_groups':len(eligible),'pool_questions':1024,'sampling_seed':261115701,'labels_projected':False,'supporting_facts_projected':False,'quality_seen':False,'GPU_hours':0,
  'pool_sha256':sha(out/'POOL.jsonl'),'exposures_sha256':sha(out/'EXPOSURES_UNION.json'),'source_sha256':sha(__file__),'plan_sha256':a.plan_sha,'seconds':time.monotonic()-start,
  'unknown_cross_project_exposure':'UNKNOWN','official_validation_test_opened':False}
 put(out/'RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
