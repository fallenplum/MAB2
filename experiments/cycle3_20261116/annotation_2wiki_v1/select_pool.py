"""Input-only deterministic second-task pool; no outcome-dependent selection."""
from pathlib import Path
import hashlib,json,sys,shlex,shutil,time
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';p=root/'artifacts/private/cycle3_20261116';d=r/'experiments/cycle3_20261116/annotation_2wiki_v1';out=p/'wiki_annotation_input_v1';out.mkdir(exist_ok=False);formal=p/'wiki_annotation_formal8_v1';formal.mkdir(exist_ok=False)
sys.path[:0]=[str(d),str(r/'experiments/cycle2_20261113/hotpot_selection_v1')]
from input_json import records,input_projection,file_sha
from select_hotpot import question_group_sha256
read=lambda q:json.loads(q.read_bytes())
def put(q,v):q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
train=root/'data/2wiki_author_ids_april7_v1/train.json';assert file_sha(train)=='b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13'
exposure=p/'EXPOSURES_UNION_AFTER_FORMAL7_SELECTION.json';groups=set(read(exposure)['question_group_hashes']);unique={};count=0
with train.open(encoding='utf-8') as stream:
    for row in records(stream):
        v=input_projection(row);row=None;count+=1;g=question_group_sha256(v['question'])
        if g in groups:continue
        rank=hashlib.sha256(('261116801:'+v['_id']).encode()).hexdigest();candidate=(rank,v['_id'])
        if g not in unique or candidate<unique[g]:unique[g]=candidate
ranked=sorted((rank,rid,g) for g,(rank,rid) in unique.items())[:512];assert len(ranked)==512
want={rid for rank,rid,g in ranked};selected={}
with train.open(encoding='utf-8') as stream:
    for row in records(stream):
        if row['_id'] in want:
            v=input_projection(row);assert v['_id'] not in selected and len(v['context'])==10
            selected[v['_id']]=dict(id=v['_id'],question=v['question'],context={'title':[t for t,s in v['context']],'sentences':[s for t,s in v['context']]},type=row['type'])
        row=None
assert set(selected)==want
rows=[selected[rid] for rank,rid,g in ranked]
with (out/'POOL.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
    for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
put(out/'POOL_MANIFEST.json',dict(seed=261116801,selection='ascending SHA256(seed colon original ID), normalized-question dedup then512 pool',
    source_sha256=file_sha(train),exposure_sha256=file_sha(exposure),rows_scanned=count,question_groups=[g for rank,rid,g in ranked],
    original_ids=[rid for rank,rid,g in ranked],pool_sha256=file_sha(out/'POOL.jsonl'),labels_used=False,JSON_parser_transiently_decodes_discarded_supervision=True))
req=read(p/'annotation_formal7_v2/DEPLOY_REQUEST.json');remote=req['remote_root'];code=remote+'/code/cycle3_evidence_annotation_v1';rp=remote+'/private/cycle3/wiki_annotation_input_v1'
plan=read(p/'annotation_formal7_v2/PLAN.json');put(out/'CAPACITY_PLAN.json',plan)
req.update(uploads=[dict(local=str(out/'POOL.jsonl'),remote=rp+'/POOL.jsonl',sha256=file_sha(out/'POOL.jsonl')),dict(local=str(out/'CAPACITY_PLAN.json'),remote=rp+'/CAPACITY_PLAN.json',sha256=file_sha(out/'CAPACITY_PLAN.json'))],
    commands=[remote+'/.venv/bin/python -B '+code+'/capacity_inputs.py --plan '+rp+'/CAPACITY_PLAN.json --questions '+rp+'/POOL.jsonl --output '+rp+'/capacity_pool'],
    downloads=[dict(remote=rp+'/capacity_pool/RECEIPT.json',local=str(out/'POOL_CAPACITY_RECEIPT.json'))])
put(out/'CAPACITY_REQUEST.json',req)
put(out/'SOURCE_DESCRIPTOR.json',dict(dataset_id='2wikimultihop/corrected_april7_2021',split='train',author_commit='13800e5be57df1b4040b9b1588c6c811779e69e9',
    train_file=dict(path=str(train),sha256=file_sha(train)),alias_file=dict(path=str(root/'data/2wiki_author_ids_april7_v1/id_aliases.json'),sha256='f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6')))
print(json.dumps(dict(status='INPUT_POOL512_READY_FOR_CPU_CAPACITY',rows_scanned=count,labels_used=False,new_GPU_calls=0)))
