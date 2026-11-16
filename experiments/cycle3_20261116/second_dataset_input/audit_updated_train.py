"""Audit corrected train inputs without using supervision in analysis.

JSON parsing necessarily decodes each complete record, including transient label
values. Only explicitly allowed input fields and label field presence/types leave
the parser; no answer/support/evidence values are inspected, emitted or scored.
"""
from pathlib import Path
from collections import Counter
import argparse, ast, hashlib, io, json, sys, time

def records(stream, chunk_size=1024*1024):
    decoder=json.JSONDecoder();buf='';eof=False
    def fill():
        nonlocal buf,eof
        part=stream.read(chunk_size)
        if not part:eof=True
        buf+=part
    def whitespace():
        nonlocal buf
        while True:
            buf=buf.lstrip()
            if buf or eof:return
            fill()
    fill();whitespace();assert buf.startswith('['),'Expected JSON array';buf=buf[1:]
    first=True
    while True:
        whitespace()
        if buf.startswith(']'):
            buf=buf[1:];whitespace();assert eof and not buf,'Trailing content';return
        if not first:
            assert buf.startswith(','),'Missing record separator';buf=buf[1:];whitespace()
        while True:
            try:
                value,end=decoder.raw_decode(buf);break
            except json.JSONDecodeError:
                if eof:raise
                fill()
        buf=buf[end:];assert isinstance(value,dict),'Expected record object'
        yield value
        value=None;first=False

def input_projection(row):
    q=row['question'];rid=row['_id'];context=row['context']
    assert isinstance(q,str) and q.strip() and isinstance(rid,str) and rid
    assert isinstance(context,list) and all(isinstance(x,list) and len(x)==2 and
        isinstance(x[0],str) and isinstance(x[1],list) and all(isinstance(s,str) for s in x[1]) for x in context)
    return {'_id':rid,'question':q,'context':context}

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def test():
    example=[{'_id':'x','question':'Chinese 测试 ] \\"','context':[['t',['a\\b','[]']]],'answer':'UNEXPOSED'},
             {'_id':'y','question':'q','context':[],'answer_id':'INTERNAL'}]
    raw=json.dumps(example,ensure_ascii=False)
    for size in [1,2,7,1024]:assert list(records(io.StringIO(raw),size))==example
    for broken in ['[{}', '[{},]', '[{}]x','[{} {}]']:
        try:list(records(io.StringIO(broken),2))
        except (AssertionError,json.JSONDecodeError):pass
        else:raise AssertionError('Malformed JSON accepted')
    assert set(input_projection(example[0]))=={'_id','question','context'}
    print('PASS_STREAMING_PARSER_AND_LABEL_PROJECTION_SYNTHETIC_TESTS')

def main():
    import pyarrow.parquet as pq
    import numpy as np
    root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';p=root/'artifacts/private/cycle3_20261116'
    sys.path.insert(0,str(r/'experiments/cycle2_20261113/hotpot_selection_v1'))
    import select_hotpot as h
    source=root/'data/2wiki_author_ids_april7_v1';desc=json.loads((source/'SOURCE_DESCRIPTOR_RESUMED.json').read_bytes())
    expected=next(x['sha256'] for x in desc['downloaded'] if x['member']=='train.json')
    assert desc['status']=='SELECTED_AUTHOR_TRAIN_AND_ALIAS_HASHED_UNOPENED' and file_sha(source/'train.json')==expected
    oldfile=root/'data/2wiki_author_train_v1/train.parquet';oldsha='a0f464f1604edf40ad990b605c61c03d56911c106d09ad32becf6afcf3201d52'
    assert file_sha(oldfile)==oldsha
    old={}
    for batch in pq.ParquetFile(oldfile).iter_batches(batch_size=128,columns=['_id','question','context']):
        for row in batch.to_pylist():
            try:row['context']=json.loads(row['context'])
            except json.JSONDecodeError:row['context']=ast.literal_eval(row['context'])
            v=input_projection(row);assert v['_id'] not in old;old[v['_id']]=(digest(v),h.question_group_sha256(v['question']))
    exposures=json.loads((p/'EXPOSURES_UNION_AFTER_FORMAL7_SELECTION.json').read_bytes());exposed=set(exposures['question_group_hashes'])
    out=p/'2wiki_updated_input_audit_v1';out.mkdir(exist_ok=False)
    ids=set();groups=set();counts=Counter();fields=Counter();types=Counter();lengths=[];overlap=0;same=0;changed=0;new=0;question_changed=0;started=time.time()
    with (source/'train.json').open(encoding='utf-8') as stream:
        for row in records(stream):
            for k in row:fields[k]+=1
            for k in ['answer','answer_id','evidences','evidences_id','supporting_facts']:
                if k in row:types[k+':'+type(row[k]).__name__]+=1
            v=input_projection(row);row=None;rid=v['_id'];assert rid not in ids;ids.add(rid)
            g=h.question_group_sha256(v['question']);groups.add(g);overlap+=int(g in exposed)
            counts[len(v['context'])]+=1;lengths.append(sum(len(t)+sum(map(len,s)) for t,s in v['context']))
            if rid not in old:new+=1
            else:
                equal=digest(v)==old[rid][0];same+=int(equal);changed+=int(not equal);question_changed+=int(g!=old[rid][1])
            if len(ids)%25000==0:print(json.dumps(dict(audited_input_records=len(ids))),flush=True)
    receipt=dict(status='CORRECTED_TRAIN_INPUT_AUDIT_COMPLETE',source_sha256=expected,source_script_sha256=file_sha(Path(__file__)),
        rows=len(ids),unique_question_groups=len(groups),context_document_count_histogram=dict(counts),field_presence_counts=dict(fields),
        supervision_field_types_only=dict(types),context_character_quantiles=dict(zip(['min','p50','p90','max'],map(float,np.quantile(lengths,[0,.5,.9,1])))),
        prior_mirror_sha256=oldsha,inputs_identical_to_mirror=same,inputs_changed_from_mirror=changed,new_ids_vs_mirror=new,
        mirror_ids_absent=len(set(old)-ids),normalized_question_changes_on_shared_ids=question_changed,
        exact_question_overlap_with_known_MAB2_exposures=overlap,exposure_union_sha256=file_sha(p/'EXPOSURES_UNION_AFTER_FORMAL7_SELECTION.json'),
        retained_input_fields=['_id','question','context'],supervision_values_decoded_transiently_by_JSON_parser=True,
        supervision_values_inspected_or_emitted=False,labels_scored=False,alias_file_parsed=False,official_dev_test_members_opened=False,
        semantic_overlap='NOT_ASSESSED',unknown_cross_project_exposure='UNKNOWN',new_GPU_hours=0,elapsed_seconds=time.time()-started)
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (out/'QUESTION_GROUP_HASHES.json').write_text(json.dumps(sorted(groups))+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test',action='store_true');args=parser.parse_args()
    test() if args.test else main()
