"""Exposed historical G2 response-stage measurements; no inference or gold files."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,sys,time
import numpy as np

NODES=[f'r{r}:w{w}' for r in (1,2) for w in range(4)]+['aggregate']
EDGES=[(s,d) for s in range(4) for d in range(4) if s!=d]
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def canonical_number(value):
    if value is None:return 'PARSE_NONE'
    if value==0:return 'NUMBER:0'
    text=format(value,'f')
    if '.' in text:text=text.rstrip('0').rstrip('.')
    return 'NUMBER:'+text

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ROOT=a.root.resolve(strict=True);OUT=a.output.resolve(strict=True)
    assert OUT.is_relative_to(ROOT)
    SOURCE=ROOT/'artifacts/private/cycle2_20261113/format_mechanism_diagnostic'
    D2=ROOT/'results/autonomous_20261113/d2_20261113/28fb250d1c46/private/inputs.npz'
    start=time.time();cpu=time.process_time()
    assert not (OUT/'SCAN_STARTED.json').exists()
    inventory=json.loads((SOURCE/'INVENTORY.json').read_bytes())
    raw=D2.read_bytes();assert sha(raw)==inventory['d2_inputs_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        arrays={k:a[k] for k in ('question_ids','row_question','masks','partition','left','right','quality_by_seed')}
    qids=arrays['question_ids'].tolist();rowq=arrays['row_question'];masks=arrays['masks'];part=arrays['partition']
    mapping={(qids[int(q)],tuple(m)):i for i,(q,m) in enumerate(zip(rowq,masks))}
    selected=np.isin(part[rowq],['train','tune'])
    index=(SOURCE/'SELECTED_EXECUTION_INDEX.jsonl').read_bytes();assert sha(index)==inventory['selected_index_sha256']
    shape=(len(masks),5,9)
    wire=np.full(shape,'',dtype='<U64');generation=np.full(shape,'',dtype='<U64');prompt=np.full(shape,'',dtype='<U64')
    answer=np.full(shape,'',dtype='<U64');numeric=np.full(shape[:2],'',dtype='<U64');tokens=np.zeros(shape,dtype=np.int32)
    present=np.zeros(shape[:2],dtype=bool)
    spath=ROOT/'src/graph_credit/scoring.py'
    assert sha(spath.read_bytes())=='da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e'
    spec=importlib.util.spec_from_file_location('historical_parse_only',spath);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    dump(OUT/'SCAN_STARTED.json',{'started_unix':start,'code_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha((OUT/'PROTOCOL.json').read_bytes()),'new_R1_read':False})
    count=0;readbytes=0
    for line in index.splitlines():
        idx=json.loads(line);path=ROOT/idx['execution_path'];assert path.resolve().is_relative_to(ROOT.resolve())
        raw=path.read_bytes();assert sha(raw)==idx['execution_sha256'];x=json.loads(raw)
        row=mapping[(idx['question_id'],tuple(idx['graph_mask']))];seed=idx['seed']-2701
        assert selected[row] and 0<=seed<5 and not present[row,seed]
        assert x['question_id']==idx['question_id'] and x['seed']==idx['seed'] and x['mask']==idx['graph_mask']
        present[row,seed]=True
        for k,node in enumerate(NODES):
            n=x['nodes'][node];g=n['generation']
            wire[row,seed,k]=sha(n['wire_message'].encode());generation[row,seed,k]=sha(g['text'].encode())
            answer[row,seed,k]=sha(n['parsed']['answer'].encode());prompt[row,seed,k]=n['input_fingerprint']
            tokens[row,seed,k]=g['input_tokens']+g['output_tokens']
        value=scorer.extract_answer(x['nodes']['aggregate']['generation']['text'])
        numeric[row,seed]=sha(canonical_number(value).encode())
        count+=1;readbytes+=len(raw)
        if count%2000==0:
            progress={'files':count,'total_files':len(index.splitlines()),'wall_seconds':time.time()-start}
            dump(OUT/'SCAN_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert present[selected].all() and not present[~selected].any()
    cache=OUT/'HASHED_NODE_CACHE.npz'
    np.savez_compressed(cache,present=present,wire=wire,generation=generation,prompt=prompt,answer=answer,numeric=numeric,tokens=tokens)
    left=arrays['left'].reshape(len(qids),2,12);right=arrays['right'].reshape(len(qids),2,12)
    quality=arrays['quality_by_seed'];rows=[];checks=0
    for q in np.flatnonzero(np.isin(part,['train','tune'])):
        for anchor in range(2):
            for edge,(_,dst) in enumerate(EDGES):
                l,r=int(left[q,anchor,edge]),int(right[q,anchor,edge]);receiver=4+dst
                assert masks[l,edge]==0 and masks[r,edge]==1 and np.sum(masks[l]!=masks[r])==1
                for s in range(5):
                    unaffected=[k for k in range(8) if k!=receiver]
                    assert np.array_equal(wire[l,s,unaffected],wire[r,s,unaffected]);checks+=1
                    equalwire=wire[l,s,receiver]==wire[r,s,receiver]
                    equalanswer=answer[l,s,receiver]==answer[r,s,receiver]
                    delta=float(quality[r,s]-quality[l,s]);aggregatechanged=numeric[l,s]!=numeric[r,s]
                    if equalwire:
                        assert prompt[l,s,8]==prompt[r,s,8] and generation[l,s,8]==generation[r,s,8] and delta==0;checks+=3
                    if not aggregatechanged:assert delta==0;checks+=1
                    category=0 if equalwire else (1 if equalanswer else 2)
                    rows.append((int(q),anchor,edge,s,category,int(aggregatechanged),delta,int(tokens[r,s,receiver]+tokens[r,s,8]),int(tokens[r,s,8]) if equalwire else 0))
    data=np.array(rows,dtype=float);np.save(OUT/'PRIVATE_MEASUREMENT_ROWS.npy',data,allow_pickle=False)
    names=['same_receiver_wire','changed_wire_same_receiver_answer','changed_receiver_answer']
    result={'status':'COMPLETE_EXPLORATORY_MECHANISM_DIAGNOSTIC','new_gpu_hours':0,'new_formal_rounds':0,'new_R1_read':False,
            'official_test_read':False,'source_index_sha256':sha(index),'cache_sha256':sha(cache.read_bytes()),'deterministic_chain_checks':checks,
            'raw_files':count,'raw_bytes':readbytes,'groups':{},'cost_scope':'marginal_suffix_only_not_complete_budget','candidate_admission':False}
    rng=np.random.default_rng(97173)
    for partition in ['train','tune']:
        ids=np.flatnonzero(part==partition);sub=data[np.isin(data[:,0],ids)];groups=[]
        for category,name in enumerate(names):
            cell=sub[sub[:,4]==category]
            groups.append({'category':name,'n':len(cell),'questions':len(np.unique(cell[:,0])),
                'event_rate':float(np.mean(cell[:,6]!=0)) if len(cell) else None,
                'aggregate_numeric_answer_change_rate':float(cell[:,5].mean()) if len(cell) else None,
                'signed_credit':float(cell[:,6].mean()) if len(cell) else None})
        contrasts=[]
        for q in ids:
            cell=sub[sub[:,0]==q];a=cell[cell[:,4]==2];b=cell[cell[:,4]==1]
            if len(a) and len(b):contrasts.append(float(np.mean(a[:,6]!=0)-np.mean(b[:,6]!=0)))
        arr=np.array(contrasts)
        ci=np.quantile(arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(axis=1),[.025,.975]).tolist() if len(arr) else None
        result['groups'][partition]={'cells':groups,'paired_answer_change_vs_answer_preserving_event_delta':float(arr.mean()) if len(arr) else None,
            'paired_questions':len(arr),'paired_95_percentile_ci':ci,
            'generic_exact_coalescence_suffix_token_saving_fraction':float(sub[:,8].sum()/sub[:,7].sum()),
            'not_a_complete_budget_saving':True}
    result['cost']={'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start}
    dump(OUT/'RESULTS.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
