"""One historical derived-cache screen. No raw text, gold, models, or Hotpot."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, itertools, json, time, zipfile
import numpy as np

D2_SHA = '7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'
ANSWER_CACHE_SHA = '2264edf673b50e1ec2f7a858b34ec33c1cb36620473285e969a6533395285759'
RESPONSE_RESULT_SHA = '575e4823518b2fa16dba2d31601f8797205d5ec4359050acf457dfba8a3efd80'
FORMAT_CACHE_SHA = '20f05a130271e136c50092c9c51d5c940bf7736f63881d9f7ec6db92dba17fc1'
EDGES = [(s,d) for s in range(4) for d in range(4) if s != d]
EMPTY = hashlib.sha256(b'').hexdigest()

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def emit(p, obj):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def selected_npy_rows(path, member, selected):
    """Decode selected rows only; discarded archive bytes are never interpreted."""
    selected = np.asarray(selected, dtype=np.int64)
    require(len(selected)>0 and np.all(np.diff(selected)>0), 'Row selection not strict')
    with zipfile.ZipFile(path) as archive, archive.open(member+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):
            shape, fort, dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):
            shape, fort, dtype=np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unsupported NPY header')
        require(not fort and not dtype.hasobject and len(shape)>=1, 'Unsafe NPY layout')
        require(selected[-1]<shape[0], 'Row outside array')
        rowbytes=int(np.prod(shape[1:], dtype=np.int64))*dtype.itemsize
        output=np.empty((len(selected),)+shape[1:], dtype=dtype)
        cursor=0
        for i,row in enumerate(selected):
            # seek on ZipExtFile decompresses/discards as needed but never decodes excluded values.
            f.seek((int(row)-cursor)*rowbytes, 1)
            block=f.read(rowbytes)
            require(len(block)==rowbytes,'Truncated selected row')
            output[i]=np.frombuffer(block,dtype=dtype).reshape(shape[1:])
            cursor=int(row)+1
    return output

def reconstruct_anchors(masks, left):
    # Each comparison e leaves every other bit equal to its anchor.
    n=len(left)
    anchors=np.empty((n,2,12), dtype=np.int8)
    for bit in range(12):
        others=[e for e in range(12) if e!=bit]
        values=masks[left[:,:,others],bit]
        require(np.all(values==values[:,:,:1]), 'Inconsistent anchor metadata')
        anchors[:,:,bit]=values[:,:,0]
    return anchors

def receiver_change(anchors):
    require(anchors.ndim==3 and anchors.shape[1:]==(2,12),'Anchor dimensions')
    out=np.zeros((len(anchors),12),dtype=bool)
    for e,(_,dest) in enumerate(EDGES):
        others=[j for j,(_,d) in enumerate(EDGES) if d==dest and j!=e]
        out[:,e]=np.any(anchors[:,0,others]!=anchors[:,1,others],axis=-1)
    return out

def theta_crossseed(drift):
    require(drift.ndim==3 and drift.shape[1:]==(12,4), 'Drift shape')
    require(np.isfinite(drift).all() and np.isin(drift,(-2,-1,0,1,2)).all(), 'Drift scale')
    return sum(drift[:,:,s]*drift[:,:,t] for s,t in itertools.combinations(range(4),2))/6.0

def classify(answer, bad):
    require(answer.shape[-1]==4 and bad.shape==answer.shape[:-1], 'Z shape')
    unknown=((bad & 15)!=0) | np.any(answer==EMPTY,axis=-1)
    disagreement=np.any(answer!=answer[...,:1],axis=-1)
    return np.where(unknown,-1,disagreement.astype(np.int8))

def summarize(z, dirty, theta, seed):
    require(z.shape==(len(theta),) and dirty.shape==theta.shape and theta.shape[1:]==(12,), 'Summary axes')
    cells={}
    paired=np.any(dirty,axis=1)&np.any(~dirty,axis=1)
    differences=np.full(len(z),np.nan)
    for q in np.flatnonzero(paired):
        differences[q]=theta[q,dirty[q]].mean()-theta[q,~dirty[q]].mean()
    for group in (-1,0,1):
        for d in (0,1):
            mask=(z[:,None]==group)&(dirty==bool(d))
            q_present=np.any(mask,axis=1)
            per_q=[float(theta[q,mask[q]].mean()) for q in np.flatnonzero(q_present)]
            cells[f'z{group}_d{d}']={
                'questions':int(q_present.sum()), 'question_edge_cells':int(mask.sum()),
                'question_equal_weight_mean':None if not per_q else float(np.mean(per_q)),
                'cell_weight_mean_descriptive':None if not mask.any() else float(theta[mask].mean())}
    a=differences[(z==0)&paired]
    b=differences[(z==1)&paired]
    estimate=None if not len(a) or not len(b) else float(b.mean()-a.mean())
    ci=None; boot_var=None
    if estimate is not None:
        rng=np.random.Generator(np.random.PCG64(seed))
        draws=np.empty(2000)
        for i in range(2000):
            draws[i]=b[rng.integers(len(b),size=len(b))].mean()-a[rng.integers(len(a),size=len(a))].mean()
        ci=np.quantile(draws,[.025,.975],method='linear').tolist()
        boot_var=float(np.var(draws))
    guard=len(a)>=20 and len(b)>=20 and boot_var is not None and boot_var>0 and ci[1]>ci[0]
    if not guard: status='INSUFFICIENT_SUPPORT_FOR_FIXED_LITERAL_PROBE'
    elif ci[1]<=0: status='EXPLORATORY_EVIDENCE_AGAINST_POSITIVE_INTERACTION'
    elif ci[0]>0: status='EXPLORATORY_SIGNAL_ONLY'
    else: status='INCONCLUSIVE'
    return {'question_count':len(z),'z_counts':{str(i):int((z==i).sum()) for i in (-1,0,1)},
        'questions_with_both_receiver_change_classes':int(paired.sum()),
        'primary_support_z0':len(a),'primary_support_z1':len(b),
        'primary_interaction':estimate,'ci95_descriptive':ci,'bootstrap_variance':boot_var,
        'inferential_support_guard':bool(guard),'status':status,'cells':cells,
        'theta_negative_cells':int((theta<0).sum()),'theta_zero_cells':int((theta==0).sum()),
        'theta_positive_cells':int((theta>0).sum()),'bootstrap_seed':seed,'bootstrap_repetitions':2000}

def inspect(d2, cache_path, response_result, format_path, format_receipt):
    paths={'d2':d2,'answer_cache':cache_path,'response_result':response_result,
           'format_cache':format_path,'format_receipt':format_receipt}
    pins={k:file_sha(p) for k,p in paths.items()}
    require(pins['d2']==D2_SHA,'D2 source pin')
    require(pins['answer_cache']==ANSWER_CACHE_SHA and pins['response_result']==RESPONSE_RESULT_SHA,
            'Response cache differs from prior independent audited source')
    require(pins['format_cache']==FORMAT_CACHE_SHA,'Format cache differs from prior scan')
    rr=json.loads(Path(response_result).read_bytes())
    fr=json.loads(Path(format_receipt).read_bytes())
    require(rr['cache_sha256']==pins['answer_cache'],'Response cache provenance')
    require(fr['cache_sha256']==pins['format_cache'],'Format cache provenance')
    with np.load(d2,allow_pickle=False) as src:
        partition=src['partition'];rowq=src['row_question']
        require(len(partition)==400,'Expected historical partition metadata')
        selected_q=np.flatnonzero(np.isin(partition,('train','tune')))
        require(len(selected_q)==280 and (partition[selected_q]=='train').sum()==200 and (partition[selected_q]=='tune').sum()==80,'Partition allowlist')
        selected_rows=np.flatnonzero(np.isin(rowq,selected_q))
        masks=src['masks'][selected_rows]
        lg=src['left'].reshape(400,2,12)[selected_q]
        rg=src['right'].reshape(400,2,12)[selected_q]
    require(np.array_equal(rowq[lg],np.broadcast_to(selected_q[:,None,None],lg.shape)) and np.array_equal(rowq[rg],rowq[lg]),'Pair question mapping')
    lookup=np.full(len(rowq),-1,dtype=int);lookup[selected_rows]=np.arange(len(selected_rows))
    left,right=lookup[lg],lookup[rg]
    require(np.all(left>=0) and np.all(right>=0),'Excluded question reached pairs')
    require(np.all((masks[left]!=masks[right]).sum(axis=-1)==1),'Non-single-edge intervention')
    for e in range(12):
        require(np.all(masks[left[:,:,e],e]==0) and np.all(masks[right[:,:,e],e]==1),'Signed edge orientation')
    anchors=reconstruct_anchors(masks,left)
    dirty=receiver_change(anchors)
    # Read only selected source rows; never request question_ids or text/gold fields.
    quality=selected_npy_rows(d2,'quality_by_seed',selected_rows).astype(float)
    require(quality.shape==(len(selected_rows),5) and np.isin(quality,(0.,1.)).all(),'Quality selected scale')
    with np.load(cache_path,allow_pickle=False) as src:
        present=src['present']
        require(present.shape==(len(rowq),5) and present[selected_rows].all() and not present[lookup<0].any(),'Historical cache coverage')
    with np.load(format_path,allow_pickle=False) as src:
        require(np.array_equal(src['present'],present),'Format and hash cache coverage')
    answer=selected_npy_rows(cache_path,'answer',selected_rows)[:,:,:4]
    prompt=selected_npy_rows(cache_path,'prompt',selected_rows)[:,:,:4]
    bad=selected_npy_rows(format_path,'bad',selected_rows)&15
    z=np.empty(280,dtype=np.int8);invariance=0
    for i,q in enumerate(selected_q):
        ix=np.flatnonzero(rowq[selected_rows]==q);first=ix[0]
        require(np.all(answer[ix]==answer[first]),'r1 answers depend on graph')
        require(np.all(prompt[ix]==prompt[first]),'r1 prompt depends on graph')
        require(np.all(bad[ix]==bad[first]),'r1 format metadata depends on graph')
        require(np.all(np.char.str_len(answer[ix])==64),'Missing answer digest')
        z[i]=classify(answer[first,0],bad[first,0])
        invariance+=len(ix)*5*4
    credit=quality[right]-quality[left]
    drift=(credit[:,1,:,1:]-credit[:,0,:,1:])
    theta=theta_crossseed(drift)
    reports={}
    for part,seed in [('train',97531),('tune',97532)]:
        use=partition[selected_q]==part
        reports[part]=summarize(z[use],dirty[use],theta[use],seed)
    for key,path in paths.items():require(file_sha(path)==pins[key],'Input changed: '+key)
    return {'status':reports['tune']['status'],'primary_partition':'tune','reports':reports,
        'input_sha256':pins,'selected_questions':280,'excluded_question_quality_decoded':0,
        'selected_graph_rows':len(selected_rows),'r1_invariance_comparisons':invariance,
        'z_seed':2701,'target_seeds':[2702,2703,2704,2705],
        'graph_transition':'qid-dependent original G2 anchor rule',
        'theta_estimand':'cross-seed squared mean credit drift; may be negative',
        'disagreement':'literal identity only; empty/format_bad -> unknown',
        'no_raw_or_gold_or_current_hotpot_access':True,'gpu_hours':0,'formal_candidates_started':0,
        'claim_limit':'Exploratory historical surrogate interaction, not action utility, causal mediation or active acquisition benefit.'}

def main():
    ap=argparse.ArgumentParser()
    for flag in ('d2','answer-cache','response-result','format-cache','format-receipt','output'):
        ap.add_argument('--'+flag,type=Path,required=True)
    args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    protocol=Path(__file__).with_name('PROTOCOL.md')
    start={'source_sha256':file_sha(__file__),'protocol_sha256':file_sha(protocol),'started_unix':time.time(),
           'raw_paths_read':False,'current_hotpot_read':False}
    emit(out/'STARTED.json',start);tic=time.monotonic()
    try:
        result=inspect(args.d2,args.answer_cache,args.response_result,args.format_cache,args.format_receipt)
        result['wall_seconds']=time.monotonic()-tic
        require(result['wall_seconds']<=120,'CPU wall budget exceeded; result not admitted')
        result['provenance']=start
        emit(out/'RESULTS.json',result)
        emit(out/'COMPLETE.json',{'status':result['status'],'results_sha256':file_sha(out/'RESULTS.json'),
                                 'protocol_sha256':start['protocol_sha256'],'source_sha256':start['source_sha256']})
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds'],
                          'train':result['reports']['train'],'tune':result['reports']['tune']},ensure_ascii=False))
    except Exception as exc:
        emit(out/'FAILED.json',{'status':'BLOCKED_DATA_OR_IMPLEMENTATION','error':type(exc).__name__,
                               'message':str(exc),'wall_seconds':time.monotonic()-tic,'provenance':start})
        raise

if __name__=='__main__':main()
