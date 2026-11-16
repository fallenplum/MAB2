"""Bounded private supervisor/worker for the frozen path operator. No scoring."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def verify(plan):
    for p,h in plan['source_pins'].items():
        if sha(p)!=h:raise ValueError('Pinned source mismatch')
    if plan['mode'] not in ['qualification','formal']:raise ValueError('Unknown mode')
    if plan['mode']=='qualification' and (len(plan['gpus'])!=1 or plan['worker_cap_seconds']>840):raise ValueError('Qualification residency guard')
    if plan['mode']=='formal':
        count=len(plan['gpus'])
        if count not in (1,2,4) or len(set(plan['gpus']))!=count or any(type(g) is not int or not 0<=g<=7 for g in plan['gpus']):raise ValueError('One, two or four distinct authorized GPUs required')
        if plan['prefix_per_worker']!=2 or plan['worker_cap_seconds']>27900/count or plan['worker_cap_seconds']<=60:raise ValueError('Frozen prefix or aggregate 7.75h residency cap differs')
        if set(plan['assignments'])!={str(i) for i in range(count)} or any(len(v)!=96//count for v in plan['assignments'].values()):raise ValueError('Frozen worker coverage differs')
        if sorted(i for v in plan['assignments'].values() for i in v)!=list(range(96)):raise ValueError('Duplicate or missing formal units')
        if plan['seeds']!=[26111671,26111672]:raise ValueError('Frozen seeds differ')
def synthetic(long=False):
    sentences=[[f'Fictional record {i}: the research token for object Zeta is Cedar.'] for i in range(10)]
    if long:
        for i in range(10):sentences[i]+=['Synthetic context filler for engineering capacity only. '*12]*4
    return {'id':'artificial_long_v2_12' if long else 'artificial_short','question':'What is the research token for fictional object Zeta?',
            'context':{'title':[f'Artificial record {i}' for i in range(10)],'sentences':sentences}}
def worker(plan,wid,out):
    verify(plan);start=time.monotonic();limit=plan['worker_cap_seconds'];deadline=start+limit-60
    out.mkdir();write(out/'STARTED.json',{'pid':os.getpid(),'monotonic':start,'unix':time.time()})
    for d in reversed(plan['import_paths']):sys.path.insert(0,d)
    from graph_credit.backend import HFBackend
    from durable_backend import DurableBackend
    from annotation_operator import AnnotationOperator as PathOperator
    from exogenous_path_operator import PathExecutionError,digest
    ref=[None]
    def guard(context=None):
        if time.monotonic()>deadline:raise TimeoutError('Worker deadline admission guard')
        if shutil.disk_usage(out).free<plan['minimum_disk_bytes']:raise OSError('Disk admission guard')
        if context is not None:
            import torch
            active=context['model'];inactive='mistral' if active=='qwen' else 'qwen'
            # Move inactive model first, then release unused allocations before restoration.
            for name,target in [(inactive,'cpu'),(active,'cuda:0')]:
                model=backends[name].underlying.model;source=str(next(model.parameters()).device)
                if target=='cuda:0':torch.cuda.empty_cache()
                if source!=target:
                    moved=time.monotonic();model.to(target);torch.cuda.synchronize();torch.cuda.empty_cache()
                    with (out/'PLACEMENT.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(request_id=context['request_id'],model=name,source=source,target=target,seconds=time.monotonic()-moved))+'\n');f.flush();os.fsync(f.fileno())
    backends={}
    for name in ['qwen','mistral']:
        guard();t=time.monotonic()
        native=HFBackend(plan['models'][name]['path'],'cuda:0',physical_log=None)
        if digest(native.identity)!=digest(plan['models'][name]['identity']):raise ValueError('Qualified model identity differs')
        backends[name]=DurableBackend(native,name,out/(name+'_requests.jsonl'),
            context_provider=lambda:{} if ref[0] is None else ref[0].active_request)
        write(out/('LOAD_'+name+'.json'),{'seconds':time.monotonic()-t,'identity':native.identity})
    op=PathOperator(backends['qwen'],backends['mistral'],before_request=guard);ref[0]=op
    if plan['mode']=='formal' and op.protocol_hash!=plan['qualified_operator_protocol_sha256']:raise ValueError('Operator differs from accepted qualification')
    write(out/'OPERATOR_IDENTITY.json',{'protocol':op.protocol,'protocol_sha256':op.protocol_hash})
    if plan['mode']=='qualification':
        units=[{'index':0,'payload':synthetic(False),'shock':'U'},{'index':1,'payload':synthetic(True),'shock':'V'}]
    else:
        if sha(plan['questions'])!=plan['questions_sha256'] or sha(plan['selection'])!=plan['selection_sha256']:raise ValueError('Input selection differs')
        rows=[json.loads(x) for x in Path(plan['questions']).read_bytes().splitlines()];manifest=read(plan['selection'])
        if len(rows)!=96 or len(manifest['questions'])!=96:raise ValueError('Fixed coverage differs')
        units=[]
        for i in plan['assignments'][str(wid)]:
            row=rows[i];m=manifest['questions'][i]
            raw=(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
            if hashlib.sha256(raw).hexdigest()!=m['input_sha256']:raise ValueError('Unit pin differs')
            units.append({'index':i,'payload':{k:row[k] for k in ['id','question','context']},'shock':m['removed_upstream']})
    for j,u in enumerate(units):
        guard();unit=out/f"unit_{u['index']:03d}";unit.mkdir();write(unit/'INPUT.json',u)
        t=time.monotonic()
        try:r=op.run_matrix(u['payload'],plan['seeds'],u['shock'])
        except PathExecutionError as e:write(unit/'PARTIAL.json',e.partial_result);raise
        write(unit/'MATRIX.json',r)
        write(unit/'COMPLETE.json',{'index':u['index'],'seconds':time.monotonic()-t,'matrix_sha256':sha(unit/'MATRIX.json'),
            'terminal_conditions':r['complete_terminal_conditions'],'physical_requests':len(r['request_trace'])})
        if plan['mode']=='formal' and j+1==plan['prefix_per_worker']:
            write(out/'PREFIX_COMPLETE.json',{'elapsed_seconds':time.monotonic()-start,'completed_units':j+1})
            while not (out.parent/'RELEASE.json').exists():guard();time.sleep(1)
    write(out/'COMPLETE.json',{'status':'COLLECTION_COMPLETE_UNSCORED','completed_units':len(units),'elapsed_seconds':time.monotonic()-start,'model_calls_scored':0})
def supervisor(plan_path,expected):
    if sha(plan_path)!=expected:raise ValueError('Plan pin differs')
    plan=read(plan_path);verify(plan)
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    # Check physical resources again immediately before launching our children.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu={int(l.split(',')[0]):int(l.split(',')[1]) for l in raw.strip().splitlines()}
    write(out/'RESOURCE.json',{'gpu_snapshot':raw,'unix':time.time(),'disk_free':shutil.disk_usage(out).free})
    if any(gpu[g]<plan['minimum_gpu_free_MiB'] for g in plan['gpus']):raise RuntimeError('GPU memory admission failed before launch')
    if shutil.disk_usage(out).free<plan['minimum_disk_bytes']:raise RuntimeError('Disk admission failed before launch')
    children=[];starts=[];ends={};logs=[];release=False;stopping=False
    try:
        for wid,g in enumerate(plan['gpus']):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(out/f'worker_{wid}.log').open('x');logs.append(log);starts.append(time.monotonic())
            p=subprocess.Popen([sys.executable,__file__,'--worker',str(wid),'--plan',str(plan_path),'--plan-sha',expected],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
    except BaseException:
        for i,p in enumerate(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            ends[i]=time.monotonic()
        for log in logs:log.close()
        seconds=[ends[i]-starts[i] for i in range(len(children))]
        write(out/'CLOSURE.json',{'status':'PARTIAL_LAUNCH_FAILURE','all_workers_exited':True,'residence_seconds_by_worker':seconds,'actual_GPU_hours':sum(seconds)/3600,'cap_imputed_as_cost':False,'quality_seen':False})
        write(out/'ARCHIVE_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        raise
    write(out/'LAUNCH.json',{'pids':[p.pid for p in children],'gpu_indices':plan['gpus'],'unix':time.time(),'plan_sha256':expected})
    while len(ends)<len(children):
        for i,p in enumerate(children):
            if i in ends:continue
            if p.poll() is not None:ends[i]=time.monotonic();continue
            if time.monotonic()-starts[i]>plan['worker_cap_seconds'] or stopping:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                ends[i]=time.monotonic()
        if any(p.poll() not in (None,0) for p in children):stopping=True
        if plan['mode']=='formal' and not release and not stopping:
            markers=[out/f'worker_{i}/PREFIX_COMPLETE.json' for i in range(len(children))]
            if all(p.exists() for p in markers):
                # Entire elapsed prefix includes loading and wait; conservative no-quality projection.
                forecasts=[1.5*read(p)['elapsed_seconds']*len(plan['assignments'][str(i)])/plan['prefix_per_worker'] for i,p in enumerate(markers)]
                if max(forecasts)>plan['worker_cap_seconds']-60:
                    write(out/'PREFIX_STOP.json',{'projected_seconds_by_worker':forecasts,'quality_seen':False});stopping=True
                else:write(out/'RELEASE.json',{'projected_seconds_by_worker':forecasts,'quality_seen':False});release=True
        time.sleep(1)
    for log in logs:log.close()
    seconds=[ends[i]-starts[i] for i in range(len(children))]
    complete=all(p.returncode==0 and (out/f'worker_{i}/COMPLETE.json').exists() for i,p in enumerate(children))
    write(out/'CLOSURE.json',{'status':'COMPLETE_UNSCORED' if complete else 'PARTIAL_OR_FAILED','exit_codes':[p.returncode for p in children],
        'residence_seconds_by_worker':seconds,'actual_GPU_hours':sum(seconds)/3600,'cap_imputed_as_cost':False,'all_workers_exited':True,
        'plan_sha256':expected,'source_sha256':sha(__file__),'quality_seen':False})
    write(out/'ARCHIVE_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha',required=True);a.add_argument('--worker',type=int);args=a.parse_args()
    if args.worker is None:supervisor(args.plan,args.plan_sha)
    else:
        if sha(args.plan)!=args.plan_sha:raise ValueError('Plan pin differs')
        p=read(args.plan);worker(p,args.worker,Path(p['output'])/f'worker_{args.worker}')
if __name__=='__main__':main()
