"""Bounded Qwen terminal-only diagnostic executor; no labels or scoring."""
import argparse,dataclasses,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())

def verify(plan):
    assert plan['mode'] in ['qualification','development']
    assert len(plan['gpus']) in [1,2] and len(set(plan['gpus']))==len(plan['gpus'])
    assert all(type(g) is int and 0<=g<8 for g in plan['gpus'])
    assert 60<plan['worker_cap_seconds']
    if plan['mode']=='qualification':assert len(plan['gpus'])==1 and plan['worker_cap_seconds']<=600 and plan['expected_requests']==8
    else:assert len(plan['gpus'])*plan['worker_cap_seconds']<=6300 and plan['expected_requests']==669
    assert plan['minimum_gpu_free_MiB']>=20000 and plan['minimum_disk_bytes']>=10*1024**3
    assert set(plan['assignments'])=={str(i) for i in range(len(plan['gpus']))}
    assert sorted(v for items in plan['assignments'].values() for v in items)==list(range(plan['expected_requests']))
    for path,h in plan['source_pins'].items():assert sha(path)==h,'Source pin mismatch'
    assert sha(plan['requests'])==plan['requests_sha256']
    jobs=[json.loads(line) for line in Path(plan['requests']).read_bytes().splitlines()]
    assert len(jobs)==plan['expected_requests'] and len({j['job_id'] for j in jobs})==len(jobs)
    for job in jobs:
        req=job['request'];assert digest(req)==job['job_id']
        assert req['model']=='qwen' and req['node']=='M' and req['model_identity_sha256']==plan['model_identity_sha256']
        assert req['config']==plan['config'] and req['max_new_tokens']==256
        assert type(req['seed']) is int and req['seed']>=0
    return jobs

def worker(plan,wid,out):
    jobs=verify(plan);start=time.monotonic();deadline=start+plan['worker_cap_seconds']-30
    out.mkdir();write(out/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    for path in reversed(plan['import_paths']):sys.path.insert(0,path)
    from graph_credit.backend import HFBackend
    from durable_backend import DurableBackend
    from exogenous_path_operator import PathConfig,validate_request_capacity
    active={}
    def guard(context=None):
        if time.monotonic()>deadline:raise TimeoutError('Worker deadline guard')
        if shutil.disk_usage(out).free<plan['minimum_disk_bytes']:raise OSError('Disk guard')
    guard();t=time.monotonic();native=HFBackend(plan['model_path'],'cuda:0',physical_log=None)
    assert digest(native.identity)==plan['model_identity_sha256'],'Pinned model identity mismatch'
    write(out/'LOAD.json',dict(seconds=time.monotonic()-t,identity=native.identity))
    backend=DurableBackend(native,'qwen',out/'REQUEST_JOURNAL.jsonl',context_provider=lambda:active,before_request=guard)
    for index in plan['assignments'][str(wid)]:
        guard();job=jobs[index];req=job['request'];config=PathConfig(**req['config'])
        capacity=validate_request_capacity(backend,req['messages'],config,req['max_new_tokens'])
        active=dict(job_id=job['job_id'],task=job.get('task','artificial'),index=job['index'],execution_seed=job['seed'],seed=req['seed'],question_fingerprint=req['question_fingerprint'],request_fingerprint=job['job_id'],model='qwen',node='M',max_new_tokens=256)
        before=len(native.physical_calls);generation=backend.generate(req['messages'],req['seed'],256,config)
        raw=dataclasses.asdict(generation);events=native.physical_calls[before:]
        assert len(events)==1 and events[0]['status']=='SUCCESS' and events[0]['call_id']==raw['call_id']
        assert raw['input_tokens']==capacity['input_tokens'] and raw['output_tokens']<=256
        append(out/'RESULTS.jsonl',dict(job_id=job['job_id'],generation=raw,capacity=capacity))
    write(out/'COMPLETE.json',dict(status='COMPLETE_UNSCORED',requests=len(plan['assignments'][str(wid)]),elapsed_seconds=time.monotonic()-start))

def supervisor(plan_path,pinned):
    assert sha(plan_path)==pinned;plan=read(plan_path);verify(plan);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpu={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.strip().splitlines()}
        write(out/'RESOURCE.json',dict(gpu_snapshot=raw,unix=time.time(),disk_free=shutil.disk_usage(out).free))
        assert all(gpu[g]>=plan['minimum_gpu_free_MiB'] for g in plan['gpus']),'GPU admission failed'
        assert shutil.disk_usage(out).free>=plan['minimum_disk_bytes'],'Disk admission failed'
    except BaseException as error:
        write(out/'CLOSURE.json',dict(status='PRE_LAUNCH_RESOURCE_REJECTED',all_workers_exited=True,residence_seconds_by_worker=[],actual_GPU_hours=0,quality_seen=False,error_type=type(error).__name__,reason=str(error),plan_sha256=pinned))
        write(out/'ARCHIVE_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});raise
    children=[];starts=[];ends={};logs=[];stopping=False;launch_error=None
    try:
        for wid,gpu in enumerate(plan['gpus']):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(out/f'worker_{wid}.log').open('x');logs.append(log);starts.append(time.monotonic())
            child=subprocess.Popen([sys.executable,__file__,'--plan',str(plan_path),'--plan-sha',pinned,'--worker',str(wid)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
        write(out/'LAUNCH.json',dict(pids=[p.pid for p in children],gpu_indices=plan['gpus'],unix=time.time(),plan_sha256=pinned))
    except BaseException as error:stopping=True;launch_error=type(error).__name__
    while len(ends)<len(children):
        for i,child in enumerate(children):
            if i in ends:continue
            if child.poll() is not None:ends[i]=time.monotonic();continue
            if stopping or time.monotonic()-starts[i]>plan['worker_cap_seconds']:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                ends[i]=time.monotonic()
        if any(c.poll() not in (None,0) for c in children):stopping=True
        time.sleep(1)
    for log in logs:log.close()
    elapsed=[ends[i]-starts[i] for i in range(len(children))]
    complete=launch_error is None and len(children)==len(plan['gpus']) and all(c.returncode==0 and (out/f'worker_{i}/COMPLETE.json').exists() for i,c in enumerate(children))
    write(out/'CLOSURE.json',dict(status='COMPLETE_UNSCORED' if complete else 'PARTIAL_OR_FAILED',all_workers_exited=True,exit_codes=[c.returncode for c in children],residence_seconds_by_worker=elapsed,actual_GPU_hours=sum(elapsed)/3600,cap_imputed_as_cost=False,quality_seen=False,launch_error=launch_error,plan_sha256=pinned,source_sha256=sha(__file__)))
    write(out/'ARCHIVE_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--plan-sha',required=True);ap.add_argument('--worker',type=int);a=ap.parse_args();assert sha(a.plan)==a.plan_sha
    if a.worker is None:supervisor(a.plan,a.plan_sha)
    else:
        plan=read(a.plan);worker(plan,a.worker,Path(plan['output'])/f'worker_{a.worker}')
if __name__=='__main__':main()
