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
    assert plan['mode'] in ('qualification','formal')
    assert len(set(plan['gpus']))==len(plan['gpus']) and all(type(g) is int and 0<=g<=7 for g in plan['gpus'])
    assert plan['seeds']==[261117101,261117102,261117103]
    assert plan['minimum_gpu_free_MiB']>=24576 and plan['minimum_disk_bytes']>=8*1024**3
    if plan['mode']=='qualification':
        assert len(plan['gpus'])==1 and plan['worker_cap_seconds']<=1740
        assert sha(plan['qualification_inputs'])==plan['qualification_inputs_sha256']
    else:
        count=len(plan['gpus']);assert count in (1,2,4)
        assert plan['prefix_per_worker']==2 and 60<plan['worker_cap_seconds']<=plan['aggregate_reservation_GPU_hours']*3600/count-30
        assert plan['aggregate_reservation_GPU_hours']<=7.5
        assert set(plan['assignments'])=={str(i) for i in range(count)}
        assert all(len(v)==16//count for v in plan['assignments'].values())
        assert sorted(i for v in plan['assignments'].values() for i in v)==list(range(16))

def worker(plan,wid,out):
    verify(plan);start=time.monotonic();limit=plan['worker_cap_seconds'];deadline=start+limit-60
    out.mkdir();write(out/'STARTED.json',{'pid':os.getpid(),'monotonic':start,'unix':time.time()})
    for d in reversed(plan['import_paths']):sys.path.insert(0,d)
    from function_backend import HFBackend
    from durable_backend import DurableBackend
    from function_operator import CodeOperator as PathOperator
    from exogenous_path_operator import PathExecutionError,digest
    ref=[None]
    def guard(context=None):
        if time.monotonic()>deadline:raise TimeoutError('Worker deadline admission guard')
        if shutil.disk_usage(out).free<plan['minimum_disk_bytes']:raise OSError('Disk admission guard')
        if context is not None:
            import torch
            for model_name,wrapped in sorted(backends.items(),key=lambda item:item[0]==context['model']):
                model=wrapped.underlying.model
                target='cuda:0' if model_name==context['model'] else 'cpu'
                current=str(next(model.parameters()).device)
                if current!=target:
                    begin=time.monotonic();model.to(target);torch.cuda.synchronize()
                    with (out/'PLACEMENT.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(request_id=context['request_id'],model=model_name,source=current,target=target,seconds=time.monotonic()-begin))+'\n');f.flush();os.fsync(f.fileno())
            torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info()
            if free<6*1024**3:raise RuntimeError('Shared GPU free-memory guard')
    backends={}
    for name in ['qwen','mistral']:
        guard();t=time.monotonic()
        native=HFBackend(plan['models'][name]['path'],'cuda:0',physical_log=None)
        if plan['mode']=='formal' and digest(native.identity)!=digest(plan['models'][name]['identity']):raise ValueError('Qualified model identity differs')
        if 'expected_model_files' in plan['models'][name] and native.identity['model_files']!=plan['models'][name]['expected_model_files']:raise ValueError('Local/server weight or tokenizer identity differs')
        backends[name]=DurableBackend(native,name,out/(name+'_requests.jsonl'),
            context_provider=lambda:{} if ref[0] is None else ref[0].active_request)
        write(out/('LOAD_'+name+'.json'),{'seconds':time.monotonic()-t,'identity':native.identity})
        native.model.to('cpu')
        import torch
        torch.cuda.empty_cache()
    op=PathOperator(backends['qwen'],backends['mistral'],str(out/'sandbox_cases'),before_request=guard);ref[0]=op
    if plan['mode']=='formal' and op.protocol_hash!=plan['qualified_operator_protocol_sha256']:raise ValueError('Operator differs from accepted qualification')
    write(out/'OPERATOR_IDENTITY.json',{'protocol':op.protocol,'protocol_sha256':op.protocol_hash})
    if plan['mode']=='qualification':
        units=read(plan['qualification_inputs']);assert len(units)==3
    else:
        if sha(plan['questions'])!=plan['questions_sha256'] or sha(plan['selection'])!=plan['selection_sha256']:raise ValueError('Input selection differs')
        rows=[json.loads(x) for x in Path(plan['questions']).read_bytes().splitlines()];manifest=read(plan['selection'])
        if len(rows)!=16 or len(manifest['questions'])!=16:raise ValueError('Fixed coverage differs')
        units=[]
        for i in plan['assignments'][str(wid)]:
            row=rows[i];m=manifest['questions'][i]
            raw=(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
            if hashlib.sha256(raw).hexdigest()!=m['input_sha256']:raise ValueError('Unit pin differs')
            units.append({'index':i,'payload':{k:row[k] for k in ['id','question','function_name','visible_cases']},'shock':m['removed_upstream']})
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
    if plan['mode']=='qualification':
        first=read(out/'unit_000/MATRIX.json');replay=read(out/'unit_002/MATRIX.json')
        assert first['protocol_sha256']==replay['protocol_sha256']
        checks=0
        for a,b in zip(first['conditions'],replay['conditions']):
            assert a['key']==b['key'] and a['answer']==b['answer']
            assert len(a['programs'])==len(b['programs'])
            for pa,pb in zip(a['programs'],b['programs']):
                assert pa['program_sha256']==pb['program_sha256'] and pa['visible']['passed']==pb['visible']['passed']
                for ca,cb in zip(pa['visible']['cases'],pb['visible']['cases']):
                    for k in ['status','passed','returned','typed_return']:
                        assert ca.get(k)==cb.get(k);checks+=1
            for n,rec in a['nodes'].items():
                other=b['nodes'][n]
                assert rec['request_fingerprint']==other['request_fingerprint']
                for key in ['text','input_tokens','output_tokens','finish_reason']:
                    assert rec['generation'][key]==other['generation'][key];checks+=1
        write(out/'REPLAY.json',dict(status='PASS_EXACT_SEEDED_REPLAY',scalar_checks=checks))
    write(out/'COMPLETE.json',{'status':'COLLECTION_COMPLETE_UNSCORED','completed_units':len(units),'elapsed_seconds':time.monotonic()-start,'hidden_calls_scored':0,'visible_execution_feedback_available':True})
def supervisor(plan_path,expected):
    if sha(plan_path)!=expected:raise ValueError('Plan pin differs')
    plan=read(plan_path);verify(plan)
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    # Check physical resources again immediately before launching our children.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu={int(l.split(',')[0]):int(l.split(',')[1]) for l in raw.strip().splitlines()}
    write(out/'RESOURCE.json',{'gpu_snapshot':raw,'unix':time.time(),'disk_free':shutil.disk_usage(out).free})
    if any(gpu[g]<plan['minimum_gpu_free_MiB'] for g in plan['gpus']) or shutil.disk_usage(out).free<plan['minimum_disk_bytes']:
        write(out/'CLOSURE.json',dict(status='RESOURCE_NOT_ADMITTED',all_workers_exited=True,residence_seconds_by_worker=[],actual_GPU_hours=0,cap_imputed_as_cost=False,quality_seen=False,worker_spawned=False))
        write(out/'ARCHIVE_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        return
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
