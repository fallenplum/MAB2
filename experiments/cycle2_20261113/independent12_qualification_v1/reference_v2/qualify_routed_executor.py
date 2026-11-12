"""Bounded, synthetic-only engineering qualification; no benchmark admission."""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from functools import partial
from gpu_job_guard import guard_before_exec,arm_child_deadline


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def pin_sources(plan):
    root=Path(plan['code_root']).resolve(strict=True)
    for name,pin in plan['source_manifest'].items():
        p=(root/name).resolve(strict=True)
        assert p.is_relative_to(root) and sha(p)==pin,'Source pin mismatch'
    return root


def child(plan,out):
    root=pin_sources(plan)
    sys.path[:0]=[str(root/'src'),str(root/'experiments/cycle2_20261113/heterogeneous_qualification_v2')]
    from graph_credit.backend import HFBackend
    from graph_credit.executor import GraphConfig,affected_closure
    from routed_executor import RoutedBackend,RoutedExecutor
    protocol=json.loads((root/'experiments/cycle2_20261113/heterogeneous_qualification_v2/PROTOCOL.json').read_bytes())
    assert protocol['status']=='ENGINEERING_ONLY_NO_BENCHMARK' and plan['max_resident_seconds']==1770
    assert protocol['model_route']=={'workers':['qwen','mistral','qwen','mistral'],'aggregate':'mistral'}
    backends={}
    for name in ('qwen','mistral'):
        spec=plan['models'][name]
        backends[name]=HFBackend(spec['path'],'cuda:0',out/(name+'_physical_calls.jsonl'))
        actual=backends[name].identity['model_files']
        expected=spec['model_files']
        assert all(actual[k]==v for k,v in expected.items()),'Unexpected model file'
        assert set(actual)-set(expected)<=set(spec['allowed_metadata_files'])
    router=RoutedBackend(backends,protocol['model_route']['workers'],protocol['model_route']['aggregate'],out/'route_calls.jsonl')
    exe=RoutedExecutor(router,GraphConfig(**protocol['graph']))
    write(out/'IDENTITY.json',{'backend':router.identity,'protocol_hash':exe.protocol_hash,
                              'runtime':{k:b.runtime for k,b in backends.items()}})
    old=tuple(protocol['old_mask']);new=tuple(protocol['new_mask']);checks=0;formats=[];calls=[]
    for q in protocol['synthetic_inputs']:
        base=exe.execute(q,old,protocol['seed'])
        changed=exe.execute(q,new,protocol['seed'],base)
        replayed=exe.execute(q,new,protocol['seed'])
        cache=exe.execute(q,new,protocol['seed'],changed)
        assert all(n.reused for n in cache.nodes.values());checks+=1
        assert {n for n,r in changed.nodes.items() if not r.reused}==affected_closure(old,new,3);checks+=1
        for node in changed.nodes:
            a,b=changed.nodes[node],replayed.nodes[node]
            assert a.messages==b.messages and a.input_fingerprint==b.input_fingerprint
            assert a.generation.text==b.generation.text and a.wire_message==b.wire_message
            assert a.parsed==b.parsed and a.message_metadata==b.message_metadata
            checks+=5
        for tag,result in [('base',base),('changed',changed),('fresh',replayed),('cache',cache)]:
            write(out/(q['id']+'_'+tag+'.json'),result.to_dict())
            for node,r in result.nodes.items():
                if not r.reused:formats.append({'model':router.model_for_node(node),'node':node,'format_ok':r.message_metadata['format_ok']})
        calls.extend([base.cost(),changed.cost(),replayed.cost(),cache.cost()])
    expected=len(protocol['synthetic_inputs'])*(13+len(affected_closure(old,new,3))+13)
    assert len(router.route_events)==expected and all(r['status']=='SUCCESS' for r in router.route_events)
    assert sum(c['calls'] for c in calls)==expected
    write(out/'QUALIFICATION.json',{'status':'PASS_ENGINEERING_REVIEW_REQUIRED','synthetic_only':True,
        'benchmark_questions_read':0,'formal_candidate_rounds_started':0,'inference_calls':expected,
        'replay_checks':checks,'format_failures':sum(not r['format_ok'] for r in formats),
        'format_by_model':{k:{'n':sum(r['model']==k for r in formats),'bad':sum(r['model']==k and not r['format_ok'] for r in formats)} for k in backends},
        'format_records':formats,'costs':calls,'no_accuracy_or_novelty_claim':True})


def parent(plan_path,plan):
    assert os.name=='posix','GPU supervisor requires the authorized Linux server'
    import fcntl
    root=pin_sources(plan)
    out=Path(plan['output']).resolve();project=Path(plan['project_root']).resolve(strict=True)
    assert out.is_relative_to(project) and not out.exists(),'Fresh output required'
    assert plan['max_resident_seconds']==1770 and plan['reserved_gpu_hours']==0.5
    assert plan['qualification_only'] is True and plan['reservation_id']=='C2_HETERO_ENGINEERING_QUALIFICATION'
    # Prefer an idle permitted card; never stop another project.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    rows=[];occupied={x.split(',')[0].strip() for x in apps.splitlines() if x.strip()}
    for line in raw.splitlines():
        index,uuid,free,util=[x.strip() for x in line.split(',')]
        if int(index) in plan['permitted_gpus'] and int(free)>=plan['minimum_free_mib']:
            rows.append({'gpu':int(index),'uuid':uuid,'free_mib':int(free),'utilization':int(util),'other_compute_present':uuid in occupied})
    rows.sort(key=lambda x:(x['other_compute_present'],x['utilization'],-x['free_mib'],x['gpu']))
    if not rows:raise RuntimeError('No qualified GPU available; no model launched')
    out.mkdir(parents=True,exist_ok=False)
    lock=(out/'LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=rows[0];write(out/'RESOURCE_ADMISSION.json',{'selected':selected,'available':rows,'observed_unix':time.time(),'other_jobs_stopped':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(selected['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    started=time.time();start_mono=time.monotonic();proc=None;reason='UNKNOWN';code=None
    status={'status':'STARTING','started_unix':started,'selected_gpu':selected['gpu'],'max_resident_seconds':1770,'plan_sha256':sha(plan_path),'qualification_only':True}
    write(out/'RESIDENT_STATUS.json',status)
    try:
        with (out/'child.log').open('wb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(plan_path),'--child','--supervisor-pid',str(os.getpid()),'--deadline-monotonic',repr(start_mono+1760)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=partial(guard_before_exec,os.getpid(),start_mono+1760))
            status.update(status='RUNNING',child_pid=proc.pid);write(out/'RESIDENT_STATUS.json',status)
            while proc.poll() is None:
                if time.monotonic()-start_mono>=1770:
                    reason='OWN_CHILD_RESIDENT_CAP';os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(2)
            code=proc.wait()
            if reason=='UNKNOWN':reason='CHILD_EXIT'
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        ended=time.time();elapsed=time.monotonic()-start_mono;status.update(status='CLOSED',ended_unix=ended,resident_seconds=elapsed,gpu_hours=elapsed/3600,reason=reason,returncode=code,reservation_exceeded=elapsed>1800)
        write(out/'RESIDENT_STATUS.json',status);lock.close()
    print(json.dumps(status),flush=True)
    return 0 if code==0 and reason=='CHILD_EXIT' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--child',action='store_true');p.add_argument('--supervisor-pid',type=int);p.add_argument('--deadline-monotonic',type=float);a=p.parse_args()
    if a.child:
        if a.supervisor_pid is None or a.deadline_monotonic is None:p.error('Child must be launched by its bounded supervisor')
        arm_child_deadline(a.supervisor_pid,a.deadline_monotonic)
    elif a.supervisor_pid is not None or a.deadline_monotonic is not None:p.error('Child-only arguments')
    plan=json.loads(a.plan.read_bytes())
    if a.child:child(plan,Path(plan['output']));return 0
    return parent(a.plan.resolve(),plan)


if __name__=='__main__':raise SystemExit(main())
