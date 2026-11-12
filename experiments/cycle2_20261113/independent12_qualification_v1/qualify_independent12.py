"""Prepared 52-call synthetic-only qualification; no launch without private admission.

The supervisor derives from qualified v2. Model imports occur only in child().
"""
from __future__ import annotations
import argparse,collections,dataclasses,hashlib,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path
from functools import partial
from gpu_job_guard import guard_before_exec,arm_child_deadline

QUAL_DIR='experiments/cycle2_20261113/independent12_qualification_v1'
CONTROL_DIR='experiments/cycle2_20261113/independent12_control_v2'
ROUTER_DIR='experiments/cycle2_20261113/heterogeneous_qualification_v2'
RESERVATION='C2_INDEPENDENT12_ENGINEERING_QUALIFICATION'
CONTROL_PIN='d3ec41ca2df218592ef8b4dc06c7e26441ee2f0f20daad5b70c6df1ae10b9dbc'
GUARD_PIN='f281ffca929966d923d82f7d61227ce5977731b3cb343c5279145d1af970265f'
SOURCE_NAMES={
 'src/graph_credit/__init__.py','src/graph_credit/executor.py','src/graph_credit/backend.py',
 ROUTER_DIR+'/routed_executor.py',CONTROL_DIR+'/independent12_executor.py',
 QUAL_DIR+'/qualify_independent12.py',QUAL_DIR+'/gpu_job_guard.py',QUAL_DIR+'/PROTOCOL.json',
}


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


def valid_hash(value,length=64):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None and set(value)!={'0'}


def validate_launch_plan(plan):
    # A public template cannot launch. Review, freeze and reservation are external decisions.
    if not __debug__:
        raise RuntimeError('Optimized Python is forbidden: qualification source assertions must remain enabled')
    assert plan['launch_authorized'] is True,'Explicit launch authorization absent'
    assert plan['preparation_state']=='ROOT_REVIEWED_FROZEN_RESERVED_LAUNCH_APPROVED'
    assert valid_hash(plan['root_review_receipt_sha256']),'Independent root review receipt absent'
    assert plan['budget_reservation_confirmed'] is True and valid_hash(plan['reservation_receipt_sha256']),'New reservation receipt absent'
    assert valid_hash(plan['source_commit'],40),'Frozen source commit absent'
    assert plan['qualification_only'] is True and plan['reservation_id']==RESERVATION
    assert plan['max_resident_seconds']==1770 and plan['reserved_gpu_hours']==0.5
    assert plan['minimum_free_mib']==32000 and plan['resource_guard_version']==2
    assert plan['output_artifact_class']=='PRIVATE'
    assert plan['permitted_gpus'] and all(type(x) is int and x>=0 for x in plan['permitted_gpus'])
    assert len(set(plan['permitted_gpus']))==len(plan['permitted_gpus'])
    assert set(plan['source_manifest'])==SOURCE_NAMES and all(valid_hash(x) for x in plan['source_manifest'].values())
    assert plan['source_manifest'][CONTROL_DIR+'/independent12_executor.py']==CONTROL_PIN
    assert plan['source_manifest'][QUAL_DIR+'/gpu_job_guard.py']==GUARD_PIN
    assert set(plan['models'])=={'qwen','mistral'}
    for name,spec in plan['models'].items():
        files=spec['model_files']
        assert {'config.json','generation_config.json','tokenizer.json','tokenizer_config.json'}<=set(files)
        assert any(k.endswith('.safetensors') or k.endswith('.bin') for k in files)
        if name=='mistral':assert 'tokenizer.model' in files
        assert all(type(v['bytes']) is int and v['bytes']>0 and valid_hash(v['sha256']) for v in files.values())
        assert set(spec['allowed_metadata_files'])<={'MAB2_SOURCE_MANIFEST.json'}


def compare_fresh(first,second,protocol):
    """Compare semantic output only; never demand cross-call digest equality."""
    assert first['status']==second['status']=='COMPLETE'
    assert first['question_fingerprint']==second['question_fingerprint']
    assert first['protocol_hash']==second['protocol_hash'] and first['seed']==second['seed']==protocol['seed']
    assert len(first['slots'])==len(second['slots'])==13
    comparisons=0
    for a,b in zip(first['slots'],second['slots']):
        for key in protocol['fresh_equality_fields']['slot']:
            assert a[key]==b[key],f'Fresh slot mismatch: {a["node"]}:{key}'
            comparisons+=1
        for key in protocol['fresh_equality_fields']['record']:
            assert a['record'][key]==b['record'][key],f'Fresh record mismatch: {a["node"]}:{key}'
            comparisons+=1
        for key in protocol['fresh_equality_fields']['generation']:
            assert a['generation'][key]==b['generation'][key],f'Fresh generation mismatch: {a["node"]}:{key}'
            comparisons+=1
    return comparisons


def verify_execution(result,protocol,digest):
    expected=[f'r{sample}:w{role}' for role in range(4) for sample in range(1,4)]+['aggregate']
    assert result['status']=='COMPLETE' and len(result['slots'])==13
    assert [s['node'] for s in result['slots']]==expected
    assert result['cache_reuse_count']==0 and result['physical_cost']==result['logical_service_cost']
    cost=result['physical_cost']
    assert cost['calls']==cost['returned_generations']==13 and cost['unattempted_calls']==0
    assert cost['service_complete'] and cost['cost_fields_complete'] and cost['failed_requests_without_physical_record']==0
    assert cost['resident_GPU_hours'] is None
    config=protocol['graph_config_used_as_sampling_config']
    integrity_keys=('node','input_fingerprint','seed','messages','generation','wire_message','parsed','message_metadata','received_message_tokens','self_state_tokens')
    for index,slot in enumerate(result['slots']):
        node=slot['node'];record=slot['record'];gen=slot['generation']
        assert slot['index']==index and slot['attempted_backend_request'] and record is not None
        assert slot['status'] in ('SUCCESS','FORMAT_ERROR') and slot['error_type'] is None
        assert gen==record['generation'] and slot['messages']==record['messages']
        assert record['output_digest']==digest({key:record[key] for key in integrity_keys})
        assert not record['reused'] and record['source_execution_id'] is None
        assert slot['self_state_tokens']==record['self_state_tokens']==0
        model='mistral' if node=='aggregate' else protocol['model_route']['workers'][int(node[-1])]
        assert slot['model']==model
        cap=config['aggregate_tokens'] if node=='aggregate' else config['message_tokens']
        derived_seed=int(digest([result['question_id'],result['seed'],node])[:16],16)%(2**31-1)
        assert slot['seed']==record['seed']==derived_seed
        fingerprint=digest({'question':result['question_fingerprint'],'node':node,'messages':slot['messages'],
            'seed':derived_seed,'protocol':result['protocol_hash'],'max_new_tokens':cap})
        assert slot['input_fingerprint']==record['input_fingerprint']==fingerprint
        assert gen['input_tokens']<=config['max_input_tokens'] and gen['output_tokens']<=cap
        assert record['message_metadata']['message_tokens']<=cap
        if node!='aggregate':assert slot['received_message_tokens']==record['received_message_tokens']==0
    for role in range(4):
        slots=result['slots'][role*3:role*3+3]
        assert slots[0]['messages']==slots[1]['messages']==slots[2]['messages']
    assert result['answer']==result['slots'][-1]['record']['parsed']['answer']
    for key in ('input_tokens','output_tokens','gpu_seconds'):
        assert math.isclose(cost[key],sum(s['generation'][key] for s in result['slots']),rel_tol=1e-12,abs_tol=1e-9)
    assert math.isclose(cost['model_elapsed_seconds'],sum(s['generation']['elapsed_seconds'] for s in result['slots']),rel_tol=1e-12,abs_tol=1e-9)
    assert cost['peak_allocated_bytes']==max(s['generation']['gpu_peak_allocated_bytes'] for s in result['slots'])
    assert cost['received_message_tokens']==sum(s['record']['received_message_tokens'] for s in result['slots']) and cost['self_state_tokens']==0
    for key in ('input_tokens','output_tokens','gpu_seconds','model_elapsed_seconds','peak_allocated_bytes'):
        assert cost[key]==cost['known_cost_components'][key]


def verify_call_closure(results,router,backends,protocol,digest):
    routes=router.route_events
    actual={name:list(backend.physical_calls) for name,backend in backends.items()}
    physical=[(name,row) for name,rows in actual.items() for row in rows]
    assert len(routes)==len(physical)==protocol['expected_complete_calls']==52
    assert all(row['status']=='SUCCESS' for row in routes)
    assert all(row['status']=='SUCCESS' for _,row in physical)
    assert {name:len(rows) for name,rows in actual.items()}==protocol['expected_complete_calls_by_model']
    route_by_id={row['call_id']:row for row in routes}
    physical_by_id={row['call_id']:(name,row) for name,row in physical}
    assert len(route_by_id)==len(physical_by_id)==52 and set(route_by_id)==set(physical_by_id)
    billed=[];formats=[]
    for result in results:
        for slot in result['slots']:
            gen=slot['generation'];call=gen['call_id'];billed.append(call)
            route=route_by_id[call];model,record=physical_by_id[call]
            assert route['model']==model==slot['model'] and route['node']==slot['node']
            assert route['model_identity_sha256']==digest(backends[model].identity)
            assert route['seed']==record['seed']==slot['seed']
            assert route['prompt_sha256']==record['prompt_sha256']==digest(slot['messages'])
            config=protocol['graph_config_used_as_sampling_config']
            cap=config['aggregate_tokens'] if slot['node']=='aggregate' else config['message_tokens']
            assert route['max_new_tokens']==record['generation_config']['max_new_tokens']==cap
            for key in ('call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason'):
                assert gen[key]==record[key]
            for key in ('call_id','input_tokens','output_tokens'):assert gen[key]==route[key]
            formats.append({'model':model,'node':slot['node'],'format_ok':slot['record']['message_metadata']['format_ok']})
    assert len(billed)==len(set(billed))==52 and set(billed)==set(physical_by_id)
    total={key:sum(row[key] for _,row in physical) for key in ('input_tokens','output_tokens','gpu_seconds')}
    for key in total:
        assert math.isclose(total[key],sum(r['physical_cost'][key] for r in results),rel_tol=1e-12,abs_tol=1e-9)
    total['calls']=52
    total['model_elapsed_seconds']=sum(row['elapsed_seconds'] for _,row in physical)
    assert math.isclose(total['model_elapsed_seconds'],sum(r['physical_cost']['model_elapsed_seconds'] for r in results),rel_tol=1e-12,abs_tol=1e-9)
    total['resident_GPU_hours']=None
    return total,formats


def child(plan,out):
    root=pin_sources(plan)
    sys.path[:0]=[str(root/'src'),str(root/ROUTER_DIR),str(root/CONTROL_DIR)]
    from graph_credit.backend import HFBackend
    from graph_credit.executor import GraphConfig,digest
    from routed_executor import RoutedBackend
    from independent12_executor import Independent12Executor,Independent12ExecutionError
    protocol=json.loads((root/QUAL_DIR/'PROTOCOL.json').read_bytes())
    assert protocol['status']=='ENGINEERING_ONLY_REQUIRES_SEPARATE_ADMISSION'
    assert protocol['control_source_sha256']==CONTROL_PIN and protocol['guard_source_sha256']==GUARD_PIN
    assert protocol['fresh_repeats_per_question']==2 and len(protocol['synthetic_inputs'])==2
    assert protocol['model_route']=={'workers':['qwen','mistral','qwen','mistral'],'aggregate':'mistral'}
    assert protocol['expected_complete_calls']==52 and protocol['expected_complete_calls_by_model']=={'qwen':24,'mistral':28}
    status={'status':'LOADING_MODELS','qualification_only':True,'benchmark_questions_read':0,'formal_candidate_rounds_started':0,
            'source_commit':plan['source_commit'],'source_manifest':plan['source_manifest'],
            'completed_execution_requests':0,'resident_GPU_hours':None,'failure_costs':None}
    write(out/'INDEPENDENT12_STATUS.json',status)
    backends={};router=None;results=[];comparisons=0;active_name=None

    class CheckpointingIndependent12Executor(Independent12Executor):
        checkpoint_path=None
        def _execute_slot(self,payload,slot,result):
            try:
                return super()._execute_slot(payload,slot,result)
            finally:
                # Boundary snapshots retain returned generations even if normalization fails.
                # RUNNING snapshots never serve as completion evidence after timeout or crash.
                boundary_error=sys.exc_info()[0]
                write(self.checkpoint_path,{'snapshot_only':True,'completion_evidence':False,
                    'last_slot_boundary':slot.node,'boundary_error_type':boundary_error.__name__ if boundary_error else None,
                    'checkpoint_unix':time.time(),'result':result.to_dict()})

    try:
        for name in ('qwen','mistral'):
            status['active_model_load']=name;write(out/'INDEPENDENT12_STATUS.json',status)
            spec=plan['models'][name]
            backends[name]=HFBackend(spec['path'],'cuda:0',out/(name+'_physical_calls.jsonl'))
            actual=backends[name].identity['model_files'];expected=spec['model_files']
            assert all(actual[k]==v for k,v in expected.items()),'Unexpected model file'
            assert set(actual)-set(expected)<=set(spec['allowed_metadata_files'])
        router=RoutedBackend(backends,protocol['model_route']['workers'],protocol['model_route']['aggregate'],out/'route_calls.jsonl')
        exe=CheckpointingIndependent12Executor(router,GraphConfig(**protocol['graph_config_used_as_sampling_config']))
        assert exe.protocol_identity['control_id']==protocol['control_id']
        write(out/'INDEPENDENT12_IDENTITY.json',{'backend':router.identity,'control_protocol_hash':exe.protocol_hash,
            'control_protocol_identity':exe.protocol_identity,'qualification_source_commit':plan['source_commit'],
            'qualification_source_manifest':plan['source_manifest'],'runtime':{name:b.runtime for name,b in backends.items()}})
        status.update(status='RUNNING',active_model_load=None)
        for question in protocol['synthetic_inputs']:
            pair=[]
            for repeat in (1,2):
                active_name=question['id']+'_independent12_fresh'+str(repeat)
                status['active_execution']=active_name;write(out/'INDEPENDENT12_STATUS.json',status)
                exe.checkpoint_path=out/(active_name+'.checkpoint.json')
                planned_nodes=[f'r{s}:w{r}' for r in range(4) for s in range(1,4)]+['aggregate']
                write(exe.checkpoint_path,{'snapshot_only':True,'completion_evidence':False,'status':'NOT_STARTED',
                    'planned_slots':[{'node':node,'status':'NOT_ATTEMPTED'} for node in planned_nodes],'costs':None})
                try:
                    result=exe.execute(question,protocol['seed']).to_dict()
                except Independent12ExecutionError as error:
                    write(out/(active_name+'.partial.json'),error.partial_result.to_dict())
                    raise
                # Retain completed execution even if later audit/equality checks reject qualification.
                write(out/(active_name+'.json'),result)
                verify_execution(result,protocol,digest)
                results.append(result);pair.append(result)
                status['completed_execution_requests']=sum(r['physical_cost']['calls'] for r in results)
                write(out/'INDEPENDENT12_STATUS.json',status)
            comparisons+=compare_fresh(pair[0],pair[1],protocol)
        costs,formats=verify_call_closure(results,router,backends,protocol,digest)
        write(out/'QUALIFICATION.json',{'status':'PASS_ENGINEERING_REVIEW_REQUIRED','synthetic_only':True,
            'benchmark_questions_read':0,'formal_candidate_rounds_started':0,'inference_calls':52,
            'calls_by_model':{name:len(b.physical_calls) for name,b in backends.items()},
            'fresh_field_equalities':comparisons,'format_failures':sum(not r['format_ok'] for r in formats),
            'format_by_model':{name:{'n':sum(r['model']==name for r in formats),'bad':sum(r['model']==name and not r['format_ok'] for r in formats)} for name in backends},
            'format_records':formats,'costs':costs,'execution_costs':[r['physical_cost'] for r in results],
            'occupancy_completion_required_from':'RESIDENT_STATUS.json','no_accuracy_or_novelty_claim':True})
        status.update(status='PASS_ENGINEERING_REVIEW_REQUIRED',active_execution=None,completed_execution_requests=52)
        write(out/'INDEPENDENT12_STATUS.json',status)
    except BaseException as error:
        # SIGKILL/SIGALRM may bypass this handler; checkpoints and the parent resident ledger remain authoritative.
        status.update(status='FAILED',error_type=type(error).__name__,active_execution=active_name,
            route_requests_recorded=len(router.route_events) if router is not None else 0,
            physical_events_recorded=sum(len(b.physical_calls) for b in backends.values()),
            resident_GPU_hours=None,failure_costs=None)
        write(out/'INDEPENDENT12_STATUS.json',status)
        write(out/'QUALIFICATION_FAILURE.json',status)
        raise


def parent(plan_path,plan):
    assert os.name=='posix','GPU supervisor requires the authorized Linux server'
    import fcntl
    root=pin_sources(plan)
    out=Path(plan['output']).resolve();project=Path(plan['project_root']).resolve(strict=True)
    assert out.is_relative_to(project) and not out.exists(),'Fresh output required'
    assert plan['max_resident_seconds']==1770 and plan['reserved_gpu_hours']==0.5
    assert plan['qualification_only'] is True and plan['reservation_id']=='C2_INDEPENDENT12_ENGINEERING_QUALIFICATION'
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
    validate_launch_plan(plan)
    if a.child:child(plan,Path(plan['output']));return 0
    return parent(a.plan.resolve(),plan)


if __name__=='__main__':raise SystemExit(main())
