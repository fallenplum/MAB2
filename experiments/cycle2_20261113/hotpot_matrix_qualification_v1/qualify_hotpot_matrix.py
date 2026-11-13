"""Prepared synthetic 174-call full-matrix qualification; explicit external admission required."""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from functools import partial
from gpu_job_guard import guard_before_exec,arm_child_deadline

QUAL_DIR='experiments/cycle2_20261113/hotpot_matrix_qualification_v1'
MATRIX_DIR='experiments/cycle2_20261113/hotpot_matrix_v1'
CONTROL_DIR='experiments/cycle2_20261113/independent12_control_v2'
ROUTER_DIR='experiments/cycle2_20261113/heterogeneous_qualification_v2'
DURABLE_DIR='experiments/cycle2_20261113/durable_backend_v1'
RESERVATION='C2_HOTPOT_MATRIX_ENGINEERING_QUALIFICATION'
FIXED_PINS={
 'src/graph_credit/__init__.py':'b0720c82632ad92e72487a3afadedd418cd44a4ca9ebc00d69c7405829800cdd',
 'src/graph_credit/executor.py':'85acab26b3b713900e8099a5d77197af698525f57c2d9d07d1193c68b2c383b1',
 'src/graph_credit/backend.py':'284cfd5acc8c970b7525e498174466e4662bfe1b35a1235ae230bdf2e7e0e2ed',
 ROUTER_DIR+'/routed_executor.py':'0273aebcfba546fd7112c82937b8f27447d0eee87243fbd9c00137d107938625',
 CONTROL_DIR+'/independent12_executor.py':'d3ec41ca2df218592ef8b4dc06c7e26441ee2f0f20daad5b70c6df1ae10b9dbc',
 MATRIX_DIR+'/matrix_executor.py':'016b175f75fca838c83f3c26073a656b901e3d2fda12133c917073304c67bb8c',
 MATRIX_DIR+'/matrix_geometry.py':'e5bdc9e47d2564fb4338516da15ad183c4a935b6e778de34788fdfc33cb8fc46',
 QUAL_DIR+'/gpu_job_guard.py':'f281ffca929966d923d82f7d61227ce5977731b3cb343c5279145d1af970265f',
 QUAL_DIR+'/SYNTHETIC_INPUT.json':'84de20fa16f107453811248a949359811fb31c34702dbbcce2f560a97c5fdd75',
}
SOURCE_NAMES=set(FIXED_PINS)|{QUAL_DIR+'/'+name for name in
    ('qualify_hotpot_matrix.py','matrix_qualification_audit.py','PROTOCOL.json')}|{DURABLE_DIR+'/durable_backend.py'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with tmp.open('wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
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
    if not __debug__:raise RuntimeError('Optimized Python is forbidden')
    assert plan['launch_authorized'] is True
    assert plan['preparation_state']=='ROOT_REVIEWED_FROZEN_RESERVED_LAUNCH_APPROVED'
    assert valid_hash(plan['root_review_receipt_sha256']) and valid_hash(plan['reservation_receipt_sha256'])
    assert plan['budget_reservation_confirmed'] is True and valid_hash(plan['source_commit'],40)
    assert plan['matrix_source_commit']=='c6a687afa179c826813b11ce3472fe1e8f3166bf'
    assert plan['qualification_only'] is True and plan['reservation_id']==RESERVATION
    assert plan['max_resident_seconds']==1770 and plan['reserved_gpu_hours']==0.5
    assert plan['minimum_free_mib']==32000 and plan['resource_guard_version']==2
    assert plan['output_artifact_class']=='PRIVATE' and plan['maximum_backend_requests']==174
    assert plan['permitted_gpus'] and all(type(x) is int and x>=0 for x in plan['permitted_gpus'])
    assert len(set(plan['permitted_gpus']))==len(plan['permitted_gpus'])
    assert set(plan['source_manifest'])==SOURCE_NAMES
    assert all(valid_hash(x) for x in plan['source_manifest'].values())
    assert all(plan['source_manifest'][name]==value for name,value in FIXED_PINS.items())
    assert valid_hash(plan['expected_matrix_protocol_hash'])
    assert set(plan['models'])=={'qwen','mistral'}
    for name,spec in plan['models'].items():
        files=spec['model_files'];assert valid_hash(spec['expected_identity_sha256'])
        assert {'config.json','generation_config.json','tokenizer.json','tokenizer_config.json'}<=set(files)
        assert any(k.endswith('.safetensors') or k.endswith('.bin') for k in files)
        if name=='mistral':assert 'tokenizer.model' in files
        assert all(type(v['bytes']) is int and v['bytes']>0 and valid_hash(v['sha256']) for v in files.values())
        assert set(spec['allowed_metadata_files'])<={'MAB2_SOURCE_MANIFEST.json'}


class RequestCap:
    def __init__(self,maximum=174):self.maximum=maximum;self.admitted=0
    def __call__(self,context):
        if self.admitted>=self.maximum:raise RuntimeError('Qualification physical request cap reached; no retry')
        self.admitted+=1


def checkpointed_matrix_class(base):
    class CheckpointedMatrix(base):
        def configure_checkpoint(self,callback):self.checkpoint=callback;self.live_conditions=[]
        def _plan(self,*args):
            self.live_conditions=super()._plan(*args);return self.live_conditions
        def _finish_condition(self,condition,*args):
            super()._finish_condition(condition,*args)
            self.checkpoint(self.live_conditions)
            if condition.status=='INPUT_REJECTED':
                raise RuntimeError('Engineering qualification stops on the first noncomplete condition; no retry')
    return CheckpointedMatrix


def child(plan,out):
    root=pin_sources(plan)
    sys.path[:0]=[str(root/'src')]+[str(root/path) for path in
        (ROUTER_DIR,CONTROL_DIR,MATRIX_DIR,DURABLE_DIR,QUAL_DIR)]
    from graph_credit.backend import HFBackend
    from graph_credit.executor import GraphConfig,digest
    from matrix_executor import HotpotApplicabilityMatrixExecutor
    from durable_backend import DurableBackend,extract_durable_error_evidence
    from matrix_qualification_audit import verify_matrix,compare_fresh,verify_global_calls,verify_durable
    protocol=json.loads((root/QUAL_DIR/'PROTOCOL.json').read_bytes())
    question=json.loads((root/QUAL_DIR/'SYNTHETIC_INPUT.json').read_bytes())
    assert protocol['status']=='ENGINEERING_ONLY_NO_BENCHMARK' and protocol['successful_physical_calls']==174
    assert digest(protocol['graph'])==digest(dataclasses.asdict(GraphConfig(rounds=3)))
    assert protocol['synthetic_input_sha256']==sha(root/QUAL_DIR/'SYNTHETIC_INPUT.json')
    raw={};wrapped={};journals={};matrices=[];matrix=None;state={'repeat':None};cap=RequestCap()
    started=time.perf_counter()
    try:
        for name in ('qwen','mistral'):
            spec=plan['models'][name]
            # Native disk I/O is disabled; durable helper captures returned raw text
            # and unmodified in-memory native events together before exposing it.
            backend=HFBackend(spec['path'],'cuda:0',physical_log=None);raw[name]=backend
            actual=backend.identity['model_files'];expected=spec['model_files']
            assert all(actual[k]==v for k,v in expected.items())
            assert set(actual)-set(expected)<=set(spec['allowed_metadata_files'])
            assert digest(backend.identity)==spec['expected_identity_sha256']
            journals[name]=out/(name+'_durable_raw.jsonl')
            wrapped[name]=DurableBackend(backend,name,journals[name],
                context_provider=lambda:{'repeat':state['repeat'],'node':matrix.router._selected.get() if matrix else None},
                before_request=cap)
        matrix=checkpointed_matrix_class(HotpotApplicabilityMatrixExecutor)(wrapped,GraphConfig(rounds=3),route_log=None)
        assert matrix.protocol_hash==plan['expected_matrix_protocol_hash'],'Matrix protocol identity mismatch'
        write(out/'IDENTITY.json',{'source_manifest':plan['source_manifest'],'matrix_protocol_identity':matrix.protocol_identity,
            'matrix_protocol_hash':matrix.protocol_hash,'backend':{k:b.identity for k,b in raw.items()},
            'runtime':{k:b.runtime for k,b in raw.items()},'native_physical_log':None,'durable_raw_enabled':True})
        full=question['question']+'\n\nSupplied context:\n'+question['context']
        lengths={name:len(b.encode(full)) for name,b in raw.items()}
        write(out/'SERVER_INPUT_LENGTHS.json',{'complete_user_text_tokens':lengths,'chat_template_included':False,
            'per_request_receiver_chat_tokens_in_generation':True,'context_truncated':False,
            'synthetic_input_sha256':sha(root/QUAL_DIR/'SYNTHETIC_INPUT.json')})
        assert 3500<=lengths['mistral']<=4000,'Unexpected server synthetic input length; do not truncate'
        for repeat in (1,2):
            state['repeat']=repeat
            matrix.configure_checkpoint(lambda conditions:write(out/f'repeat_{repeat}_partial.json',{
                'status':'PARTIAL_CONDITION_CHECKPOINT','repeat':repeat,'conditions':[dataclasses.asdict(c) for c in conditions],
                'admitted_requests':cap.admitted,'total_cost_complete':False,'resident_GPU_hours':None}))
            result=matrix.execute(question,protocol['seed1'],protocol['seed2'],protocol['probe_edge']).to_dict()
            matrices.append(result);write(out/f'repeat_{repeat}_matrix.json',result)
            verify_matrix(result,question,protocol,digest)
            assert result['matrix_protocol_hash']==plan['expected_matrix_protocol_hash']
        assert cap.admitted==174
        fresh=compare_fresh(*matrices)
        costs,formats=verify_global_calls(matrices,matrix.router,raw,protocol,digest)
        durable=verify_durable(matrices,raw,journals,digest)
        wall=sum(m['elapsed_seconds'] for m in matrices)
        write(out/'QUALIFICATION.json',{'status':'PASS_ENGINEERING_REVIEW_REQUIRED','synthetic_only':True,
            'benchmark_questions_read':0,'formal_candidate_rounds_started':0,'scientific_admission':False,
            'matrix_protocol_hash':matrix.protocol_hash,'inference_calls':174,'logical_service_calls':262,
            'fresh_comparison':fresh,'durable_audit':durable,'physical_cost':costs,
            'logical_costs':[m['total_logical_service_cost'] for m in matrices],
            'format_failures':sum(not r['format_ok'] for r in formats),'format_records':formats,
            'format_is_not_accuracy':True,'matrix_wall_seconds':wall,'child_elapsed_seconds':time.perf_counter()-started,
            'successful_calls_per_matrix_wall_second':174/wall,
            'conditional_384_question_success_path_gpu_hours':384*(wall/2)/3600,
            'projection_scope':'single synthetic length, actual hardware/software, successful cache geometry; not measured benchmark throughput or resource admission',
            'resident_GPU_hours':None,'parent_closed_status_required_for_occupancy':True})
    except BaseException as error:
        evidence=extract_durable_error_evidence(error)
        partial_result=getattr(error,'partial_result',None)
        failure={'status':'INCOMPLETE','error_type':type(error).__name__,'repeat':state['repeat'],
            'admitted_requests':cap.admitted,'maximum_requests':174,'completed_matrices':len(matrices),
            'partial_matrix':partial_result.to_dict() if partial_result is not None else None,
            'live_conditions':[dataclasses.asdict(c) for c in getattr(matrix,'live_conditions',[])],
            'native_physical_events':{name:b.physical_calls for name,b in raw.items()},
            'route_events':matrix.router.route_events if matrix else [],'durable_error_evidence':evidence,
            'unknown_cost_is_zero':False,'incomplete_total_physical_cost':None,'resident_GPU_hours':None,
            'no_retry':True,'scientific_admission':False}
        try:write(out/'FAILURE_PARTIAL.json',failure)
        except Exception as save_error:
            # The supervisor records exit; a persistent disk failure cannot be
            # represented as successful persistence or recovered raw output.
            print(json.dumps({'partial_write_failed':type(save_error).__name__,'original_error':type(error).__name__}),flush=True)
        raise


def parent(plan_path,plan,plan_sha256):
    assert sha(plan_path)==plan_sha256,'Plan changed before parent admission'
    assert os.name=='posix','GPU supervisor requires the authorized Linux server'
    import fcntl
    root=pin_sources(plan)
    out=Path(plan['output']).resolve();project=Path(plan['project_root']).resolve(strict=True)
    assert out.is_relative_to(project) and not out.exists(),'Fresh output required'
    assert plan['max_resident_seconds']==1770 and plan['reserved_gpu_hours']==0.5
    assert plan['qualification_only'] is True and plan['reservation_id']==RESERVATION
    # Prefer an idle permitted card; never stop another project.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    rows=[];occupied={x.split(',')[0].strip() for x in apps.splitlines() if x.strip()}
    for line in raw.splitlines():
        index,uuid,free,util=[x.strip() for x in line.split(',')]
        if int(index) in plan['permitted_gpus'] and int(free)>=plan['minimum_free_mib'] and uuid not in occupied:
            rows.append({'gpu':int(index),'uuid':uuid,'free_mib':int(free),'utilization':int(util),'other_compute_present':uuid in occupied})
    rows.sort(key=lambda x:(x['other_compute_present'],x['utilization'],-x['free_mib'],x['gpu']))
    if not rows:raise RuntimeError('No qualified GPU available; no model launched')
    out.mkdir(parents=True,exist_ok=False)
    lock=(out/'LOCK').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=rows[0];write(out/'RESOURCE_ADMISSION.json',{'selected':selected,'available':rows,'observed_unix':time.time(),'other_jobs_stopped':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(selected['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    started=time.time();start_mono=time.monotonic();proc=None;reason='UNKNOWN';code=None
    status={'status':'STARTING','started_unix':started,'selected_gpu':selected['gpu'],'max_resident_seconds':1770,'plan_sha256':plan_sha256,'qualification_only':True}
    write(out/'RESIDENT_STATUS.json',status)
    try:
        with (out/'child.log').open('wb') as log:
            assert sha(plan_path)==plan_sha256,'Plan changed before child launch'
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(plan_path),'--child','--expected-plan-sha256',plan_sha256,'--supervisor-pid',str(os.getpid()),'--deadline-monotonic',repr(start_mono+1760)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=partial(guard_before_exec,os.getpid(),start_mono+1760))
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
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--child',action='store_true');p.add_argument('--supervisor-pid',type=int);p.add_argument('--deadline-monotonic',type=float);p.add_argument('--expected-plan-sha256');a=p.parse_args()
    if a.child:
        if a.supervisor_pid is None or a.deadline_monotonic is None:p.error('Child must be launched by its bounded supervisor')
        arm_child_deadline(a.supervisor_pid,a.deadline_monotonic)
    elif a.supervisor_pid is not None or a.deadline_monotonic is not None or a.expected_plan_sha256 is not None:p.error('Child-only arguments')
    plan_bytes=a.plan.read_bytes();plan_sha256=hashlib.sha256(plan_bytes).hexdigest()
    if a.child:
        assert valid_hash(a.expected_plan_sha256) and plan_sha256==a.expected_plan_sha256,'Parent-child plan byte identity mismatch'
    plan=json.loads(plan_bytes)
    validate_launch_plan(plan)
    if a.child:child(plan,Path(plan['output']));return 0
    return parent(a.plan.resolve(),plan,plan_sha256)


if __name__=='__main__':raise SystemExit(main())
