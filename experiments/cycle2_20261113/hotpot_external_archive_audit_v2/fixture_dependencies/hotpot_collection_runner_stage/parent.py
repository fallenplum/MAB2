"""Linux parent for one new frozen attempt; never opens raw matrix contents."""
import argparse
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from orchestration_core import ResidenceLedger,TERMINALS,forecast,monitor_counts,require
from plan_tools import load_execution_plan
from production_guard import arm_before_exec,kill_owned_group
from storage import ensure_disk,file_sha,json_load,progress,write_new


def collect_receipts(root,plan,plan_sha,cache=None):
    records=[];units={u['index']:u for u in plan['schedule']['units']}
    cache={} if cache is None else cache
    for w in range(plan['schedule']['worker_count']):
        directory=root/f'worker_{w}'
        if not directory.exists():continue
        for path in sorted(directory.glob('unit_*/UNIT_RECEIPT.json')):
            index=int(path.parent.name.removeprefix('unit_'))
            if index in cache:
                records.append(cache[index]['row']);continue
            row=json_load(path);unit=units[row['index']]
            require(index==row['index'] and path.parent.name==f'unit_{index:03d}','Receipt filename/index differs')
            require(row['worker']==w==unit['worker'] and row['plan_sha256']==plan_sha,'Unit receipt lineage differs')
            require(row['matrix_protocol_hash']==plan['runtime']['expected_matrix_protocol_hash'],'Unit runtime identity differs')
            require(row['terminal_conditions']==11 and row['scores_computed'] is False,'Unit terminal/scoring scope differs')
            require(row['terminal_keys']==list(TERMINALS),'Unit terminal identities/order differ')
            # Hash the opaque raw bytes, never decode or inspect answers.
            pin=file_sha(path)
            require(file_sha(path.parent/'MATRIX.json')==row['matrix_sha256'],'Raw matrix bytes changed')
            cache[row['index']]={'pin':pin,'row':row}
            records.append(row)
    require(len({r['index'] for r in records})==len(records),'Duplicate unit terminal receipt')
    return records


def future_fits(budget,used,predicted_remaining):
    total=[a+b for a,b in zip(used,predicted_remaining)]
    prior=sum(t['seconds'] for t in budget['prior_candidate_transactions'])
    return (all(v<=cap for v,cap in zip(total,budget['worker_caps_seconds']))
            and sum(total)<=budget['global_available_seconds_for_attempt']
            and prior+sum(total)<=budget['candidate_cap_seconds'])


def verify_first_seen_receipts(root,verified):
    for index,entry in verified.items():
        path=root/f"worker_{entry['row']['worker']}"/f'unit_{index:03d}'/'UNIT_RECEIPT.json'
        require(file_sha(path)==entry['pin'],'Immutable receipt changed after first observation')


def supervise(plan,plan_sha,root,processes,ledger,*,clock=time.monotonic,sleep=time.sleep,
              disk_guard=ensure_disk,kill_group=kill_owned_group):
    """Injectable process/clock boundaries enable CPU tests without launching models."""
    released=False;stop_reason=None;kill_errors=[];records=[];closed=set();verified={}
    minimum=plan['budget']['disk_min_free_bytes'];count=plan['schedule']['worker_count']
    try:
        while True:
            now=clock()
            for w,p in enumerate(processes):
                code=p.poll()
                if code is not None and w not in closed:
                    ledger.close(w,now);closed.add(w)
                    if code!=0:stop_reason=stop_reason or 'INCOMPLETE_WORKER_EXIT'
            if ledger.exceeded(now):stop_reason=stop_reason or 'INCOMPLETE_RESOURCE_HARDCAP'
            try:disk_guard(root,minimum)
            except OSError:stop_reason=stop_reason or 'INCOMPLETE_DISK_GUARD'
            records=collect_receipts(root,plan,plan_sha,verified)
            progress(root/'PROGRESS.json',{'status':stop_reason or ('COLLECTING_REMAINDER' if released else 'COLLECTING_PAID_PREFIX'),
                    **monitor_counts(records,ledger.spent(now)),'scores_accessed':False})
            prefix_files=[root/f'worker_{w}/PREFIX_COMPLETE.json' for w in range(count)]
            if not released and not stop_reason and all(p.exists() for p in prefix_files):
                profiles=[]
                for w,path in enumerate(prefix_files):
                    receipt=json_load(path)
                    require(receipt['worker']==w and receipt['plan_sha256']==plan_sha,'Prefix receipt lineage differs')
                    profiles.extend(receipt['profiles'])
                require(len(records)==8 and {r['index'] for r in records}==set(plan['schedule']['prefix_indices']),'No remainder may run before release')
                try:
                    predicted=forecast(plan['schedule'],profiles,plan['request_keys_by_edge'],ledger.spent(clock()),plan['budget'],plan['forecast_margin'])
                except ValueError:
                    stop_reason='INCOMPLETE_RESOURCE_FORECAST_UNSUPPORTED'
                else:
                    predicted.update({'plan_sha256':plan_sha,'attempt_id':plan['attempt_id'],
                         'prefix_receipt_sha256':[file_sha(path) for path in prefix_files]})
                    pin=write_new(root/'FORECAST_FROZEN.json',predicted)
                    if predicted['status']!='FIT':stop_reason='INCOMPLETE_RESOURCE_FORECAST'
                    elif any(p.poll() is not None for p in processes):stop_reason='INCOMPLETE_WORKER_BEFORE_RELEASE'
                    elif not future_fits(plan['budget'],ledger.spent(clock()),predicted['predicted_remaining_seconds_by_worker']):
                        stop_reason='INCOMPLETE_RESOURCE_RELEASE_DELAY'
                    else:
                        write_new(root/'RELEASE_REMAINDER.json',{'plan_sha256':plan_sha,'attempt_id':plan['attempt_id'],
                              'forecast_sha256':pin,'release_recheck_resident_seconds':ledger.spent(clock()),
                              'release_recheck_fits':True,'scores_accessed':False})
                        released=True
            if stop_reason:
                for p in processes:
                    try:kill_group(p,p.pid)
                    except Exception as error:kill_errors.append(type(error).__name__)
                if kill_errors:break  # fail closed; parent-death guard remains armed
            if len(closed)==count:break
            sleep(.25)
    except Exception as error:
        stop_reason='INCOMPLETE_PARENT_'+type(error).__name__
        for p in processes:
            try:kill_group(p,p.pid)
            except Exception as kill_error:kill_errors.append(type(kill_error).__name__)
    # Observe exits conservatively, charging kill latency instead of clamping to cap.
    until=clock()+5
    while len(closed)<count and clock()<until and not kill_errors:
        for w,p in enumerate(processes):
            if w not in closed and p.poll() is not None:ledger.close(w,clock());closed.add(w)
        if len(closed)<count:sleep(.05)
    spent=ledger.spent(clock())
    complete=(not stop_reason and released and len(closed)==count and len(records)==384
              and {r['index'] for r in records}==set(range(384)) and sum(r['terminal_conditions'] for r in records)==4224)
    if not complete:stop_reason=stop_reason or 'INCOMPLETE_COVERAGE_OR_RESIDENCE'
    if complete:
        verify_first_seen_receipts(root,verified)
        records=collect_receipts(root,plan,plan_sha)  # final raw-byte recheck after every child has exited
        for w in range(count):
            done=json_load(root/f'worker_{w}/WORKER_COMPLETE.json')
            require(done['status']=='COMPLETE_COLLECTOR_ONLY' and done['plan_sha256']==plan_sha,'Worker completion differs')
    status=('COMPLETE_WITH_CAPACITY_FAILURES' if any(r['overflow_conditions'] for r in records) else 'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT') if complete else stop_reason
    raw_pins={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ('PROGRESS.json','PARENT_RESIDENCE.json','ATTEMPT_RESULT.json')}
    write_new(root/'RAW_OUTPUT_MANIFEST.json',{'status':'CLOSED_SNAPSHOT' if len(closed)==count else 'UNSTABLE_INCOMPLETE_OBSERVED_SNAPSHOT',
               'plan_sha256':plan_sha,'files':raw_pins,'scores_accessed':False})
    residence={'transaction_id':plan['attempt_id'],'worker_started_monotonic':ledger.started,
        'worker_closed_monotonic':ledger.closed,'observed_seconds_by_worker':spent,
        'closed_sum_seconds':sum(spent) if len(closed)==count else None,'observed_residence_lower_bound_seconds':sum(spent),
        'all_workers_closed':len(closed)==count,'includes_model_loading_and_barrier_idle':True,
        'measurement':'parent monotonic from before Popen until observed process exit; parallel workers summed',
        'cap_used_as_cost':False,'unknown_unclosed_residence':len(closed)!=count,'kill_integrity_errors':kill_errors}
    write_new(root/'PARENT_RESIDENCE.json',residence)
    result={'status':status,'attempt_id':plan['attempt_id'],'plan_sha256':plan_sha,
        **monitor_counts(records,spent),'all_worker_residences_closed':len(closed)==count,
        'all_raw_request_cost_fields_complete':all(r['cost_fields_complete'] for r in records) if complete else False,
        'score_release_authorized':False,'scores_computed':False,'automatic_retry':False,
        'raw_manifest_sha256':file_sha(root/'RAW_OUTPUT_MANIFEST.json'),'residence_sha256':file_sha(root/'PARENT_RESIDENCE.json')}
    write_new(root/'ATTEMPT_RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    for key in ('plan','plan-sha256','admission','admission-sha256','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    require(sys.platform.startswith('linux'),'Production parent requires qualified Linux; CPU staging only elsewhere')
    plan,admission=load_execution_plan(args.plan,args.plan_sha256,args.admission,args.admission_sha256)
    root=Path(args.output).resolve();require(not root.exists(),'No overwrite/resume; new attempt directory required')
    require(str(root)==admission['output_directory'],'Output differs from private admission')
    root.mkdir(parents=True,exist_ok=False);ensure_disk(root,plan['budget']['disk_min_free_bytes'])
    write_new(root/'ATTEMPT_STARTED.json',{'attempt_id':plan['attempt_id'],'plan_sha256':args.plan_sha256,
           'admission_sha256':args.admission_sha256,'parent_pid':os.getpid(),'scores_accessed':False})
    ledger=ResidenceLedger(plan['budget']['worker_caps_seconds'],plan['budget']['global_available_seconds_for_attempt'])
    processes=[];logs=[]
    try:
        for w,spec in enumerate(plan['budget']['workers']):
            ensure_disk(root,plan['budget']['disk_min_free_bytes'])
            start=time.monotonic();ledger.start(w,start);deadline=start+plan['budget']['worker_caps_seconds'][w]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(spec['gpu']),PYTHONDONTWRITEBYTECODE='1',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(root/f'worker_{w}.private.log').open('xb');logs.append(log)
            command=[sys.executable,'-B',str(Path(__file__).with_name('worker.py')),'--plan',args.plan,'--plan-sha256',args.plan_sha256,
                '--admission',args.admission,'--admission-sha256',args.admission_sha256,'--output',str(root),'--worker',str(w),
                '--parent-pid',str(os.getpid()),'--deadline',repr(deadline)]
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                   preexec_fn=functools.partial(arm_before_exec,os.getpid(),deadline))
            processes.append(child)
        result=supervise(plan,args.plan_sha256,root,processes,ledger)
    except Exception as error:
        for child in processes:
            try:kill_owned_group(child,child.pid)
            except Exception:pass
        # No retry or attempt reuse; retain all partial outputs and raw journals.
        write_new(root/'PARENT_FAILED.json',{'status':'INCOMPLETE_PARENT_SETUP_OR_FINALIZE','error_type':type(error).__name__,
                    'started_workers':len(processes),'observed_resident_seconds':ledger.spent(time.monotonic()),'closed_residence_unknown':True})
        raise SystemExit(2)
    finally:
        for log in logs:log.close()
    print(json.dumps({'status':result['status'],'completed_units':result['completed_units'],
                      'terminal_conditions':result['terminal_conditions'],'score_release_authorized':False}))
    if not result['status'].startswith(('COLLECTION_COMPLETE','COMPLETE_WITH_CAPACITY')):raise SystemExit(2)


if __name__=='__main__':main()
