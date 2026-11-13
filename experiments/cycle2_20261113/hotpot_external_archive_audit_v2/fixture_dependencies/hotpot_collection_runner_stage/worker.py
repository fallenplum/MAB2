"""One guarded worker; no retry, evaluator or label-vault interface."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
from orchestration_core import canonical,cost_profile,digest,graph_digest,require,validate_matrix_result
from plan_tools import load_execution_plan
from production_guard import verify_child_guard
from storage import ensure_disk,file_sha,json_load,progress,write_new


def execute_unit(runtime,row,unit,unit_dir,plan_sha,context,*,clock=time.monotonic):
    started=clock();unit_dir.mkdir(exist_ok=False)
    payload=runtime['collector'](row)
    require(set(row)=={'id','question','context','type','level'} and digest(row)==unit['input_sha256'],'Input-only row hash differs')
    require(graph_digest(payload)==unit['payload_sha256'],'Collector payload/length identity differs')
    context.update({'unit_index':unit['index'],'unit_payload_sha256':unit['payload_sha256']})
    write_new(unit_dir/'INPUT.json',{'plan_sha256':plan_sha,'unit':unit,'payload':payload})
    try:
        returned=runtime['matrix'].execute(payload,unit['seed1'],unit['seed2'],unit['edge_index'])
        data=returned.to_dict();validate_matrix_result(data,unit)
        require(data['status'] in ('COMPLETE','COMPLETE_WITH_INPUT_REJECTIONS'),'Matrix did not terminate normally')
        raw_pin=write_new(unit_dir/'MATRIX.json',data)
        profile=cost_profile(data,unit,clock()-started)
        receipt={'schema':'hotpot_unit_receipt_v1','index':unit['index'],'worker':unit['worker'],'plan_sha256':plan_sha,
                 'status':data['status'],'matrix_protocol_hash':data['matrix_protocol_hash'],'matrix_sha256':raw_pin,
                 'input_file_sha256':file_sha(unit_dir/'INPUT.json'),'terminal_keys':[c['key'] for c in data['conditions']],
                 'terminal_conditions':profile['terminal_conditions'],'overflow_conditions':profile['overflow_conditions'],
                 'cost_fields_complete':profile['cost_fields_complete'],'known_physical_cost_components':data['known_physical_cost_components'],
                 'total_physical_cost':data['total_physical_cost'],'cost_profile':profile,'scores_computed':False}
        write_new(unit_dir/'UNIT_RECEIPT.json',receipt)
        return receipt
    except Exception as error:
        partial=getattr(error,'partial_result',None)
        evidence=runtime['extract_error'](error)
        checkpoints={}
        for name,value in [('PARTIAL_MATRIX.json',partial.to_dict() if partial is not None else locals().get('data')),
                           ('DURABLE_ERROR_EVIDENCE.json',evidence or None)]:
            if value is None:continue
            try:write_new(unit_dir/name,value);checkpoints[name]='SAVED'
            except Exception as storage_error:checkpoints[name]='UNSAVED_'+type(storage_error).__name__
        try:
            write_new(unit_dir/'FAILED.json',{'status':'INCOMPLETE','error_type':type(error).__name__,
                      'partial_present':partial is not None,'private_checkpoints':checkpoints,
                      'persistent_storage_failure_may_prevent_recovery':any(v!='SAVED' for v in checkpoints.values()),'retry':False})
        except Exception:pass  # Parent sees nonzero exit/missing closure; never claim persistence.
        raise


def run_worker(plan,plan_sha,worker_id,root,deadline,*,runtime_builder=None):
    output=root/f'worker_{worker_id}';output.mkdir(exist_ok=False)
    minimum=plan['budget']['disk_min_free_bytes'];context={'attempt_id':plan['attempt_id'],'plan_sha256':plan_sha,'worker_id':worker_id}
    def before_request(_context=None):
        if time.monotonic()>=deadline:raise TimeoutError('WORKER_MONOTONIC_CAP')
        ensure_disk(root,minimum)
    before_request()
    rows=[json.loads(line) for line in Path(plan['paths']['questions']).read_bytes().splitlines()]
    require(len(rows)==384,'Fixed question coverage differs')
    if runtime_builder is None:
        from runtime_factory import build_runtime
        runtime_builder=build_runtime
    runtime=runtime_builder(plan,worker_id,output,context,before_request)
    assigned=[u for u in plan['schedule']['units'] if u['worker']==worker_id]
    receipts=[];released=False;prefix_published=False
    for unit in assigned:
        if unit['phase']=='remainder' and not released:
            prefix=[r['cost_profile'] for r in receipts]
            write_new(output/'PREFIX_COMPLETE.json',{'worker':worker_id,'plan_sha256':plan_sha,'profiles':prefix,'model_loads':runtime['loads']})
            prefix_published=True
            release=root/'RELEASE_REMAINDER.json'
            while not release.exists():before_request();time.sleep(.25)
            value=json_load(release)
            require(value['plan_sha256']==plan_sha and value['attempt_id']==plan['attempt_id'],'Release belongs to another attempt')
            frozen=root/'FORECAST_FROZEN.json'
            require(file_sha(frozen)==value['forecast_sha256'] and json_load(frozen)['status']=='FIT','Unbound or unsuccessful forecast release')
            released=True
        before_request()
        receipt=execute_unit(runtime,rows[unit['index']],unit,output/f"unit_{unit['index']:03d}",plan_sha,context)
        receipts.append(receipt)
        progress(output/'PROGRESS.json',{'completed_units':len(receipts),'terminal_conditions':sum(r['terminal_conditions'] for r in receipts),
                   'overflow_conditions':sum(r['overflow_conditions'] for r in receipts),'status':'RUNNING','scores_computed':False})
    require(prefix_published and released,'Worker never passed the prefix barrier')
    require(len(receipts)==len(assigned),'Worker coverage differs')
    write_new(output/'WORKER_COMPLETE.json',{'status':'COMPLETE_COLLECTOR_ONLY','plan_sha256':plan_sha,'worker':worker_id,
                'units':len(receipts),'terminal_conditions':sum(r['terminal_conditions'] for r in receipts),
                'overflow_conditions':sum(r['overflow_conditions'] for r in receipts),'scores_computed':False})


def main():
    parser=argparse.ArgumentParser()
    for key in ('plan','plan-sha256','admission','admission-sha256','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--worker',type=int,required=True);parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();verify_child_guard(args.parent_pid,args.deadline)
    plan,admission=load_execution_plan(args.plan,args.plan_sha256,args.admission,args.admission_sha256)
    require(str(Path(args.output).resolve())==admission['output_directory'],'Worker output differs from pinned admission')
    require(0<=args.worker<plan['schedule']['worker_count'],'Unknown worker')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(plan['budget']['workers'][args.worker]['gpu']),'GPU assignment differs')
    try:run_worker(plan,args.plan_sha256,args.worker,Path(args.output),args.deadline)
    except Exception as error:
        # No exception text/answer in progress output; private journals retain detail.
        print(json.dumps({'status':'INCOMPLETE','error_type':type(error).__name__,'retry':False}),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
