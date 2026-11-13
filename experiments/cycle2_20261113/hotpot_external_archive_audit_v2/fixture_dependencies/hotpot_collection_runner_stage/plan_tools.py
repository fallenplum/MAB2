"""Prepare only an already selected, unlabelled 384-unit plan; never admit it."""
import argparse
import importlib.util
from pathlib import Path
from orchestration_core import *
from storage import file_sha,json_load,pinned_json,write_new


def validate_lengths(length,runtime):
    require(length.get('server_runtime_equivalence') is True,'Server tokenizer-only equivalence required; local capacity precheck is insufficient')
    require(set(length['tokenizer_identity_sha256'])=={'qwen','mistral'},'Both tokenizers must be pinned')
    for name in ('qwen','mistral'):
        model=runtime['models'][name]
        require(is_sha(model['tokenizer_identity_sha256']) and length['tokenizer_identity_sha256'][name]==model['tokenizer_identity_sha256'],'Tokenizer identity differs')
        require(length['model_identity_sha256'][name]==graph_digest(model['identity']),'Length/runtime model identity differs')
    require(length['collector_source_sha256']==runtime['collector']['sha256'],'Collector length/payload source differs')


def verify_payloads(rows,manifest,length,runtime):
    collector=runtime['collector']
    require(file_sha(collector['path'])==collector['sha256'],'Collector source differs')
    spec=importlib.util.spec_from_file_location('planning_pinned_input_collector',collector['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    by_id={r['id']:r for r in length['rows']}
    require(len(rows)==384,'Question file has wrong length')
    for index,row in enumerate(rows):
        require(set(row)=={'id','question','context','type','level'},'Question JSONL must use five input-only fields')
        require(row['id']==manifest[index]['id'] and digest(row)==manifest[index]['input_sha256'],'Question row/selection identity mismatch')
        require(graph_digest(module.collector_payload(row))==by_id[row['id']]['payload_sha256'],'Length and actual full collector payload differ')


def validate_engineering(runtime,budget):
    acceptance=runtime['engineering_admission']
    value=pinned_json(acceptance['path'],acceptance['sha256'])
    require(value['decision']=='PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION','Independent engineering acceptance missing')
    require(value['matrix_protocol_hash']==runtime['expected_matrix_protocol_hash'],'Qualified matrix identity differs')
    for name in ('qwen','mistral'):
        require(value['model_identity_sha256'][name]==graph_digest(runtime['models'][name]['identity']),'Qualified model identity differs')
    for name in ('matrix','durable'):
        require(value['module_sha256'][name]==runtime['modules'][name]['sha256'],'Qualified module source differs')
    required={'qualification','identity','residence','plan','source_manifest','root_closed_review'}
    require(required<=set(value['evidence']),'Qualification evidence closure missing')
    evidence_root=Path(acceptance['path']).resolve().parent
    for evidence in value['evidence'].values():
        relative=Path(evidence['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Engineering evidence must use bundle-relative paths without traversal')
        resolved=(evidence_root/relative).resolve()
        require(resolved.is_relative_to(evidence_root),'Engineering evidence escapes acceptance bundle')
        require(file_sha(resolved)==evidence['sha256'],'Engineering evidence changed')
    tx=[t for t in budget['prior_candidate_transactions'] if t['transaction_id']==value['candidate_transaction_id']]
    require(len(tx)==1 and tx[0]['seconds']==value['candidate_closed_seconds'],'Current-round qualification residence must be attributed once')
    require(finite(value['candidate_closed_seconds']) and value['candidate_closed_seconds']>0,'Closed qualification residence required')
    return value


def geometry_request_keys(geometry):
    answer={}
    for row in geometry['per_probe_edge']:
        keys=[]
        for name in FIRST+SECOND:
            arch='independent12' if name=='INDEPENDENT12' else 'single_m' if name=='SINGLE_M' else 'graph'
            for node in row['closures'][name]:
                model='mistral' if node=='aggregate' or int(node[-1])%2 else 'qwen'
                keys.append('|'.join((arch,model,node)))
        require(len(keys)==87,'Wrong geometry request count');answer[str(row['probe_edge'])]=keys
    require(set(answer)=={str(e) for e in EDGES},'Geometry edges differ')
    return answer


def prepare_plan(*,selection,selection_receipt,questions,lengths,protocol,bindings,budget,output,attempt_id):
    paths={name:Path(value).resolve() for name,value in locals().copy().items()
           if name in ('selection','selection_receipt','questions','lengths','protocol','bindings','budget')}
    source={name:json_load(path) for name,path in paths.items() if name!='questions'}
    pins={name:file_sha(path) for name,path in paths.items()}
    m=source['selection'];receipt=source['selection_receipt'];length=source['lengths'];science=source['protocol'];runtime=source['bindings'];resources=source['budget']
    require(receipt['status']=='COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION','Selection receipt incomplete')
    require(receipt['selection_manifest_sha256']==pins['selection'] and receipt['questions_jsonl_sha256']==pins['questions'],'Selection lineage differs')
    require(length['status']=='TOKENIZER_LENGTHS_COMPLETE_NO_LABELS' and length['labels_read'] is False,'Unlabelled length receipt required')
    require(length['questions_jsonl_sha256']==pins['questions'] and length['selection_manifest_sha256']==pins['selection'],'Tokenizer lengths belong to other inputs')
    require(length['payload_hash_algorithm']=='graph_credit_canonical_utf8_sha256_no_newline','Payload fingerprint convention differs')
    validate_lengths(length,runtime)
    validate_protocol(science)
    worker_count=len(resources['workers']);validate_budget(resources,worker_count)
    require(len({w['gpu'] for w in resources['workers']})==worker_count,'Workers require distinct preassigned GPU devices')
    schedule=build_schedule(m['questions'],length['rows'],worker_count)
    rows=[]
    for line in paths['questions'].read_bytes().splitlines():
        value=__import__('json').loads(line)
        require(set(value)=={'id','question','context','type','level'},'Question JSONL must use five input-only fields')
        rows.append(value)
    verify_payloads(rows,m['questions'],length,runtime)
    geo=pinned_json(runtime['geometry']['path'],runtime['geometry']['sha256'])
    plan={'schema':'hotpot_collection_plan_v1','status':'PREPARED_NOT_FROZEN_NOT_ADMITTED','attempt_id':attempt_id,
          'paths':{name:str(path) for name,path in paths.items()},'input_sha256':pins,'schedule':schedule,
          'request_keys_by_edge':geometry_request_keys(geo),'budget':resources,'runtime':runtime,
          'forecast_margin':science['resource']['minimum_forecast_time_margin'],
          'sources':runtime['source_files'],'labels_read':False,'new_model_calls':0,'new_GPU_hours':0}
    require(isinstance(attempt_id,str) and attempt_id and '/' not in attempt_id and '\\' not in attempt_id,'Invalid attempt identity')
    require(all(file_sha(path)==pins[name] for name,path in paths.items()),'Input changed during planning')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_new(out/'PLAN.json',plan)
    return plan


def load_execution_plan(path,pin,admission_path,admission_pin):
    plan=pinned_json(path,pin);admission=pinned_json(admission_path,admission_pin)
    require(plan['status']=='FROZEN_COLLECTION_PLAN','Collection plan is not frozen')
    require(admission['decision']=='APPROVED_FOR_FROZEN_COLLECTION' and admission['plan_sha256']==pin,'No pinned collection admission')
    require(admission['attempt_id']==plan['attempt_id'] and admission['new_attempt_only'] is True,'Attempt lineage differs')
    require(plan['attempt_id'] not in admission['previous_attempt_ids'],'Attempt transaction already used; no silent resume')
    require(admission['qualification_source_bindings_checked'] is True and admission['R1_closed'] is True,'Engineering or earlier-round closure missing')
    require(admission['budget_sha256']==digest(plan['budget']),'Budget admission differs')
    validate_budget(plan['budget'],plan['schedule']['worker_count'],executable=True)
    protocol=pinned_json(plan['paths']['protocol'],plan['input_sha256']['protocol']);validate_protocol(protocol,executable=True)
    require(protocol['resource']['cap_is_authorized'] is True
            and protocol['resource']['actual_cap_seconds']==plan['budget']['candidate_cap_seconds']
            and protocol['resource']['per_worker_caps_seconds']==plan['budget']['worker_caps_seconds'],
            'Frozen scientific resource cap differs from allocated plan')
    require(plan['forecast_margin']==protocol['resource']['minimum_forecast_time_margin'],'Forecast margin differs from protocol')
    require(all(v not in (None,'REQUIRED_BEFORE_ADMISSION') for v in protocol['required_pins_before_execution'].values()),'Scientific hard dependency missing')
    validate_engineering(plan['runtime'],plan['budget'])
    for name,path in plan['paths'].items():require(file_sha(path)==plan['input_sha256'][name],'Plan input changed: '+name)
    require(plan['runtime']==json_load(plan['paths']['bindings']),'Embedded runtime differs from pinned bindings')
    require(plan['budget']==json_load(plan['paths']['budget']),'Embedded budget differs from pinned allocation input')
    require(plan['sources']==plan['runtime']['source_files'],'Source closure differs from runtime bindings')
    selected=json_load(plan['paths']['selection'])['questions'];length=json_load(plan['paths']['lengths'])
    validate_lengths(length,plan['runtime'])
    require(build_schedule(selected,length['rows'],plan['schedule']['worker_count'])==plan['schedule'],'Frozen schedule differs from deterministic manifest/length order')
    rows=[__import__('json').loads(line) for line in Path(plan['paths']['questions']).read_bytes().splitlines()]
    verify_payloads(rows,selected,length,plan['runtime'])
    geometry=pinned_json(plan['runtime']['geometry']['path'],plan['runtime']['geometry']['sha256'])
    require(geometry_request_keys(geometry)==plan['request_keys_by_edge'],'Forecast geometry differs')
    for row in plan['sources']:require(file_sha(row['path'])==row['sha256'],'Execution source changed')
    source_pairs={(str(Path(row['path']).resolve()),row['sha256']) for row in plan['sources']}
    require(all((str(Path(spec['path']).resolve()),spec['sha256']) in source_pairs
                for spec in plan['runtime']['modules'].values()),'Runtime module missing from frozen source closure')
    names={Path(row['path']).name for row in plan['sources']}
    require({'parent.py','worker.py','plan_tools.py','orchestration_core.py','storage.py','production_guard.py','runtime_factory.py','durable_backend.py','matrix_executor.py'}<=names,'Frozen source closure omits an execution component')
    paths={Path(row['path']).resolve() for row in plan['sources']}
    require(all((Path(__file__).parent/name).resolve() in paths for name in ('parent.py','worker.py','plan_tools.py','orchestration_core.py','storage.py','production_guard.py','runtime_factory.py')),'Actual orchestration files are not in the frozen source closure')
    require(is_sha(plan['runtime']['expected_matrix_protocol_hash']),'Qualified matrix identity missing')
    return plan,admission


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('selection','selection-receipt','questions','lengths','protocol','bindings','budget','output','attempt-id'):
        parser.add_argument('--'+key,required=True)
    values=vars(parser.parse_args());result=prepare_plan(**values)
    print(__import__('json').dumps({'status':result['status'],'questions':384,'conditions':4224,'new_model_calls':0}))
