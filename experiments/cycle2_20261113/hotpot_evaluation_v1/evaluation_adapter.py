"""Closed-archive Hotpot binding/scoring adapter. CPU preparation, no archive IO.

All dataset/archive inputs are caller-supplied bytes. Real labels may be loaded
only through evaluate_after_release after an externally pinned review is bound.
This module cannot authenticate the human/provider supplying that trusted pin.
It does not perform the external native/durable/attempt-ledger audit itself.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

N=384
SEEDS=(26111301,26111302)
EDGES=(1,5,7,9)
GRAPHS=('A','B','A_MINUS_E','B_MINUS_E')
BASELINES={'NO_EDGE':'no_edge','INDEPENDENT12':'independent12','SINGLE_M':'single'}
ORDER=tuple('S1/'+n for n in ('A','B','NO_EDGE','A_MINUS_E','B_MINUS_E','INDEPENDENT12','SINGLE_M'))+tuple('S2/'+n for n in GRAPHS)
NODES=tuple(f'r{r}:w{w}' for r in range(1,4) for w in range(4))+('aggregate',)
ROUTE=('qwen','mistral','qwen','mistral')
FIELDS=('calls','input_tokens','output_tokens','CUDA_event_seconds')
INTEGRITY=('node','input_fingerprint','seed','messages','generation','wire_message','parsed',
           'message_metadata','received_message_tokens','self_state_tokens')
HELPER_PINS={'hotpot_support':'60d056c6c74080ff1186762074b6fb85b9ea7e229143bedd74299d87a1864c0e',
 'applicability_statistics':'3417dc11bc583d72a2ec8c8cc139c6ee490aa039d8eb386c97f7e2aa52060bfe',
 'point_frontier':'6f641a3ff9c00afc2b045bc08435e42c602df5756747c58a7852f017da7bf7f9'}
SELECTOR_PIN='5edc8dc62b349be49c2c939c9c3f93ba1a8795b3c6fc7c951aec3ab9cd3d8bfd'
BINDINGS=('scientific_protocol_sha256','selection_manifest_sha256','plan_sha256','source_manifest_sha256',
 'matrix_protocol_hash','graph_protocol_hash','independent12_protocol_hash','single_protocol_hash',
 'scorer_sha256','statistics_sha256','frontier_sha256','selector_sha256','evaluation_adapter_sha256')
METADATA=('RAW_OUTPUT_MANIFEST.json','PARENT_RESIDENCE.json','ATTEMPT_RESULT.json')
REQUIRED_REVIEW_SCOPES={'source_model_tokenizer_identity','native_durable_generation_and_cost_lineage',
 'cache_input_and_output_lineage','all_attempts_and_physical_residence_ledger','no_labels_during_collection'}
_BOUND_TOKEN=object()


class BindingError(ValueError):pass


def require(value,message):
    if not value:raise BindingError(message)


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(raw):
    require(type(raw) is bytes,'Archive inputs must be exact bytes')
    return hashlib.sha256(raw).hexdigest()


def is_hash(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def decode(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    require(type(raw) is bytes,'Exact JSON bytes required')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(BindingError('Nonfinite JSON value')))


def _helper(name):
    path=Path(__file__).with_name('frozen_dependencies')/(name+'.py')
    require(sha(path.read_bytes())==HELPER_PINS[name],'Frozen helper bytes differ')
    spec=importlib.util.spec_from_file_location('_hotpot_evaluation_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _model(node):
    require(node in NODES,'Unexpected node alias')
    return 'mistral' if node=='aggregate' else ROUTE[int(node[-1])]


def _mask(name,edge):
    if name in ('INDEPENDENT12','SINGLE_M'):return None
    number=4095 if name in ('A','A_MINUS_E') else 674 if name in ('B','B_MINUS_E') else 0
    if name.endswith('MINUS_E'):number &= ~(1<<edge)
    return [(number>>i)&1 for i in range(12)]


def _finite(value,integer=False,positive=False):
    return (type(value) is int if integer else type(value) in (int,float)) and math.isfinite(value) and (value>0 if positive else value>=0)


def _records(execution,name):
    if name not in ('INDEPENDENT12','SINGLE_M'):
        records=execution['nodes'];require(set(records)==set(NODES),'Graph logical nodes missing/extra')
        return {node:records[node] for node in NODES}
    slots=execution['slots']
    expected=('aggregate',) if name=='SINGLE_M' else tuple(f'r{s}:w{r}' for r in range(4) for s in range(1,4))+('aggregate',)
    require(len(slots)==len(expected),'Independent/single slot coverage differs')
    # Node names and role/sample are authoritative; never guess an aggregator by position.
    require({s['node'] for s in slots}==set(expected),'Independent/single node aliases differ')
    records={}
    for slot in slots:
        node=slot['node'];require(slot['model']==_model(node),'Slot model route differs')
        require(slot['attempted_backend_request'] is True and slot['status'] in ('SUCCESS','FORMAT_ERROR'),'Slot incomplete')
        expected_index=expected.index(node)
        require(slot['index']==expected_index,'Slot index/node binding differs')
        require(slot['role']==(None if node=='aggregate' else int(node[-1])),'Slot role differs')
        require(slot['sample']==(None if node=='aggregate' else int(node[1])),'Slot sample differs')
        record=slot['record'];require(record is not None,'Completed slot lacks normalized record')
        require(slot['generation']==record['generation'] and slot['seed']==record['seed'],'Slot generation/seed differs')
        require(slot['messages']==record['messages'] and slot['input_fingerprint']==record['input_fingerprint'],'Slot input differs')
        records[node]=record
    return {node:records[node] for node in expected}


def _node_costs(records):
    per_model={m:{f:0 for f in FIELDS} for m in ('qwen','mistral')}
    total={'calls':len(records),'input_tokens':0,'output_tokens':0,'received_message_tokens':0,'self_state_tokens':0,
           'model_elapsed_seconds':0.,'gpu_seconds':0.,'peak_allocated_bytes':0}
    for node,record in records.items():
        generation=record['generation'];cost=per_model[_model(node)]
        cost['calls']+=1
        for key in ('input_tokens','output_tokens'):
            require(_finite(generation.get(key),integer=True,positive=key=='input_tokens'),'Unknown/invalid complete logical token cost')
            cost[key]+=generation[key];total[key]+=generation[key]
        require(_finite(generation.get('gpu_seconds')),'Unknown/invalid complete CUDA event cost')
        cost['CUDA_event_seconds']+=generation['gpu_seconds'];total['gpu_seconds']+=generation['gpu_seconds']
        require(_finite(generation.get('elapsed_seconds')) and _finite(generation.get('gpu_peak_allocated_bytes'),integer=True),'Invalid logical timing/allocation')
        total['model_elapsed_seconds']+=generation['elapsed_seconds']
        total['peak_allocated_bytes']=max(total['peak_allocated_bytes'],generation['gpu_peak_allocated_bytes'])
        for key in ('received_message_tokens','self_state_tokens'):
            require(_finite(record.get(key),integer=True),'Invalid logical receiver/self count');total[key]+=record[key]
    return per_model,total


def _equal_cost(reported,recomputed):
    for key,value in recomputed.items():
        require(_finite(reported.get(key)) and math.isclose(reported[key],value,rel_tol=1e-12,abs_tol=1e-9),
                'Full logical service cost differs: '+key)


def _validate_matrix(matrix,unit,bindings,payload):
    require(matrix['operator_id']=='single_question_hotpot_applicability_matrix_v1_cpu_staging','Wrong matrix operator')
    require(matrix['matrix_protocol_hash']==bindings['matrix_protocol_hash'],'Matrix protocol differs')
    require(matrix['question_id']=='hotpotqa_'+unit['id'] and matrix['question_fingerprint']==unit['payload_sha256'],'Matrix question binding differs')
    require((matrix['seed1'],matrix['seed2'])==SEEDS and matrix['probe_edge']==unit['edge_index'],'Matrix seed/edge differs')
    conditions=matrix['conditions'];require([c['key'] for c in conditions]==list(ORDER),'Exact unique 11 condition keys required')
    require(matrix['condition_set_terminal'] is True and matrix['scores_computed'] is False,'Unclosed/scored collector matrix')
    require(all(c['status'] in ('COMPLETE','INPUT_REJECTED') for c in conditions),'Infrastructure/partial conditions cannot be scored')
    overflow=sum(c['status']=='INPUT_REJECTED' for c in conditions)
    require(matrix['status']==('COMPLETE_WITH_INPUT_REJECTIONS' if overflow else 'COMPLETE'),'Matrix status differs')
    require(matrix['all_conditions_completed']==(overflow==0),'Matrix completion summary differs')
    lookup={c['key']:c for c in conditions};execution_ids={}
    full_records={};logical={};totals=[]
    for condition in conditions:
        prefix,name=condition['key'].split('/');number=int(prefix[1:]);seed=SEEDS[number-1]
        require(condition['name']==name and condition['seed_number']==number and condition['seed']==seed,'Condition name/seed mismatch')
        mask=_mask(name,unit['edge_index']);require(condition['mask']==mask,'Wrong graph mask/probe edge')
        protocol=bindings['independent12_protocol_hash' if name=='INDEPENDENT12' else 'single_protocol_hash' if name=='SINGLE_M' else 'graph_protocol_hash']
        if condition['status']=='INPUT_REJECTED':
            require(condition['execution'] is None and condition['logical_service_cost'] is None,'Overflow must not contain completed execution/cost')
            require(condition['failure_category'] in ('INPUT_TOKEN_LIMIT','MODEL_POSITION_LIMIT'),'Unrecognized overflow source')
            partial=condition['partial_execution'];require(partial is not None and partial['protocol_hash']==protocol,'Overflow protocol binding missing')
            require(partial['question_fingerprint']==unit['payload_sha256'] and partial['seed']==seed,'Overflow input binding differs')
            trace=condition['request_trace'];require(trace and trace[-1]['generation'] is None and trace[-1]['error_type']=='ValueError','Overflow lacks preflight failure trace')
            logical[condition['key']]={m:{f:(0 if name=='SINGLE_M' and m=='qwen' else None) for f in FIELDS} for m in ('qwen','mistral')}
            continue
        execution=condition['execution'];require(execution is not None and condition['error_type'] is None,'Complete execution missing/error set')
        require(execution['question_id']==matrix['question_id'] and execution['question_fingerprint']==unit['payload_sha256'],'Execution question differs')
        require(execution['seed']==seed and execution['protocol_hash']==protocol,'Execution seed/protocol differs')
        if mask is not None:require(execution['mask']==mask,'Execution mask differs')
        else:
            expected_id=('independent_4roles_x3samples_plus_aggregate_v2_cost_reconciliation_staging' if name=='INDEPENDENT12' else 'single_mistral_full_context_direct_v1_cpu_staging')
            require(execution['control_id']==expected_id,'Separate control identity differs')
            require(execution['status']=='COMPLETE','Separate control not complete')
        require(execution['execution_id'] not in execution_ids,'Duplicate execution identity')
        execution_ids[execution['execution_id']]=condition['key']
        records=_records(execution,name);full_records[condition['key']]=records
        require(execution['answer']==records['aggregate']['parsed']['answer'],'Execution answer differs from strict aggregate record')
        costs,total=_node_costs(records);logical[condition['key']]=costs;totals.append(total)
        _equal_cost(condition['logical_service_cost'],total);_equal_cost(execution['logical_service_cost'],total)
        expected={'qwen':0,'mistral':1} if name=='SINGLE_M' else {'qwen':6,'mistral':7}
        require({m:costs[m]['calls'] for m in costs}==expected,'Complete logical route geometry differs')
        for node,record in records.items():
            require(record['node']==node and record['output_digest']==digest({k:record[k] for k in INTEGRITY}),'Node integrity digest mismatch')
            node_seed=int(digest([matrix['question_id'],seed,node])[:16],16)%(2**31-1)
            cap=256 if node=='aggregate' else 128
            require(record['seed']==node_seed,'Node seed differs')
            full=payload['question']+'\n\nSupplied context:\n'+payload['context']
            require(len(record['messages'])==2 and record['messages'][0]['role']=='system' and record['messages'][1]['role']=='user'
                and record['messages'][1]['content'].startswith(full),'Full original question/context absent from node')
            require(record['input_fingerprint']==digest({'question':unit['payload_sha256'],'node':node,'messages':record['messages'],
                'seed':node_seed,'protocol':protocol,'max_new_tokens':cap}),'Node input fingerprint differs')
            require(record['generation']['input_tokens']<=8192 and record['generation']['output_tokens']<=cap,'Generation violates frozen cap')
            require(type(record['message_metadata']['format_ok']) is bool,'Explicit formatter flag required')
            parsed=record['parsed'];require(set(parsed)=={'answer','evidence','uncertainty'} and all(isinstance(v,str) for v in parsed.values()),'Malformed normalized parsed message')
    # Validate original-base lineage using explicit keys, not list positions.
    for condition in conditions:
        if condition['status']!='COMPLETE':continue
        name=condition['name'];key=condition['key'];number=condition['seed_number']
        source_name={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(name)
        base_key=f'S{number}/{source_name}' if source_name else None
        expected_source=base_key if base_key and lookup[base_key]['status']=='COMPLETE' else None
        require(condition['cache_source_condition']==expected_source,'Wrong/missing original base condition')
        records=full_records[key];trace={r['generation']['call_id']:r for r in condition['request_trace'] if r['generation'] is not None}
        require(len(trace)==len(condition['request_trace']),'Complete condition has missing/duplicate raw request')
        reused=0
        for node,record in records.items():
            if record['reused']:
                reused+=1
                require(expected_source is not None and execution_ids.get(record['source_execution_id'])==expected_source,'Cross-condition/cache source leakage')
                original=full_records[expected_source][node]
                require(all(record[k]==original[k] for k in INTEGRITY),'Reused record differs from original base')
            else:
                require(record['source_execution_id'] is None,'Nonreused record has cache source')
                call=record['generation']['call_id'];require(call in trace,'Physical generation absent from own condition')
                observed=trace[call]
                require(observed['node']==node and observed['model']==_model(node),'Actual physical route differs')
                require(observed['generation']==record['generation'] and observed['messages']==record['messages'] and observed['seed']==record['seed'],'Observed generation/input differs')
        require(condition['execution']['cache_reuse_count']==reused and len(trace)==len(records)-reused,'Cache/physical count differs')
    if not overflow:
        total={key:(max(row[key] for row in totals) if key=='peak_allocated_bytes' else sum(row[key] for row in totals)) for key in totals[0]}
        _equal_cost(matrix['total_logical_service_cost'],total)
    else:require(matrix['total_logical_service_cost'] is None,'Overflow full logical cost cannot be fabricated')
    return full_records,logical,overflow


@dataclass(frozen=True)
class BoundArchive:
    questions:tuple
    matrices:tuple
    logical_costs:tuple
    overflow_conditions:int
    review_sha256:str
    label_vault_sha256:str
    bindings:dict
    physical_archive_pins:dict
    semantic_sha256:str
    _validation_token:object


def _bound_semantics(questions,matrices,logical,overflow,review_sha,label_sha,bindings,physical):
    return digest([questions,matrices,logical,overflow,review_sha,label_sha,bindings,physical])


def bind_archive(*,selection_manifest_bytes,files,review_bytes,expected_review_sha256,expected_bindings):
    """Pure byte-to-bound-archive validation; never loads or scores labels.

    files contains only the three final archive metadata files and every unit's
    INPUT/MATRIX/UNIT_RECEIPT. Native/durable raw journals remain with the external
    reviewer; their audited immutable manifest is pinned by the review receipt.
    """
    require(is_hash(expected_review_sha256) and sha(review_bytes)==expected_review_sha256,'External archive review SHA mismatch/absent')
    review=decode(review_bytes)
    require(review['schema']=='hotpot_closed_archive_score_release_v1' and review['decision']=='APPROVED_FOR_SCORING','No external score release')
    require(set(expected_bindings)==set(BINDINGS) and all(is_hash(v) for v in expected_bindings.values()),'Required externally frozen bindings absent')
    require(review['bindings']==expected_bindings,'Review/frozen binding mismatch')
    bindings=dict(expected_bindings)
    require(bindings['evaluation_adapter_sha256']==sha(Path(__file__).read_bytes()),'Evaluation adapter source differs')
    require(bindings['scorer_sha256']==HELPER_PINS['hotpot_support'] and bindings['statistics_sha256']==HELPER_PINS['applicability_statistics']
        and bindings['frontier_sha256']==HELPER_PINS['point_frontier'] and bindings['selector_sha256']==SELECTOR_PIN,'Frozen helper identity differs')
    require(review['scientific_protocol_status']=='FROZEN','Candidate protocol cannot release scoring')
    require(set(review['reviewed_scopes'])==REQUIRED_REVIEW_SCOPES,'External audit scope incomplete')
    require(review['unique_questions']==384 and review['terminal_conditions']==4224 and review['runtime_failure_count']==0,'Archive not fully closed')
    require(review['all_attempts_closed'] is True and review['physical_ledger_closed'] is True,'Unclosed attempt/physical ledger')
    require(is_hash(review['label_vault_sha256']),'Label vault pin required before release')
    require(sha(selection_manifest_bytes)==bindings['selection_manifest_sha256'],'Selection manifest bytes differ')
    manifest=decode(selection_manifest_bytes);questions=manifest['questions']
    require(manifest['classification']=='PRIVATE_NO_LABELS' and len(questions)==384,'Fixed private selection384 required')
    require(manifest['pins']['protocol_sha256']==bindings['scientific_protocol_sha256'] and manifest['pins']['selector_sha256']==SELECTOR_PIN,'Selection source/protocol differs')
    require(len({q['id'] for q in questions})==384 and len({q['question_group_sha256'] for q in questions})==384,'Duplicate question/group')
    require(Counter(q['edge_index'] for q in questions)==Counter({e:96 for e in EDGES}),'Expected 96 questions per edge')
    for q in questions:
        require(isinstance(q['id'],str) and q['id'] and not q['id'].startswith('hotpotqa_'),'Raw official question ID required')
        require(all(is_hash(q[k]) for k in ('question_group_sha256','input_sha256','content_sha256')),'Invalid selection identity hash')
        require((q['generation_seed1'],q['generation_seed2'])==SEEDS,'Selection seeds differ')
    pins=review['archive_metadata_sha256'];require(set(pins)==set(METADATA),'Final archive metadata pins missing')
    require(all(sha(files[name])==pins[name] for name in METADATA),'Archive metadata bytes differ')
    raw,residence,attempt=[decode(files[name]) for name in METADATA]
    require(raw['status']=='CLOSED_SNAPSHOT' and raw['plan_sha256']==bindings['plan_sha256'] and raw['scores_accessed'] is False,'Unclosed raw snapshot')
    require(residence['all_workers_closed'] is True and residence['unknown_unclosed_residence'] is False and not residence['kill_integrity_errors'],'Residence not closed')
    require(_finite(residence['closed_sum_seconds']) and residence['cap_used_as_cost'] is False,'Invalid physical residence measurement')
    require(attempt['all_worker_residences_closed'] is True and attempt['completed_units']==384 and attempt['terminal_conditions']==4224,'Attempt coverage incomplete')
    require(attempt['plan_sha256']==bindings['plan_sha256'] and attempt['raw_manifest_sha256']==pins[METADATA[0]] and attempt['residence_sha256']==pins[METADATA[1]],'Attempt ancestry mismatch')
    require(attempt['scores_computed'] is False and attempt['score_release_authorized'] is False and attempt['automatic_retry'] is False,'Collector authority/scores differ')
    entries=review['units'];require(len(entries)==384 and {e['index'] for e in entries}==set(range(384)),'Reviewed unit index coverage differs')
    by_index={e['index']:e for e in entries};matrices=[];logical=[];overflow=0;required_files=set(METADATA)
    for index,q in enumerate(questions):
        entry=by_index[index];prefix=entry['directory']
        require(re.fullmatch(r'worker_[0-3]/unit_'+f'{index:03d}',prefix) is not None,'Unit path/index binding differs')
        decoded={}
        for filename in ('INPUT.json','MATRIX.json','UNIT_RECEIPT.json'):
            name=prefix+'/'+filename;required_files.add(name)
            require(sha(files[name])==entry['sha256'][filename]==raw['files'][name],'Unit file/archive/review hash mismatch')
            decoded[filename]=decode(files[name])
        inp=decoded['INPUT.json'];unit=inp['unit'];matrix=decoded['MATRIX.json'];receipt=decoded['UNIT_RECEIPT.json']
        require(inp['plan_sha256']==bindings['plan_sha256'] and unit['index']==index and unit['id']==q['id'],'Input unit order/question/plan differs')
        require(prefix==f"worker_{unit['worker']}/unit_{index:03d}",'Unit worker directory differs')
        require(unit['input_sha256']==q['input_sha256'] and unit['edge_index']==q['edge_index'] and (unit['seed1'],unit['seed2'])==SEEDS,'Unit selection input/edge/seed differs')
        payload=inp['payload'];require(set(payload)=={'id','question','context'} and payload['id']=='hotpotqa_'+q['id'],'Collector payload contains labels/wrong ID')
        require(all(isinstance(payload[k],str) and payload[k] for k in ('question','context')),'Complete question/context strings required')
        require(digest(payload)==unit['payload_sha256'],'Collector payload hash differs')
        require(receipt['schema']=='hotpot_unit_receipt_v1' and receipt['index']==index and receipt['worker']==unit['worker'],'Receipt unit differs')
        require(receipt['plan_sha256']==bindings['plan_sha256'] and receipt['matrix_protocol_hash']==bindings['matrix_protocol_hash'],'Receipt source differs')
        require(receipt['matrix_sha256']==entry['sha256']['MATRIX.json'] and receipt['input_file_sha256']==entry['sha256']['INPUT.json'],'Receipt exact file binding differs')
        require(receipt['terminal_keys']==list(ORDER) and receipt['terminal_conditions']==11 and receipt['scores_computed'] is False,'Receipt terminal coverage differs')
        r,c,o=_validate_matrix(matrix,unit,bindings,payload)
        require(receipt['status']==matrix['status'] and receipt['overflow_conditions']==o,'Receipt status/overflow differs')
        require(receipt['total_physical_cost']==matrix['total_physical_cost'],'Physical research ledger differs')
        matrices.append(matrix);logical.append(c);overflow+=o
    require(set(files)==required_files,'Unexpected/missing input files; label vault belongs to separate release')
    require(review['input_overflow_conditions']==overflow and attempt['overflow_conditions']==overflow,'Archive overflow total differs')
    expected_status='COMPLETE_WITH_CAPACITY_FAILURES' if overflow else 'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT'
    require(attempt['status']==expected_status,'Attempt terminal status differs')
    seal=_bound_semantics(questions,matrices,logical,overflow,expected_review_sha256,review['label_vault_sha256'],bindings,pins)
    return BoundArchive(tuple(questions),tuple(matrices),tuple(logical),overflow,expected_review_sha256,
        review['label_vault_sha256'],bindings,dict(pins),seal,_BOUND_TOKEN)


def score_bound_archive(bound,label_vault_bytes,*,run_statistics=True):
    """Pure scoring of an already verified archive. No raw-text answer fallback."""
    require(isinstance(bound,BoundArchive),'Bound archive required')
    require(bound.bindings['evaluation_adapter_sha256']==sha(Path(__file__).read_bytes()),'Evaluation source changed after binding')
    require(bound._validation_token is _BOUND_TOKEN and bound.semantic_sha256==_bound_semantics(bound.questions,bound.matrices,
        bound.logical_costs,bound.overflow_conditions,bound.review_sha256,bound.label_vault_sha256,bound.bindings,bound.physical_archive_pins),
        'Bound archive was not validated or changed before scoring')
    require(sha(label_vault_bytes)==bound.label_vault_sha256,'Released label vault bytes differ')
    labels=decode(label_vault_bytes)
    require(set(labels)=={q['id'] for q in bound.questions} and all(isinstance(v,str) for v in labels.values()),'Exact384 raw-ID string labels required')
    scorer=_helper('hotpot_support');frontier_module=_helper('point_frontier')
    metrics={metric:{'graphs':[[[0.]*4 for _ in range(2)] for _ in range(N)],'no_edge':[0.]*N,'independent12':[0.]*N,'single':[0.]*N} for metric in ('em','f1')}
    statuses=[];format_failures=0
    costs={name:{model:{field:0 for field in FIELDS} for model in ('qwen','mistral')} for name in ('B',*BASELINES)}
    for index,q in enumerate(bound.questions):
        for condition in bound.matrices[index]['conditions']:
            key=condition['key'];name=condition['name'];seed_index=condition['seed_number']-1
            if condition['status']=='INPUT_REJECTED':score={'em':0.,'f1':0.};status='INPUT_REJECTED_ZERO'
            else:
                score=scorer.score_aggregate_node(_records(condition['execution'],name)['aggregate'],labels[q['id']]);status='SCORED_STRICT_AGGREGATE'
                format_failures+=not score['format_ok']
            for metric in metrics:
                if name in GRAPHS:metrics[metric]['graphs'][index][seed_index][GRAPHS.index(name)]=score[metric]
                else:metrics[metric][BASELINES[name]][index]=score[metric]
            statuses.append({'selection_index':index,'condition_key':key,'status':status})
            if seed_index==0 and name in costs:
                source=bound.logical_costs[index][key]
                for model in costs[name]:
                    for field in FIELDS:
                        a,b=costs[name][model][field],source[model][field]
                        costs[name][model][field]=None if a is None or b is None else a+b
    counts={'B':int(sum(row[0][1] for row in metrics['em']['graphs']))}
    counts.update({name:int(sum(metrics['em'][column])) for name,column in BASELINES.items()})
    frontier=frontier_module.review_frontier(counts,costs,archive_integrity_passed=True,
        unique_questions=len(bound.questions),complete_conditions=len(statuses),input_rejections=bound.overflow_conditions,runtime_failures=0)
    audit={'integrity_passed':True,'unique_questions':len(bound.questions),'complete_conditions':len(statuses),
        'runtime_failure_count':0,'input_overflow_conditions':bound.overflow_conditions,'cost_frontier_review':frontier['status']}
    result={'status':'COMPLETE_WITH_CAPACITY_FAILURES' if bound.overflow_conditions else 'COMPLETE_BOUND_SCORING',
        'array_axes':{'rows':'exact selection manifest order','graphs':['A','B','A_MINUS_E','B_MINUS_E'],'seeds':list(SEEDS)},
        'em':metrics['em'],'f1':metrics['f1'],'edge_indices':[q['edge_index'] for q in bound.questions],
        'condition_statuses':statuses,'format_failure_conditions':format_failures,'full_logical_costs_by_model':costs,
        'frontier':frontier,'audit_inputs':audit,'archive_review_sha256':bound.review_sha256,'bindings':bound.bindings,
        'physical_research_ledger':{'archive_metadata_sha256':bound.physical_archive_pins,'recomputed_as_logical_cost':False},
        'private_per_question_arrays':True,'scientific_or_GPU_admission_granted':False}
    if run_statistics:
        result['statistics']=_helper('applicability_statistics').analyze(metrics['em'],metrics['f1'],result['edge_indices'],audit)
    return result


def evaluate_after_release(*,label_loader,run_statistics=True,**archive_arguments):
    """Only release entry: all384/all4224 and external review validated BEFORE loader."""
    bound=bind_archive(**archive_arguments)
    label_bytes=label_loader({'archive_review_sha256':bound.review_sha256,'label_vault_sha256':bound.label_vault_sha256,
        'raw_question_ids':tuple(q['id'] for q in bound.questions)})
    return score_bound_archive(bound,label_bytes,run_statistics=run_statistics)
