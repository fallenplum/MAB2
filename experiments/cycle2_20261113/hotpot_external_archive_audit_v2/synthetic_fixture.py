"""Artificial 384-unit full archive builder using CPU-only frozen operators."""
from pathlib import Path
import copy
import dataclasses
import hashlib
import sys
import threading
import types
import archive_audit as a

BASE=Path(__file__).resolve().parent/'fixture_dependencies'
MATRIX=BASE/'hotpot_applicability_executor_stage'
sys.dont_write_bytecode=True
sys.path[:0]=[str(MATRIX/'frozen_dependencies/src'),str(MATRIX/'frozen_dependencies/router'),str(MATRIX/'frozen_dependencies/control'),str(MATRIX),str(BASE/'durable_backend_stage'),str(BASE/'hotpot_collection_runner_stage')]
from graph_credit.executor import Generation,GraphConfig
from matrix_executor import HotpotApplicabilityMatrixExecutor
from durable_backend import DurableBackend
import orchestration_core as core

def encode(x):return (a.canonical(x)+'\n').encode()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Native:
    def __init__(self,m,w):
        self.m,self.w=m,w;self.identity={'model':m,'synthetic_only':True};self.physical_calls=[]
        self.runtime={'device':'CPU_ARTIFICIAL','worker':w};self.position_limit=32768
        self.provenance={'backend_identity_sha256':a.digest(self.identity),'chat_template_reexecuted':True,'synthetic_only':True}
    def encode(self,text):return list(text.encode())
    def decode(self,tokens):return bytes(tokens).decode()
    def chat_count(self,messages):
        if hasattr(self,'overflow_predicate') and self.overflow_predicate(messages):return 8193
        return len(a.canonical(messages).encode())
    def generate(self,messages,seed,max_new_tokens,config):
        if self.chat_count(messages)>8192:raise ValueError('Prompt has 8193 tokens; frozen cap is 8192')
        text=a.canonical({'answer':'x','evidence':'','uncertainty':''});n=len(self.physical_calls)
        result=Generation(text,self.chat_count(messages),len(self.encode(text)),.01,.005,1024,'eos',f'worker{self.w}:{self.m}:{n}')
        settings={'max_new_tokens':max_new_tokens,'min_new_tokens':0,'do_sample':True,'temperature':.7,'top_p':.9,'top_k':0,
            'num_beams':1,'num_return_sequences':1,'repetition_penalty':1.,'length_penalty':1.,'use_cache':True,
            'return_dict_in_generate':False,'output_scores':False}
        self.physical_calls.append({'call_id':result.call_id,'status':'SUCCESS','seed':seed,'started_unix':100.+n*.01,
            'input_tokens':result.input_tokens,'output_tokens':result.output_tokens,'elapsed_seconds':result.elapsed_seconds,
            'gpu_seconds':result.gpu_seconds,'gpu_peak_allocated_bytes':1024,'finish_reason':'eos','prompt_sha256':a.digest(messages),
            'generation_config':settings,'runtime':self.runtime})
        return result

class MemoryDurable(DurableBackend):
    def __init__(self,native,m,context):
        self.underlying=native;self.model_name=m;self.context_provider=context;self.before_request=None
        self._lock=threading.Lock();self._poisoned=False
        self.events=[{'event':'HEADER','schema':'durable_backend_journal_v1','model':m,'backend_identity':native.identity,
            'helper_sha256':file_sha(BASE/'durable_backend_stage/durable_backend.py')}]
    def _append(self,event):self.events.append(copy.deepcopy(event))

def fixture():
    source_paths=[BASE/'hotpot_collection_runner_stage'/n for n in ('parent.py','worker.py','runtime_factory.py','plan_tools.py','orchestration_core.py','storage.py','production_guard.py')]
    source_paths += [MATRIX/'frozen_dependencies/src/graph_credit'/n for n in ('executor.py','backend.py')]
    source_paths += [MATRIX/'frozen_dependencies/router/routed_executor.py',MATRIX/'frozen_dependencies/control/independent12_executor.py',
                     MATRIX/'matrix_executor.py',MATRIX/'matrix_geometry.py',BASE/'durable_backend_stage/durable_backend.py',
                     BASE/'hotpot_evaluation_stage/frozen_dependencies/hotpot_support.py']
    sources={str(path):path.read_bytes() for path in source_paths};source_rows=[{'path':path,'sha256':a.sha(raw)} for path,raw in sources.items()]
    model_objs={m:Native(m,0) for m in ('qwen','mistral')};identity_matrix=HotpotApplicabilityMatrixExecutor(model_objs,GraphConfig(rounds=3))
    runtime={'modules':{name:{'path':str(path),'sha256':file_sha(path)} for name,path in
             [('backend',MATRIX/'frozen_dependencies/src/graph_credit/backend.py'),('matrix',MATRIX/'matrix_executor.py'),('durable',BASE/'durable_backend_stage/durable_backend.py')]},
        'models':{m:{'identity':b.identity,'tokenizer_identity_sha256':a.digest(['synthetic_tokenizer',m])} for m,b in model_objs.items()},
        'source_files':source_rows,'graph_config':dataclasses.asdict(GraphConfig(rounds=3)),
        'expected_matrix_protocol_hash':identity_matrix.protocol_hash}
    runtime=a.decode(encode(runtime))
    collector_path=BASE/'hotpot_evaluation_stage/frozen_dependencies/hotpot_support.py'
    runtime['collector']={'path':str(collector_path),'sha256':file_sha(collector_path)}
    eng={'decision':'PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION','matrix_protocol_hash':identity_matrix.protocol_hash,
         'model_identity_sha256':{m:a.digest(b.identity) for m,b in model_objs.items()},
         'module_sha256':{m:runtime['modules'][m]['sha256'] for m in ('matrix','durable')},
         'candidate_transaction_id':'synthetic_qual','candidate_closed_seconds':25.}
    auxiliary={'ENGINEERING.json':encode(eng),'SOURCES.json':encode({'source_files':source_rows}),
        'SNAPSHOT.json':(BASE/'hotpot_collection_runner_stage/ROOT_SOURCE_FREEZE.json').read_bytes()}
    runtime['engineering_admission']={'path':'ENGINEERING.json','sha256':a.sha(auxiliary['ENGINEERING.json'])}
    budget={'candidate_cap_seconds':200025.,'global_available_seconds_for_attempt':200000.,'worker_caps_seconds':[100000.,100000.],
        'workers':[{'gpu':0},{'gpu':1}],'disk_min_free_bytes':1,'release_guard_seconds':10.,'allocation_authorized':True,
        'prior_candidate_transactions':[{'transaction_id':'synthetic_qual','seconds':25.,'already_in_global_ledger':True}]}
    protocol={'status':'FROZEN','population':{'n_questions':384,'edge_indices':[1,5,7,9]},'execution':dict(core.EXPECTED_EXECUTION),
        'required_pins_before_execution':{key:'d'*64 for key in a.REQUIRED_PINS},
        'resource':{'cap_is_authorized':True,'actual_cap_seconds':200025.,'per_worker_caps_seconds':[100000.,100000.]}}
    rows=[{'id':f'artificial_{i:03d}','question':f'Artificial marker {i}?','context':{'title':['Artificial'],'sentences':[['A generated synthetic sentence.']]},'type':'bridge','level':'synthetic'} for i in range(384)]
    selected=[{'id':r['id'],'input_sha256':a.newline_digest(r),'content_sha256':a.digest(['content',i]),'question_group_sha256':a.digest(['group',i]),'edge_index':a.PROBES[i//96],
               'generation_seed1':a.SEEDS[0],'generation_seed2':a.SEEDS[1]} for i,r in enumerate(rows)]
    protocol_raw=encode(protocol);selector='5edc8dc62b349be49c2c939c9c3f93ba1a8795b3c6fc7c951aec3ab9cd3d8bfd'
    selection={'classification':'PRIVATE_NO_LABELS','pins':{'protocol_sha256':a.sha(protocol_raw),'selector_sha256':selector},'questions':selected}
    lengths=[]
    for row in rows:
        payload=a.collector(row);full=payload['question']+'\n\nSupplied context:\n'+payload['context']
        lengths.append({'id':row['id'],'payload_sha256':a.digest(payload),**{m+'_tokens':len(b.encode(full)) for m,b in model_objs.items()}})
    pinned={'protocol':protocol_raw,'selection':encode(selection),'questions':b''.join(encode(r) for r in rows),'budget':encode(budget),'bindings':encode(runtime)}
    pinned['selection_receipt']=encode({'status':'COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION','selection_manifest_sha256':a.sha(pinned['selection']),'questions_jsonl_sha256':a.sha(pinned['questions'])})
    pinned['lengths']=encode({'status':'TOKENIZER_LENGTHS_COMPLETE_NO_LABELS','labels_read':False,'server_runtime_equivalence':True,
        'payload_hash_algorithm':'graph_credit_canonical_utf8_sha256_no_newline','questions_jsonl_sha256':a.sha(pinned['questions']),'selection_manifest_sha256':a.sha(pinned['selection']),
        'collector_source_sha256':runtime['collector']['sha256'],
        'model_identity_sha256':eng['model_identity_sha256'],'tokenizer_identity_sha256':{m:x['tokenizer_identity_sha256'] for m,x in runtime['models'].items()},'rows':lengths})
    schedule=core.build_schedule(selected,lengths,2)
    keys={}
    for e in a.PROBES:
        keys[str(e)]=[]
        for name in a.FIRST+a.SECOND:
            arch='independent12' if name=='INDEPENDENT12' else 'single_m' if name=='SINGLE_M' else 'graph'
            base={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(name)
            affected=a.closure(a.mask(base,e) if base else None,a.mask(name,e)) if arch=='graph' else set(a.ordering(name))
            for node in a.ordering(name):
                if node in affected:keys[str(e)].append('|'.join((arch,a.model(node),node)))
    plan={'schema':'hotpot_collection_plan_v1','status':'FROZEN_COLLECTION_PLAN','attempt_id':'ARTIFICIAL_FULL_ARCHIVE',
        'paths':{k:k for k in pinned},'input_sha256':{k:a.sha(v) for k,v in pinned.items()},'runtime':runtime,'sources':source_rows,
        'budget':budget,'schedule':schedule,'request_keys_by_edge':keys,'forecast_margin':.25}
    plan_raw=encode(plan);plan_sha=a.sha(plan_raw)
    admission={'decision':'APPROVED_FOR_FROZEN_COLLECTION','plan_sha256':plan_sha,'attempt_id':plan['attempt_id'],'new_attempt_only':True,
        'previous_attempt_ids':[],'R1_closed':True,'qualification_source_bindings_checked':True,'budget_sha256':a.newline_digest(budget)}
    admission_raw=encode(admission)
    evaluator=BASE/'hotpot_evaluation_stage'
    bindings={'scientific_protocol_sha256':a.sha(protocol_raw),'selection_manifest_sha256':a.sha(pinned['selection']),
        'plan_sha256':plan_sha,'source_manifest_sha256':a.sha(auxiliary['SOURCES.json']),'matrix_protocol_hash':identity_matrix.protocol_hash,
        'graph_protocol_hash':identity_matrix.graph.protocol_hash,'independent12_protocol_hash':identity_matrix.independent.protocol_hash,
        'single_protocol_hash':identity_matrix.single.protocol_hash,'selector_sha256':selector,'evaluation_adapter_sha256':file_sha(evaluator/'evaluation_adapter.py'),
        'scorer_sha256':file_sha(evaluator/'frozen_dependencies/hotpot_support.py'),'statistics_sha256':file_sha(evaluator/'frozen_dependencies/applicability_statistics.py'),
        'frontier_sha256':file_sha(evaluator/'frozen_dependencies/point_frontier.py')}
    prior_receipts={'synthetic_qual':encode({'status':'CLOSED','resident_seconds':25.})}
    registry={'status':'CLOSED_REGISTRY_EXTERNAL_ROOT_REVIEW','all_known_attempts_enumerated':True,'authorized_global_cap_seconds':576000.,
        'candidate_prior_transaction_ids':['synthetic_qual'],'transactions':[{'transaction_id':'synthetic_qual','seconds':25.,'receipt_sha256':a.sha(prior_receipts['synthetic_qual']),
            'identity_binding':'ROOT_REGISTRY_BINDING_LEGACY_RECEIPT_WITHOUT_ID'}]}
    auxiliary['REGISTRY.json']=encode(registry)
    contract={'plan_sha256':plan_sha,'admission_sha256':a.sha(admission_raw),'source_manifest_key':'SOURCES.json','source_manifest_sha256':bindings['source_manifest_sha256'],
        'runner_snapshot_key':'SNAPSHOT.json','prior_ledger_key':'REGISTRY.json','prior_ledger_sha256':a.sha(auxiliary['REGISTRY.json']),
        'bindings':bindings,'label_vault_sha256':a.digest('HASH_ONLY_NO_LABELS')}
    archive={'ATTEMPT_STARTED.json':encode({'attempt_id':plan['attempt_id'],'plan_sha256':plan_sha,'admission_sha256':a.sha(admission_raw),'parent_pid':100,'scores_accessed':False})}
    receipts={};prefix_profiles=[]
    for w in range(2):
        natives={m:Native(m,w) for m in ('qwen','mistral')};context={'attempt_id':plan['attempt_id'],'plan_sha256':plan_sha,'worker_id':w}
        reference=[None]
        wrappers={m:MemoryDurable(native,m,lambda:{**context,'node':reference[0].router._selected.get()}) for m,native in natives.items()}
        matrix=HotpotApplicabilityMatrixExecutor(wrappers,GraphConfig(rounds=3));reference[0]=matrix
        loads=[{'model':m,'load_wall_seconds':1.} for m in natives]
        for m,native in natives.items():
            archive[f'worker_{w}/MODEL_LOAD_STARTED_{m}.json']=encode({'model':m,'worker':w,'monotonic':100.+(0 if m=='qwen' else 1)})
            archive[f'worker_{w}/MODEL_LOAD_COMPLETE_{m}.json']=encode({'model':m,'worker':w,'wall_seconds':1.,'identity':native.identity,'runtime_private':native.runtime})
        archive[f'worker_{w}/RUNTIME_IDENTITY.json']=encode({'matrix_protocol_hash':matrix.protocol_hash,'loads':loads,'durable_helper_sha256':runtime['modules']['durable']['sha256'],'model_source_binding_checked':True})
        assigned=[u for u in schedule['units'] if u['worker']==w]
        for unit in assigned:
            i=unit['index'];payload=a.collector(rows[i]);context.update(unit_index=i,unit_payload_sha256=unit['payload_sha256'])
            data=matrix.execute(payload,*a.SEEDS,unit['edge_index']).to_dict();directory=f'worker_{w}/unit_{i:03d}'
            archive[directory+'/INPUT.json']=encode({'plan_sha256':plan_sha,'unit':unit,'payload':payload});archive[directory+'/MATRIX.json']=encode(data)
            profile=core.cost_profile(data,unit,1.);receipt={'schema':'hotpot_unit_receipt_v1','index':i,'worker':w,'plan_sha256':plan_sha,
                'status':data['status'],'matrix_protocol_hash':matrix.protocol_hash,'matrix_sha256':a.sha(archive[directory+'/MATRIX.json']),
                'input_file_sha256':a.sha(archive[directory+'/INPUT.json']),'terminal_keys':list(a.ORDER),'terminal_conditions':11,
                'overflow_conditions':0,'cost_fields_complete':True,'known_physical_cost_components':data['known_physical_cost_components'],
                'total_physical_cost':data['total_physical_cost'],'cost_profile':profile,'scores_computed':False}
            archive[directory+'/UNIT_RECEIPT.json']=encode(receipt);receipts[i]=receipt
        profiles=[receipts[u['index']]['cost_profile'] for u in assigned if u['phase']=='prefix'];prefix_profiles.extend(profiles)
        archive[f'worker_{w}/PREFIX_COMPLETE.json']=encode({'worker':w,'plan_sha256':plan_sha,'profiles':profiles,'model_loads':loads})
        archive[f'worker_{w}/WORKER_COMPLETE.json']=encode({'status':'COMPLETE_COLLECTOR_ONLY','plan_sha256':plan_sha,'worker':w,'units':len(assigned),'terminal_conditions':11*len(assigned),'overflow_conditions':0,'scores_computed':False})
        for m,wrapper in wrappers.items():archive[f'worker_{w}/{m}_raw_requests.jsonl']=b''.join(encode(event) for event in wrapper.events)
    forecast=core.forecast(schedule,prefix_profiles,keys,[10.,10.],budget,.25)
    forecast.update(plan_sha256=plan_sha,attempt_id=plan['attempt_id'],prefix_receipt_sha256=[a.sha(archive[f'worker_{w}/PREFIX_COMPLETE.json']) for w in range(2)])
    archive['FORECAST_FROZEN.json']=encode(forecast);archive['RELEASE_REMAINDER.json']=encode({'plan_sha256':plan_sha,'attempt_id':plan['attempt_id'],
        'forecast_sha256':a.sha(archive['FORECAST_FROZEN.json']),'release_recheck_resident_seconds':[11.,11.],'release_recheck_fits':True,'scores_accessed':False})
    archive['RAW_OUTPUT_MANIFEST.json']=encode({'status':'CLOSED_SNAPSHOT','plan_sha256':plan_sha,'files':{k:a.sha(v) for k,v in archive.items()},'scores_accessed':False})
    archive['PARENT_RESIDENCE.json']=encode({'transaction_id':plan['attempt_id'],'worker_started_monotonic':{'0':100.,'1':100.},
        'worker_closed_monotonic':{'0':1100.,'1':1100.},'observed_seconds_by_worker':[1000.,1000.],'closed_sum_seconds':2000.,
        'observed_residence_lower_bound_seconds':2000.,'all_workers_closed':True,'includes_model_loading_and_barrier_idle':True,
        'cap_used_as_cost':False,'unknown_unclosed_residence':False,'kill_integrity_errors':[]})
    archive['ATTEMPT_RESULT.json']=encode({'status':'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT','attempt_id':plan['attempt_id'],'plan_sha256':plan_sha,
        'completed_units':384,'terminal_conditions':4224,'overflow_conditions':0,'all_worker_residences_closed':True,'scores_computed':False,
        'score_release_authorized':False,'automatic_retry':False,'raw_manifest_sha256':a.sha(archive['RAW_OUTPUT_MANIFEST.json']),
        'residence_sha256':a.sha(archive['PARENT_RESIDENCE.json'])})
    return {'plan_bytes':plan_raw,'admission_bytes':admission_raw,'contract_bytes':encode(contract),'pinned_inputs':pinned,
        'auxiliary':auxiliary,'sources':sources,'archive':archive,'tokenizers':model_objs,'prior_receipts':prior_receipts}
