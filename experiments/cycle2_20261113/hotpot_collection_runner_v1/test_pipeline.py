"""Synthetic JSON planning/admission/worker/factory paths; never model loading."""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import orchestration_core as core
import plan_tools
import runtime_factory
import storage
import worker
from test_orchestration import BASE,GEOMETRY,fixture,matrix_data,payload

RUNNER_NAMES=('parent.py','worker.py','plan_tools.py','orchestration_core.py','storage.py','production_guard.py','runtime_factory.py')


def put(path,value):
    path.write_bytes(core.canonical(value))
    return storage.file_sha(path)


def input_bundle(directory):
    rows,manifest,length_rows,p=fixture()
    collector=directory/'collector.py'
    collector.write_text('from test_orchestration import payload as collector_payload\n',encoding='utf-8')
    module_paths={}
    for name,file in [('backend','backend.py'),('matrix','matrix_executor.py'),('durable','durable_backend.py')]:
        path=directory/file;path.write_text('# Synthetic source pin only; no models.\n',encoding='utf-8')
        module_paths[name]={'path':str(path),'sha256':storage.file_sha(path)}
    models={name:{'path':'UNUSED_SYNTHETIC_MODEL','identity':{'model':name,'operator':'synthetic'},
                  'tokenizer_identity_sha256':core.graph_digest({'tokenizer':name,'synthetic':True})} for name in ('qwen','mistral')}
    evidence={}
    for name in ('qualification','identity','residence','plan','source_manifest','root_closed_review'):
        path=directory/(name+'_synthetic_evidence.json')
        evidence[name]={'path':path.name,'sha256':put(path,{'synthetic':True,'name':name})}
    acceptance=directory/'ENGINEERING_ACCEPTANCE.json'
    put(acceptance,{'decision':'PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION','matrix_protocol_hash':'b'*64,
         'model_identity_sha256':{name:core.graph_digest(m['identity']) for name,m in models.items()},
         'module_sha256':{name:module_paths[name]['sha256'] for name in ('matrix','durable')},
         'evidence':evidence,'candidate_transaction_id':'synthetic_qualification','candidate_closed_seconds':25.})
    geo=directory/'GEOMETRY.json';put(geo,GEOMETRY)
    runtime={'collector':{'path':str(collector),'sha256':storage.file_sha(collector)},
        'geometry':{'path':str(geo),'sha256':storage.file_sha(geo)},
        'modules':module_paths,'models':models,'python_import_paths':[],
        'graph_config':{'rounds':3,'role_prompts':['role0','role1','role2','role3']},
        'expected_matrix_protocol_hash':'b'*64,
        'engineering_admission':{'path':str(acceptance),'sha256':storage.file_sha(acceptance)},
        'source_files':[{'path':str(BASE/name),'sha256':storage.file_sha(BASE/name)} for name in RUNNER_NAMES]+list(module_paths.values())}
    budget=p['budget'];budget['allocation_authorized']=True
    protocol={'status':'FROZEN','population':{'n_questions':384,'edge_indices':list(core.EDGES)},
         'seeds':{'generation_seed1':core.SEEDS[0],'generation_seed2':core.SEEDS[1],'blind_prefix_pcg64':261113104},
         'execution':{**copy.deepcopy(core.EXPECTED_EXECUTION),'seed1_sequence':list(core.FIRST),'seed2_sequence':list(core.SECOND)},
         'resource':{'blind_benchmark_prefix_units':8,'minimum_forecast_time_margin':.25,'cap_is_authorized':True,
              'actual_cap_seconds':budget['candidate_cap_seconds'],'per_worker_caps_seconds':budget['worker_caps_seconds']},
         'required_pins_before_execution':{key:'d'*64 for key in core.REQUIRED_PINS}}
    paths={name:directory/(name+'.json') for name in ('selection','selection_receipt','lengths','protocol','bindings','budget')}
    paths['questions']=directory/'questions.jsonl'
    paths['questions'].write_bytes(b''.join(core.canonical(row) for row in rows))
    put(paths['selection'],{'schema_version':'1.0','classification':'PRIVATE_NO_LABELS','questions':manifest})
    put(paths['selection_receipt'],{'status':'COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION',
        'selection_manifest_sha256':storage.file_sha(paths['selection']),'questions_jsonl_sha256':storage.file_sha(paths['questions'])})
    put(paths['lengths'],{'status':'TOKENIZER_LENGTHS_COMPLETE_NO_LABELS','labels_read':False,
        'server_runtime_equivalence':True,'questions_jsonl_sha256':storage.file_sha(paths['questions']),
        'selection_manifest_sha256':storage.file_sha(paths['selection']),
        'payload_hash_algorithm':'graph_credit_canonical_utf8_sha256_no_newline',
        'tokenizer_identity_sha256':{name:m['tokenizer_identity_sha256'] for name,m in models.items()},
        'model_identity_sha256':{name:core.graph_digest(m['identity']) for name,m in models.items()},
        'collector_source_sha256':runtime['collector']['sha256'],'rows':length_rows})
    put(paths['protocol'],protocol);put(paths['bindings'],runtime);put(paths['budget'],budget)
    return rows,paths


def admit_synthetic(plan,directory):
    plan=copy.deepcopy(plan);plan['status']='FROZEN_COLLECTION_PLAN'
    path=directory/'FROZEN_PLAN_SYNTHETIC.json';pin=put(path,plan)
    admission=directory/'ADMISSION_SYNTHETIC.json'
    apin=put(admission,{'decision':'APPROVED_FOR_FROZEN_COLLECTION','plan_sha256':pin,
        'attempt_id':plan['attempt_id'],'new_attempt_only':True,'previous_attempt_ids':[],
        'qualification_source_bindings_checked':True,'R1_closed':True,'budget_sha256':core.digest(plan['budget']),
        'output_directory':str(directory/'attempt')})
    return path,pin,admission,apin


class PipelineTests(unittest.TestCase):
    def test_json_prepare_admit_and_worker_crosses_paid_prefix_barrier(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);rows,paths=input_bundle(directory)
            prepared=plan_tools.prepare_plan(**paths,output=directory/'prepared',attempt_id='synthetic_attempt')
            self.assertEqual(prepared['status'],'PREPARED_NOT_FROZEN_NOT_ADMITTED')
            # Never promote the prepare-only artifact in-place.
            args=admit_synthetic(prepared,directory)
            loaded,_=plan_tools.load_execution_plan(*args)
            self.assertEqual(loaded, json.loads(args[0].read_bytes()))
            out=directory/'attempt';out.mkdir()
            units={u['id']:u for u in loaded['schedule']['units']}
            calls=[]
            class Matrix:
                def execute(self,payload,seed1,seed2,edge):
                    u=units[payload['id'].removeprefix('hotpotqa_')]
                    if u['phase']=='remainder':
                        self_outer.assertTrue((out/'RELEASE_REMAINDER.json').exists())
                    calls.append(u['index'])
                    return types.SimpleNamespace(to_dict=lambda:matrix_data(u))
            self_outer=self
            def builder(*args):
                return {'matrix':Matrix(),'collector':payload,'loads':[],'extract_error':lambda e:[]}
            released=[]
            def wait_for_parent(seconds):
                prefix=json.loads((out/'worker_0/PREFIX_COMPLETE.json').read_bytes())
                self.assertEqual(len(prefix['profiles']),4)
                self.assertEqual(len(calls),4)
                forecast_pin=storage.write_new(out/'FORECAST_FROZEN.json',{'status':'FIT'})
                storage.write_new(out/'RELEASE_REMAINDER.json',{'plan_sha256':args[1],
                    'attempt_id':loaded['attempt_id'],'forecast_sha256':forecast_pin})
                released.append(True)
            with patch.object(worker.time,'sleep',wait_for_parent):
                worker.run_worker(loaded,args[1],0,out,float('inf'),runtime_builder=builder)
            self.assertEqual(released,[True])
            self.assertEqual(calls,[u['index'] for u in loaded['schedule']['units'] if u['worker']==0])
            done=json.loads((out/'worker_0/WORKER_COMPLETE.json').read_bytes())
            self.assertEqual((done['units'],done['terminal_conditions']),(192,2112))
            self.assertFalse(done['scores_computed'])

    def test_unadmitted_and_changed_evidence_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);_,paths=input_bundle(directory)
            plan=plan_tools.prepare_plan(**paths,output=directory/'prepared',attempt_id='synthetic_attempt')
            args=admit_synthetic(plan,directory)
            with self.assertRaises(ValueError):
                plan_tools.load_execution_plan(directory/'prepared/PLAN.json',storage.file_sha(directory/'prepared/PLAN.json'),args[2],args[3])
            source=Path(plan['runtime']['engineering_admission']['path'])
            acceptance=json.loads(source.read_bytes())
            (source.parent/acceptance['evidence']['residence']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):plan_tools.load_execution_plan(*args)

    def test_length_payload_or_local_tokenizer_substitution_rejected(self):
        for mutate in (lambda v:v.update(server_runtime_equivalence=False),
                       lambda v:v['rows'][0].update(payload_sha256='e'*64)):
            with tempfile.TemporaryDirectory() as td:
                directory=Path(td);_,paths=input_bundle(directory)
                value=json.loads(paths['lengths'].read_bytes());mutate(value);put(paths['lengths'],value)
                with self.assertRaises(ValueError):
                    plan_tools.prepare_plan(**paths,output=directory/'prepared',attempt_id='synthetic_attempt')

    def test_empty_missing_or_placeholder_protocol_pins_and_operator_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);_,paths=input_bundle(directory)
            protocol=json.loads(paths['protocol'].read_bytes())
            for mutate in (lambda p:p.update(required_pins_before_execution={}),
                           lambda p:p['required_pins_before_execution'].pop('model_files'),
                           lambda p:p['required_pins_before_execution'].update(model_files=None),
                           lambda p:p['execution'].update(worker_max_new_tokens=256),
                           lambda p:p['execution'].update(B_mask=4095)):
                altered=copy.deepcopy(protocol);mutate(altered)
                with self.assertRaises(ValueError):core.validate_protocol(altered,executable=True)

    def test_acceptance_bundle_rejects_relative_escape(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);_,paths=input_bundle(directory)
            runtime=json.loads(paths['bindings'].read_bytes());budget=json.loads(paths['budget'].read_bytes())
            acceptance=Path(runtime['engineering_admission']['path']);value=json.loads(acceptance.read_bytes())
            value['evidence']['residence']['path']='../outside.json'
            runtime['engineering_admission']['sha256']=put(acceptance,value)
            with self.assertRaises(ValueError):plan_tools.validate_engineering(runtime,budget)

    def test_factory_json_roles_use_unchanged_default_and_no_native_logs(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);_,paths=input_bundle(directory)
            plan=plan_tools.prepare_plan(**paths,output=directory/'prepared',attempt_id='synthetic_attempt')
            plan=json.loads(core.canonical(plan));out=directory/'worker';out.mkdir();seen=[]
            @dataclasses.dataclass(frozen=True)
            class Config:
                rounds:int=2
                role_prompts:tuple=('role0','role1','role2','role3')
            self.assertIsInstance(plan['runtime']['graph_config']['role_prompts'],list)
            class HF:
                def __init__(self,path,device,physical_log):
                    seen.append((device,physical_log));name=('qwen','mistral')[len(seen)-1]
                    self.identity=plan['runtime']['models'][name]['identity'];self.runtime={};self.physical_calls=[]
            class Matrix:
                GraphConfig=Config
                def __init__(self,backends,config,route_log):
                    self_outer.assertEqual(config,Config(rounds=3));self_outer.assertIsNone(route_log)
                    self.protocol_hash='b'*64;self.router=types.SimpleNamespace(_selected=types.SimpleNamespace(get=lambda:None))
            self_outer=self
            modules={
              'graph_credit.backend':types.SimpleNamespace(HFBackend=HF),
              'matrix_executor':types.SimpleNamespace(GraphConfig=Config,HotpotApplicabilityMatrixExecutor=Matrix),
              'durable_backend':types.SimpleNamespace(DurableBackend=lambda native,*a,**k:native,extract_durable_error_evidence=lambda e:[])}
            for key,name in [('graph_credit.backend','backend'),('matrix_executor','matrix'),('durable_backend','durable')]:
                modules[key].__file__=plan['runtime']['modules'][name]['path']
            with patch.object(runtime_factory.importlib,'import_module',side_effect=lambda name:modules[name]):
                result=runtime_factory.build_runtime(plan,0,out,{},lambda c:None)
            self.assertEqual(result['matrix'].protocol_hash,'b'*64)
            self.assertEqual(seen,[('cuda:0',None),('cuda:0',None)])

    def test_real_durable_helper_returned_journal_failure_survives_matrix_cause(self):
        helper_path=Path(os.environ.get('MAB2_DURABLE_HELPER',str(BASE.parent/'durable_backend_stage/durable_backend.py')))
        spec=importlib.util.spec_from_file_location('synthetic_pipeline_durable_helper',helper_path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        rows,_,_,plan=fixture();unit=plan['schedule']['units'][0]
        @dataclasses.dataclass
        class Generation:
            text:str='Synthetic paid generation'
            elapsed_seconds:float=.1
        class Native:
            identity={'synthetic':True}
            def __init__(self):self.physical_calls=[]
            def generate(self,*a,**k):
                self.physical_calls.append({'call_id':'synthetic_call','status':'SUCCESS','input_tokens':3,'output_tokens':4})
                return Generation()
        class MatrixError(RuntimeError):
            partial_result=types.SimpleNamespace(to_dict=lambda:matrix_data(unit))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);native=Native()
            durable=helper.DurableBackend(native,'qwen',root/'journal.jsonl')
            original=durable._append
            def append(event):
                if event['event']=='RETURNED':raise OSError('Synthetic returned-write failure')
                return original(event)
            durable._append=append
            class Matrix:
                def execute(self,*a):
                    try:durable.generate([{'role':'user','content':'Artificial prompt'}],1,8,{})
                    except Exception as error:raise MatrixError('Frozen wrapper boundary') from error
            runtime={'collector':payload,'matrix':Matrix(),'extract_error':helper.extract_durable_error_evidence}
            with self.assertRaises(MatrixError):worker.execute_unit(runtime,rows[unit['index']],unit,root/'unit','a'*64,{})
            evidence=json.loads((root/'unit/DURABLE_ERROR_EVIDENCE.json').read_bytes())
            self.assertEqual(evidence[0]['generation']['text'],'Synthetic paid generation')
            self.assertEqual(evidence[0]['native_events'],native.physical_calls)
            self.assertEqual(len(helper.read_journal(root/'journal.jsonl')['pending_request_ids']),1)
            self.assertFalse((root/'unit/UNIT_RECEIPT.json').exists())


if __name__=='__main__':unittest.main()
