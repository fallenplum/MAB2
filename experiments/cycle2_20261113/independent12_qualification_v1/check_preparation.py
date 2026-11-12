"""CPU preparation checks; no supervisor, model, guard timer or GPU launched."""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('prepared_qualification',ROOT/'qualify_independent12.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def function_text(text,name):
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)


def fixture(task,repeat,protocol):
    """Hand-built synthetic ledger fixture, not a model execution."""
    order=[f'r{s}:w{r}' for r in range(4) for s in range(1,4)]+['aggregate']
    qid='synthetic_static_fixture_'+str(task);qfp=digest({'id':qid,'question':'synthetic fixture'})
    phash=digest({'fixture_control':'independent12'});seed=protocol['seed'];slots=[]
    for index,node in enumerate(order):
        role=None if node=='aggregate' else int(node[-1]);sample=None if role is None else int(node[1])
        model='mistral' if role is None else protocol['model_route']['workers'][role]
        node_seed=int(digest([qid,seed,node])[:16],16)%(2**31-1)
        messages=[{'role':'user','content':'synthetic role '+str(role)}]
        cap=256 if role is None else 128
        fp=digest({'question':qfp,'node':node,'messages':messages,'seed':node_seed,'protocol':phash,'max_new_tokens':cap})
        parsed={'answer':'fixture','evidence':'','uncertainty':''}
        wire=json.dumps(parsed,sort_keys=True,separators=(',',':'))
        gen={'text':wire,'input_tokens':20,'output_tokens':10,'elapsed_seconds':.1+repeat*.01,
             'gpu_seconds':.01+repeat*.001,'gpu_peak_allocated_bytes':1000+repeat,
             'finish_reason':'eos','call_id':f'fixture-{task}-{repeat}-{node}'}
        received=120 if node=='aggregate' else 0
        rec={'node':node,'input_fingerprint':fp,'seed':node_seed,'messages':messages,'generation':copy.deepcopy(gen),
             'wire_message':wire,'parsed':parsed,'message_metadata':{'format_ok':True,'type_normalized':False,
             'format_error':None,'message_trimmed':False,'message_tokens':10},
             'received_message_tokens':received,'self_state_tokens':0}
        rec['output_digest']=digest(rec);rec.update(reused=False,source_execution_id=None)
        slots.append({'index':index,'node':node,'role':role,'sample':sample,'model':model,'seed':node_seed,
                      'status':'SUCCESS','attempted_backend_request':True,'messages':messages,
                      'input_fingerprint':fp,'received_message_tokens':received,'self_state_tokens':0,
                      'generation':gen,'record':rec,'error_type':None,'route_events':[],'physical_events':[]})
    cost={'calls':13,'returned_generations':13,'unattempted_calls':0,'failed_requests_without_physical_record':0,
          'service_complete':True,'cost_fields_complete':True,'resident_GPU_hours':None,
          'input_tokens':260,'output_tokens':130,'gpu_seconds':sum(s['generation']['gpu_seconds'] for s in slots),
          'model_elapsed_seconds':sum(s['generation']['elapsed_seconds'] for s in slots),
          'peak_allocated_bytes':1000+repeat,'received_message_tokens':120,'self_state_tokens':0}
    cost['known_cost_components']={key:cost[key] for key in ('input_tokens','output_tokens','gpu_seconds','model_elapsed_seconds','peak_allocated_bytes')}
    return {'status':'COMPLETE','question_id':qid,'question_fingerprint':qfp,'protocol_hash':phash,'seed':seed,
            'slots':slots,'answer':'fixture','cache_reuse_count':0,'physical_cost':cost,'logical_service_cost':copy.deepcopy(cost)}


def closure_inputs(results,protocol):
    backends={name:SimpleNamespace(identity={'fixture_model':name},physical_calls=[]) for name in ('qwen','mistral')}
    routes=[]
    for result in results:
        for slot in result['slots']:
            gen=slot['generation'];model=slot['model'];cap=256 if slot['node']=='aggregate' else 128
            physical={key:gen[key] for key in ('call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason')}
            physical.update(status='SUCCESS',seed=slot['seed'],prompt_sha256=digest(slot['messages']),generation_config={'max_new_tokens':cap})
            backends[model].physical_calls.append(physical)
            routes.append({'call_id':gen['call_id'],'status':'SUCCESS','model':model,'node':slot['node'],
                'model_identity_sha256':digest(backends[model].identity),'seed':slot['seed'],
                'prompt_sha256':digest(slot['messages']),'input_tokens':gen['input_tokens'],
                'output_tokens':gen['output_tokens'],'max_new_tokens':cap})
    return SimpleNamespace(route_events=routes),backends


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-source',type=Path,default=ROOT/'deployment_sources'/q.CONTROL_DIR/'independent12_executor.py')
    args=parser.parse_args()
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append({'name':name,'pass':True})
    def rejects(name,fn):
        try:fn()
        except (AssertionError,KeyError,ValueError,RuntimeError):check(name,True)
        else:raise AssertionError(name+' did not reject')
    protocol=json.loads((ROOT/'PROTOCOL.json').read_bytes())
    old_protocol=json.loads((ROOT/'reference_v2/PROTOCOL.json').read_bytes())
    original=(ROOT/'reference_v2/qualify_routed_executor.py').read_text(encoding='utf-8')
    current=(ROOT/'qualify_independent12.py').read_text(encoding='utf-8')
    for path in ROOT.glob('*.py'):
        compile(path.read_bytes(),path.name,'exec')
    check('all_preparation_python_sources_compile',True)
    old_parent=function_text(original,'parent');new_parent=function_text(current,'parent')
    expected=old_parent.replace('C2_HETERO_ENGINEERING_QUALIFICATION',q.RESERVATION)
    check('parent_exact_text_except_reservation_id',expected==new_parent)
    old_main=function_text(original,'main');new_main=function_text(current,'main')
    check('main_exact_text_except_launch_plan_validation',new_main.replace('    validate_launch_plan(plan)\n','')==old_main)
    for name in ('sha','write','pin_sources'):
        check(name+'_helper_exact_text_reuse',function_text(original,name)==function_text(current,name))
    diffs=list(difflib.unified_diff((old_parent+'\n\n'+old_main+'\n').splitlines(True),
                                  (new_parent+'\n\n'+new_main+'\n').splitlines(True),
                                  fromfile='qualified_v2_parent_main',tofile='prepared_independent12_parent_main'))
    (ROOT/'SUPERVISOR_DIFF.patch').write_text(''.join(diffs),encoding='utf-8')
    check('guard_exact_original_bytes',q.sha(ROOT/'gpu_job_guard.py')==q.GUARD_PIN==protocol['guard_source_sha256'])
    check('original_two_synthetic_inputs_unchanged',protocol['synthetic_inputs']==old_protocol['synthetic_inputs'])
    check('original_graph_config_and_model_route_seed_unchanged',protocol['graph_config_used_as_sampling_config']==old_protocol['graph'] and protocol['model_route']==old_protocol['model_route'] and protocol['seed']==old_protocol['seed'])
    repetitions=len(protocol['synthetic_inputs'])*protocol['fresh_repeats_per_question']
    check('exact_52_call_matrix',repetitions*13==52==protocol['expected_complete_calls'])
    check('exact_24_Qwen_28_Mistral',protocol['expected_complete_calls_by_model']=={'qwen':repetitions*6,'mistral':repetitions*7})
    check('control_pin_consistent',protocol['control_source_sha256']==q.CONTROL_PIN)
    control_path=args.control_source
    check('sealed_control_dependency_matches_current_file',q.sha(control_path)==q.CONTROL_PIN)
    deployment=json.loads((ROOT/'DEPLOYMENT_SOURCE_MANIFEST.json').read_bytes())
    check('deployment_manifest_contains_exact_eight_sources',len(deployment['files'])==8 and {r['path'] for r in deployment['files']}==q.SOURCE_NAMES)
    for row in deployment['files']:
        source=ROOT/'deployment_sources'/row['path']
        check('deployment_source_pin_'+row['path'],source.stat().st_size==row['bytes'] and q.sha(source)==row['sha256'])
    for name in ('qualify_independent12.py','gpu_job_guard.py','PROTOCOL.json'):
        check('deployment_qualification_matches_reviewed_'+name,(ROOT/name).read_bytes()==(ROOT/'deployment_sources'/q.QUAL_DIR/name).read_bytes())
    template=json.loads((ROOT/'PRIVATE_PLAN.template.json').read_bytes())
    with patch.object(q.subprocess,'Popen',side_effect=AssertionError('Popen forbidden during static checks')), \
         patch.object(q.subprocess,'check_output',side_effect=AssertionError('external processes forbidden during static checks')):
        rejects('unapproved_template_is_fail_closed',lambda:q.validate_launch_plan(template))
        with patch.object(q,'parent',side_effect=AssertionError('parent must not be reached')) as parent_mock, \
             patch.object(sys,'argv',['qualify_independent12.py','--plan',str(ROOT/'PRIVATE_PLAN.template.json')]):
            rejects('normal_main_rejects_template_before_supervisor',q.main)
            check('template_does_not_reach_parent',not parent_mock.called)
    optimized_namespace={'__name__':'qualification_optimized_static_probe','__file__':str(ROOT/'qualify_independent12.py')}
    exec(compile(current,'qualify_independent12.py','exec',optimize=2),optimized_namespace)
    rejects('optimized_python_cannot_remove_admission_gate',lambda:optimized_namespace['validate_launch_plan'](template))
    check('no_template_reservation_or_launch',template['launch_authorized'] is False and template['budget_reservation_confirmed'] is False)
    results=[fixture(task,repeat,protocol) for task in range(2) for repeat in (1,2)]
    for result in results:q.verify_execution(result,protocol,digest)
    check('four_handmade_13_slot_ledgers_pass_new_auditor',True)
    equalities=q.compare_fresh(results[0],results[1],protocol)+q.compare_fresh(results[2],results[3],protocol)
    check('fresh_semantics_ignore_call_id_time_and_own_digest',equalities==572)
    for field in ('text','input_tokens','output_tokens','finish_reason'):
        bad=copy.deepcopy(results[1]);bad['slots'][0]['generation'][field]='altered'
        rejects('fresh_rejects_generation_'+field,lambda bad=bad:q.compare_fresh(results[0],bad,protocol))
    for field in ('messages','input_fingerprint','seed'):
        bad=copy.deepcopy(results[1]);bad['slots'][0][field]='altered'
        rejects('fresh_rejects_slot_'+field,lambda bad=bad:q.compare_fresh(results[0],bad,protocol))
    for field in ('parsed','wire_message'):
        bad=copy.deepcopy(results[1]);bad['slots'][0]['record'][field]='altered'
        rejects('fresh_rejects_record_'+field,lambda bad=bad:q.compare_fresh(results[0],bad,protocol))
    missing=copy.deepcopy(results[0]);missing['slots'].pop()
    rejects('missing_worker_or_aggregate_slot_rejected',lambda:q.verify_execution(missing,protocol,digest))
    bad=copy.deepcopy(results[0]);bad['physical_cost']['input_tokens']+=1;bad['logical_service_cost']=copy.deepcopy(bad['physical_cost'])
    rejects('internally_matching_but_wrong_execution_cost_rejected',lambda:q.verify_execution(bad,protocol,digest))
    router,backends=closure_inputs(results,protocol)
    costs,formats=q.verify_call_closure(results,router,backends,protocol,digest)
    check('52_route_physical_and_result_calls_close',costs['calls']==52 and len(formats)==52)
    duplicate=copy.deepcopy(backends);duplicate['qwen'].physical_calls.append(copy.deepcopy(duplicate['qwen'].physical_calls[0]))
    rejects('duplicate_physical_event_rejected',lambda:q.verify_call_closure(results,router,duplicate,protocol,digest))
    bad_router=copy.deepcopy(router);bad_router.route_events[0]['model']='mistral'
    rejects('wrong_model_route_rejected',lambda:q.verify_call_closure(results,bad_router,backends,protocol,digest))
    check('no_model_runtime_imported',not any(name in sys.modules for name in ('torch','transformers')))
    receipt={'status':'PASS_CPU_PREPARATION_CHECKS_NOT_ENGINEERING_QUALIFIED','checks':checks,
             'source_pins':{name:q.sha(ROOT/name) for name in ('qualify_independent12.py','gpu_job_guard.py','PROTOCOL.json','PRIVATE_PLAN.template.json','check_preparation.py')},
             'control_dependency_sha256':q.CONTROL_PIN,'fresh_field_equalities_fixture_only':equalities,
             'guard_tests_repeated':False,'supervisor_launched':False,'budget_reserved':False,
             'new_GPU_calls':0,'model_loads':0,'benchmark_or_R1_reads':0,'scientific_admission':False,
             'limitations':['Static source equivalence and synthetic ledger fixtures only','No new Linux guard exercise','No real tokenizer or model execution','Root review, source freeze and new budget reservation still required']}
    (ROOT/'STATIC_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'checks':len(checks),'guard_tests_repeated':False,'budget_reserved':False,'new_GPU_calls':0}))


if __name__=='__main__':main()
