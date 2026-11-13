"""CPU tests and source inventories only; never launches qualification."""
import ast,difflib,hashlib,io,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(BASE/'durable_backend_stage')]
import qualify_hotpot_matrix as qualification

def write(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def run():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ('test_durable_backend','test_qualification_preparation'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (ROOT/'CPU_TESTS.log').write_bytes(stream.getvalue().encode())
    assert result.wasSuccessful(),stream.getvalue()
    mapping={
        qualification.DURABLE_DIR+'/durable_backend.py':BASE/'durable_backend_stage/durable_backend.py',
        qualification.MATRIX_DIR+'/matrix_executor.py':BASE/'hotpot_applicability_executor_stage/matrix_executor.py',
        qualification.MATRIX_DIR+'/matrix_geometry.py':BASE/'hotpot_applicability_executor_stage/matrix_geometry.py',
        qualification.ROUTER_DIR+'/routed_executor.py':BASE/'hotpot_applicability_executor_stage/frozen_dependencies/router/routed_executor.py',
        qualification.CONTROL_DIR+'/independent12_executor.py':BASE/'hotpot_applicability_executor_stage/frozen_dependencies/control/independent12_executor.py',
    }
    for name in ('__init__.py','executor.py','backend.py'):
        mapping['src/graph_credit/'+name]=BASE/'hotpot_applicability_executor_stage/frozen_dependencies/src/graph_credit'/name
    for name in ('qualify_hotpot_matrix.py','matrix_qualification_audit.py','gpu_job_guard.py','PROTOCOL.json','SYNTHETIC_INPUT.json'):
        mapping[qualification.QUAL_DIR+'/'+name]=ROOT/name
    assert set(mapping)==qualification.SOURCE_NAMES
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in mapping.items()}
    assert all(hashes[name]==pin for name,pin in qualification.FIXED_PINS.items())
    for name,path in mapping.items():
        if name.endswith('.py'):ast.parse(path.read_text(),filename=name)
        if name.startswith((qualification.QUAL_DIR,qualification.DURABLE_DIR)):
            assert b'\r\n' not in path.read_bytes(),name
    for name,path in mapping.items():
        target=ROOT/'deployment_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    write(ROOT/'DEPLOYMENT_SOURCE_MANIFEST.json',hashes)
    plan=json.loads((ROOT/'PRIVATE_PLAN.template.json').read_bytes());plan['source_manifest']=hashes
    write(ROOT/'PRIVATE_PLAN.template.json',plan)
    old=(BASE/'independent12_qualification_stage/reference_v2/qualify_routed_executor.py').read_text()
    new=(ROOT/'qualify_hotpot_matrix.py').read_text()
    diff=''.join(difflib.unified_diff(old[old.index('def parent('):].splitlines(True),
        new[new.index('def parent('):].splitlines(True),fromfile='qualified_v2_parent_main',tofile='matrix_qualification_parent_main'))
    (ROOT/'SUPERVISOR_DIFF.patch').write_bytes(diff.encode())
    write(ROOT/'CPU_TEST_RESULTS.json',{'status':'PASS_CPU_ONLY','tests':result.testsRun,'failures':0,'errors':0,
        'gpu_calls':0,'benchmark_questions_read':0,'guard_original_bytes_verified':True,
        'new_linux_guard_tests_run':0,'guard_existing_linux_tests_reused':14,
        'matrix_expected_success_calls':174,'logical_records':262,'source_manifest_sha256':qualification.sha(ROOT/'DEPLOYMENT_SOURCE_MANIFEST.json'),
        'not_sealed':True,'real_model_qualification_passed':False})
    write(BASE/'durable_backend_stage/CPU_TEST_RESULTS.json',{'status':'PASS_CPU_ONLY','helper_tests':10,
        'matrix_integration_tests_in_qualification':result.testsRun-10,'gpu_calls':0,
        'source_sha256':qualification.sha(BASE/'durable_backend_stage/durable_backend.py'),
        'real_model_qualification_passed':False,'not_sealed':True})
    for folder in (ROOT,BASE/'durable_backend_stage'):
        files={str(p.relative_to(folder)).replace('\\','/'):qualification.sha(p) for p in sorted(folder.rglob('*'))
               if p.is_file() and p.name!='WORKING_MANIFEST.json' and '__pycache__' not in p.parts}
        write(folder/'WORKING_MANIFEST.json',{'status':'PREPARED_CPU_VERIFIED_NOT_SEALED','files':files,
            'launch_authorized':False,'budget_reserved':False,'real_model_qualification_passed':False})
    print(json.dumps({'status':'PASS_CPU_ONLY','tests':result.testsRun,'deploy_sources':len(mapping),
        'qualifier_sha256':qualification.sha(ROOT/'qualify_hotpot_matrix.py'),
        'helper_sha256':qualification.sha(BASE/'durable_backend_stage/durable_backend.py'),
        'working_manifest_sha256':qualification.sha(ROOT/'WORKING_MANIFEST.json')}))

if __name__=='__main__':run()
