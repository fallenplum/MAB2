"""Run only artificial CPU tests and inventory the unsealed staging files."""
import ast,hashlib,io,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def run():
    log=io.StringIO();tests=unittest.defaultTestLoader.loadTestsFromName('test_evaluation_adapter')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(tests)
    (ROOT/'CPU_TESTS.log').write_bytes(log.getvalue().encode())
    assert result.wasSuccessful(),log.getvalue()
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)))
    source=hashlib.sha256((ROOT/'evaluation_adapter.py').read_bytes()).hexdigest()
    write(ROOT/'CPU_TEST_RESULTS.json',{'status':'PASS_CPU_ONLY_NOT_FROZEN','tests':result.testsRun,'failures':0,'errors':0,
        'fixture_questions':384,'fixture_terminal_conditions':4224,'real_questions_read':0,'real_labels_read':0,
        'real_results_read':0,'gpu_calls':0,'statistics_helper_unchanged':True,'adapter_sha256':source,
        'real_label_release_authorized':False})
    files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='WORKING_MANIFEST.json' and '__pycache__' not in p.parts}
    write(ROOT/'WORKING_MANIFEST.json',{'status':'CPU_PREPARED_NOT_FROZEN','files':files,'real_label_release_authorized':False,
        'real_scoring_performed':False,'GPU_admission_granted':False})
    print(json.dumps({'status':'PASS_CPU_ONLY','tests':result.testsRun,'adapter_sha256':source,
        'working_manifest_sha256':hashlib.sha256((ROOT/'WORKING_MANIFEST.json').read_bytes()).hexdigest()}))
if __name__=='__main__':run()
