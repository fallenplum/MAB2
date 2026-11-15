"""Portable test entrypoint supplying the frozen legacy input/scoring adapters."""
from pathlib import Path
import argparse,hashlib,io,json,sys,time,unittest
def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).parent);p.add_argument('--receipt',type=Path);a=p.parse_args()
    package=a.package.resolve();cycle=package.parent
    legacy=cycle/'hotpot_selection_v1/select_hotpot.py';support=cycle/'hotpot_support'
    assert legacy.is_file() and (support/'hotpot_support.py').is_file()
    sys.path[:0]=[str(package),str(support)];sys.argv_legacy=str(legacy)
    start=time.monotonic();cpu=time.process_time();stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(package),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'status':'PASS' if result.wasSuccessful() else 'FAILED','tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,'new_GPU_calls':0,'benchmark_labels_read':False,'synthetic_only':True,'log':stream.getvalue(),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(package.glob('*.py'))}}
    if a.receipt:
        with a.receipt.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(stream.getvalue());print(json.dumps({k:v for k,v in receipt.items() if k not in ['log','source_sha256']}))
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()
