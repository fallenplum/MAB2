"""Portable synthetic-only test entrypoint. No model or data-source arguments."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--geometry')
    parser.add_argument('--durable-helper')
    args=parser.parse_args()
    if args.geometry:os.environ['MAB2_GEOMETRY']=str(Path(args.geometry).resolve())
    if args.durable_helper:os.environ['MAB2_DURABLE_HELPER']=str(Path(args.durable_helper).resolve())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();started=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_orchestration','test_pipeline','test_posix_guard'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (output/'STDOUT.txt').write_text(stream.getvalue(),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    dependency_hashes={}
    for key in ('MAB2_GEOMETRY','MAB2_DURABLE_HELPER'):
        if key in os.environ:dependency_hashes[key]=hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
    receipt={'status':'PASS_CPU_ONLY_NOT_FREEZE_OR_ADMISSION' if result.wasSuccessful() else 'FAIL',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':[{'test':str(test),'reason':reason} for test,reason in result.skipped],
        'elapsed_seconds':time.perf_counter()-started,'platform':platform.platform(),
        'python':sys.version,'source_sha256':hashes,'dependency_sha256':dependency_hashes,
        'synthetic_only':True,'models_loaded':False,'benchmark_read':False,'labels_read':False,
        'GPU_calls':0,'GPU_hours':0}
    (output/'RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(stream.getvalue(),end='');print(json.dumps({k:receipt[k] for k in ('status','tests_run','failures','errors','elapsed_seconds')}))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
