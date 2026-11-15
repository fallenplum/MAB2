"""Explicit test classes avoid rerunning imported fixture TestCases."""
import importlib,sys,unittest,json,time
TESTS=[('test_closure_operator','Tests'),('test_closure_policy','Tests'),('test_closure_replay_policies','Tests'),('test_evaluate_closure','Tests'),('test_closure_runtime','Tests'),('test_audit_closure_independent','IndependentTests'),('test_audit_closure_raw_scoring','RawTests')]
if __name__=='__main__':
    start=time.perf_counter();suite=unittest.TestSuite()
    for name,cls in TESTS:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(importlib.import_module(name),cls)))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'wall_seconds':time.perf_counter()-start,'new_model_calls':0,'data_scope':'artificial only'}));sys.exit(not result.wasSuccessful())
