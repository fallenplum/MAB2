"""Retain synthetic CPU validation only; never import a model runtime."""
import dataclasses
import hashlib
import io
import json
import platform
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
import test_matrix_executor as tests
from matrix_executor import MATRIX_ID,SINGLE_ID,SINGLE_SYSTEM_PROMPT,DEPENDENCY_PINS,GraphConfig
from matrix_geometry import EDGES,B_EDGES,geometry


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.passed=[]
    def addSuccess(self,test):
        self.passed.append(test.id());super().addSuccess(test)


def write(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    loaded=any(name in sys.modules for name in ('torch','transformers'))
    pins={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
          ('matrix_executor.py','matrix_geometry.py','test_matrix_executor.py','run_cpu_checks.py')}
    write('CPU_TEST_RESULTS.json',{'status':'PASS_CPU_MOCK_ONLY' if result.wasSuccessful() and not loaded else 'FAIL',
          'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
          'skipped':len(result.skipped),'passed_test_ids':result.passed,'source_pins':pins,
          'python_version':platform.python_version(),'real_model_runtime_imported':loaded,
          'direct_frozen_operator_condition_comparisons':44,'probe_edges_checked':sorted(B_EDGES),
          'new_GPU_calls':0,'model_weights_read':0,'benchmark_or_R1_reads':0,
          'scientific_admission':False,'real_model_engineering_qualification':'NOT_RUN_REQUIRED_SEPARATELY'})
    (ROOT/'CPU_TESTS.log').write_text(stream.getvalue(),encoding='utf-8')
    write('SINGLE_M_PROTOCOL.json',{'status':'NEW_OPERATOR_CPU_PREPARED_NOT_REAL_QUALIFIED','control_id':SINGLE_ID,
          'source_sha256':pins['matrix_executor.py'],'system_prompt':SINGLE_SYSTEM_PROMPT,
          'user_prompt':'question verbatim; append two newlines, Supplied context:, newline and complete context when nonempty',
          'model':'mistral','max_new_tokens':256,'node_alias':'aggregate',
          'seed_rule':'derive_seed(question_id, caller_seed1, aggregate)',
          'sender_normalization':'frozen normalize_message using Mistral tokenizer',
          'received_message_tokens':0,'self_state_tokens':0,'candidate_inputs':[],
          'failure_policy':'retain format errors and paid raw generation; no retry; no scores'})
    write('INTERFACE_CONTRACT.json',{'status':'CPU_IMPLEMENTATION_CONTRACT_NOT_SCIENTIFIC_FREEZE','operator_id':MATRIX_ID,
          'source_pins':pins,'dependency_pins':DEPENDENCY_PINS,'config':dataclasses.asdict(GraphConfig(rounds=3)),
          'input_interface':'execute(question_mapping, seed1, seed2, probe_edge)',
          'question_keys':'id, question, optional complete context; frozen forbidden label fields rejected',
          'seed_rule':'two distinct caller-supplied integers; no seed selection or sampling inside module',
          'edge_indexing':'zero-based source-major','edge_order':EDGES,'allowed_probe_edges':sorted(B_EDGES),
          'success_geometry':[geometry(e) for e in sorted(B_EDGES)],
          'original_base_policy':{'B':'same-seed A','NO_EDGE':'same-seed A','A_MINUS_E':'same-seed A','B_MINUS_E':'same-seed B'},
          'missing_base_policy':'fresh execution of same condition and seed; never reuse a partial or alternate-control base',
          'overflow_policy':'retain INPUT_REJECTED terminal state and continue remaining conditions; no truncation or scores',
          'infrastructure_or_unknown_failure':'stop INCOMPLETE with MatrixExecutionError.partial_result; no automatic retry',
          'observer_policy':'identity, encode/decode, prompt, seed/RNG arguments and returned Generation unchanged',
          'result_privacy':'full question/context and per-node raw outputs are private caller-owned results',
          'resident_GPU_hours':None,'labels_loaded':False,'questions_sampled':False,'scoring_implemented':False,
          'statistical_selection_implemented':False,'scientific_admission':False,'real_model_engineering_qualification':'NOT_RUN'})
    print(json.dumps({'status':'PASS_CPU_MOCK_ONLY' if result.wasSuccessful() and not loaded else 'FAIL',
                      'tests':result.testsRun,'direct_condition_comparisons':44,'new_GPU_calls':0}))
    return 0 if result.wasSuccessful() and not loaded else 1


if __name__=='__main__':raise SystemExit(main())
