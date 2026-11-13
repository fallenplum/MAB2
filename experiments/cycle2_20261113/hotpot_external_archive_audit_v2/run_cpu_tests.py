"""Only artificial byte maps and frozen CPU operators; no model/dataset access."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import traceback
import unittest
sys.dont_write_bytecode=True
import archive_audit as a
import synthetic_fixture as f

ROOT=Path(__file__).resolve().parent

class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=f.fixture()
        cls.result=a.audit_bundle(**cls.bundle)
        cls.plan=a.decode(cls.bundle['plan_bytes']);cls.contract=a.decode(cls.bundle['contract_bytes'])

    def test_full384_frozen_cpu_operator_archive_never_auto_releases(self):
        r=self.result
        self.assertEqual(r['status'],'REVIEW_REQUIRED')
        self.assertEqual(r['physical_request_count'],33408)
        self.assertEqual(r['returned_generations'],33408)
        self.assertEqual(r['full_logical_cost']['calls'],50304)
        self.assertEqual(r['residence']['current_seconds'],2000.)
        self.assertEqual(r['residence']['global_seconds'],2025.)
        self.assertFalse(r['label_release_authorized']);self.assertTrue(r['synthetic_tokenizer_witness'])
        self.assertEqual(r['external_release_template']['decision'],'REVIEW_REQUIRED')
        self.assertEqual(len(r['external_release_template']['units']),384)

    def test_evaluation_adapter_rejects_unapproved_audit_template_before_labels(self):
        sys.path.insert(0,str(f.BASE/'hotpot_evaluation_stage'))
        import evaluation_adapter as evaluation
        template=self.result['external_release_template'];raw=f.encode(template);files={}
        for name in a.METADATA:files[name]=self.bundle['archive'][name]
        for entry in template['units']:
            for name in entry['sha256']:files[entry['directory']+'/'+name]=self.bundle['archive'][entry['directory']+'/'+name]
        loaded=[]
        with self.assertRaisesRegex(evaluation.BindingError,'No external score release'):
            evaluation.evaluate_after_release(selection_manifest_bytes=self.bundle['pinned_inputs']['selection'],files=files,
                review_bytes=raw,expected_review_sha256=a.sha(raw),expected_bindings=template['bindings'],label_loader=lambda _:loaded.append(True))
        self.assertEqual(loaded,[])

    def test_pending_durable_start_and_noncontiguous_native_interval_rejected(self):
        unit=self.plan['schedule']['units'][0];w=unit['worker'];m='qwen'
        events=[a.decode(x) for x in self.bundle['archive'][f'worker_{w}/{m}_raw_requests.jsonl'].splitlines()]
        short=events[:3]
        def invoke(rows):return a.parse_journal(b''.join(f.encode(x) for x in rows),w,m,self.plan,a.sha(self.bundle['plan_bytes']),{u['index']:u for u in self.plan['schedule']['units']},self.bundle['tokenizers'][m])
        with self.assertRaisesRegex(a.AuditError,'unclosed START'):invoke(short[:2])
        changed=copy.deepcopy(short);changed[1]['native_event_start']=1;changed[2]['native_event_start']=1
        with self.assertRaisesRegex(a.AuditError,'native event interval start'):invoke(changed)
        changed=copy.deepcopy(short);changed[2]['generation']['output_tokens']+=1
        with self.assertRaisesRegex(a.AuditError,'native/returned cost disagreement'):invoke(changed)

    def test_unknown_preflight_error_requires_actual_tokenizer_evidence(self):
        unit=self.plan['schedule']['units'][0];w=unit['worker'];m='qwen'
        rows=[a.decode(x) for x in self.bundle['archive'][f'worker_{w}/{m}_raw_requests.jsonl'].splitlines()][:3]
        rows[2].update(event='ERROR',generation=None,error_type='ValueError',native_events=[],native_event_end=0)
        with self.assertRaisesRegex(a.AuditError,'unproven capacity rejection'):
            a.parse_journal(b''.join(f.encode(x) for x in rows),w,m,self.plan,a.sha(self.bundle['plan_bytes']),{u['index']:u for u in self.plan['schedule']['units']},self.bundle['tokenizers'][m])

    def test_duplicate_transaction_and_uncounted_resident_rejected(self):
        registry=a.decode(self.bundle['auxiliary']['REGISTRY.json']);registry['transactions']*=2
        with self.assertRaisesRegex(a.AuditError,'duplicate/current transaction'):
            a.check_residence(self.plan,a.sha(self.bundle['plan_bytes']),self.bundle['archive'],registry,self.bundle['prior_receipts'])
        registry=a.decode(self.bundle['auxiliary']['REGISTRY.json']);archive=dict(self.bundle['archive'])
        resident=a.decode(archive['PARENT_RESIDENCE.json']);resident['closed_sum_seconds']=0;archive['PARENT_RESIDENCE.json']=f.encode(resident)
        with self.assertRaisesRegex(a.AuditError,'resident total'):
            a.check_residence(self.plan,a.sha(self.bundle['plan_bytes']),archive,registry,self.bundle['prior_receipts'])

    def test_same_prior_receipt_cannot_be_billed_under_two_aliases(self):
        registry=a.decode(self.bundle['auxiliary']['REGISTRY.json']);first=registry['transactions'][0]
        registry['transactions'].append({**first,'transaction_id':'alias'})
        receipts={**self.bundle['prior_receipts'],'alias':self.bundle['prior_receipts'][first['transaction_id']]}
        with self.assertRaisesRegex(a.AuditError,'same prior receipt'):
            a.check_residence(self.plan,a.sha(self.bundle['plan_bytes']),self.bundle['archive'],registry,receipts)

    def test_whole_worker_two_complete_unit_blocks_cannot_swap(self):
        units=[u for u in self.plan['schedule']['units'] if u['worker']==0][:2];m='qwen'
        events=[a.decode(x) for x in self.bundle['archive']['worker_0/qwen_raw_requests.jsonl'].splitlines()]
        pairs=[]
        for u in units:
            i=next(i for i in range(1,len(events),2) if events[i]['context']['unit_index']==u['index'])
            pairs.append(copy.deepcopy(events[i:i+2]))
        rows=[events[0]]
        for counter,pair in enumerate(reversed(pairs)):
            pair[0]['native_event_start']=counter;pair[1]['native_event_start']=counter;pair[1]['native_event_end']=counter+1
            rows.extend(pair)
        plan=copy.deepcopy(self.plan);plan['schedule']['units']=units
        with self.assertRaisesRegex(a.AuditError,'whole-worker unit order'):
            a.parse_journal(b''.join(f.encode(x) for x in rows),0,m,plan,a.sha(self.bundle['plan_bytes']),{u['index']:u for u in units},self.bundle['tokenizers'][m])

    def test_real_frozen_operator_partial_independent_and_single_overflows_keep_unknown(self):
        plan=copy.deepcopy(self.plan);unit=dict(plan['schedule']['units'][0]);unit['worker']=0;plan['schedule']['units']=[unit]
        payload=a.collector(a.decode(self.bundle['pinned_inputs']['questions'].splitlines()[unit['index']]))
        natives={m:f.Native(m,0) for m in ('qwen','mistral')}
        natives['mistral'].overflow_predicate=lambda messages: ('Independent worker samples' in messages[1]['content'] or messages[0]['content'].startswith('You are a careful question-answering assistant.'))
        context={'attempt_id':plan['attempt_id'],'plan_sha256':a.sha(self.bundle['plan_bytes']),'worker_id':0,'unit_index':unit['index'],'unit_payload_sha256':unit['payload_sha256']}
        ref=[None];wrappers={m:f.MemoryDurable(n,m,lambda:{**context,'node':ref[0].router._selected.get()}) for m,n in natives.items()}
        matrix=f.HotpotApplicabilityMatrixExecutor(wrappers,f.GraphConfig(rounds=3));ref[0]=matrix
        data=a.decode(f.encode(matrix.execute(payload,*a.SEEDS,unit['edge_index']).to_dict()));calls={};rejects=[]
        for m,wrapper in wrappers.items():
            c,r=a.parse_journal(b''.join(f.encode(x) for x in wrapper.events),0,m,plan,context['plan_sha256'],{unit['index']:unit},natives[m]);calls.update(c);rejects.extend(r)
        constants={}
        for filename in ('executor.py','independent12_executor.py','matrix_executor.py'):
            raw=next(raw for path,raw in self.bundle['sources'].items() if Path(path).name==filename);constants=a.literal_constants(raw,constants)
        result=a.check_matrix(data,payload,unit,self.contract['bindings'],constants,natives,calls,rejects)
        self.assertEqual(result['rejections'],2);self.assertIsNone(result['full_logical'])
        self.assertEqual(len(rejects),2);self.assertIsNone(data['total_physical_cost']['output_tokens'])
        self.assertEqual(result['physical']['calls'],85)

    def test_formatter_numeric_duplicate_fallback_trim_independent_of_production(self):
        tok=self.bundle['tokenizers']['qwen']
        wire,parsed,meta=a.normalize('{"answer":25,"evidence":"x","uncertainty":""}',tok,128)
        self.assertEqual(parsed['answer'],'25');self.assertTrue(meta['type_normalized'])
        _,parsed,meta=a.normalize('{"answer":"a","answer":"b","evidence":"","uncertainty":""}',tok,128)
        self.assertFalse(meta['format_ok']);self.assertEqual(meta['format_error'],'duplicate_json_key');self.assertEqual(parsed['uncertainty'],'FORMAT_ERROR')
        wire,parsed,meta=a.normalize(a.canonical({'answer':'answer','evidence':'e'*200,'uncertainty':'u'*80}),tok,70)
        self.assertLessEqual(len(tok.encode(wire)),70);self.assertTrue(meta['message_trimmed']);self.assertEqual(parsed['answer'],'answer')

    def test_forecast_cannot_replace_maximum_with_faster_estimate(self):
        archive=dict(self.bundle['archive']);forecast=a.decode(archive['FORECAST_FROZEN.json'])
        forecast['predicted_remaining_seconds_by_worker'][0]=0
        archive['FORECAST_FROZEN.json']=f.encode(forecast)
        release=a.decode(archive['RELEASE_REMAINDER.json']);release['forecast_sha256']=a.sha(archive['FORECAST_FROZEN.json']);archive['RELEASE_REMAINDER.json']=f.encode(release)
        receipts={}
        for unit in self.plan['schedule']['units']:
            if unit['phase']=='prefix':receipts[unit['index']]=a.decode(archive[f"worker_{unit['worker']}/unit_{unit['index']:03d}/UNIT_RECEIPT.json"])
        with self.assertRaisesRegex(a.AuditError,'forecast arithmetic'):
            a.check_forecast(self.plan,a.sha(self.bundle['plan_bytes']),archive,receipts,self.result['residence'])

    def test_request_profile_must_be_rederived_from_trace(self):
        unit=self.plan['schedule']['units'][0];prefix=f"worker_{unit['worker']}/unit_{unit['index']:03d}"
        matrix=a.decode(self.bundle['archive'][prefix+'/MATRIX.json']);receipt=a.decode(self.bundle['archive'][prefix+'/UNIT_RECEIPT.json'])
        receipt['cost_profile']['observed_request_seconds_sum']=0
        with self.assertRaisesRegex(a.AuditError,'profile request times'):a.check_profile(matrix,unit,receipt['cost_profile'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    with (args.output/'CPU_TESTS.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArchiveTests))
    receipt={'status':'PASS_CPU_SYNTHETIC_PREPARATION_ONLY' if result.wasSuccessful() else 'FAIL_CPU_PREPARATION',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
        'elapsed_seconds':time.monotonic()-started,'benchmark_or_labels_read':False,'models_loaded':False,'GPU_calls':0,
        'source_sha256':{p.name:f.file_sha(p) for p in ROOT.glob('*.py')},'test_log_sha256':f.file_sha(args.output/'CPU_TESTS.log')}
    if hasattr(ArchiveTests,'result'):
        aggregate={k:v for k,v in ArchiveTests.result.items() if k!='external_release_template'}
        (args.output/'ARTIFICIAL_AGGREGATE.json').write_bytes(f.encode(aggregate))
        receipt['artificial_aggregate_sha256']=f.file_sha(args.output/'ARTIFICIAL_AGGREGATE.json')
    (args.output/'RECEIPT.json').write_bytes(f.encode(receipt));print(a.canonical(receipt),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()
