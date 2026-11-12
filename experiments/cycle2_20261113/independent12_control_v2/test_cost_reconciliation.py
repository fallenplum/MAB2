"""CPU-only repeated-state/identity/unknown-cost regressions. No HF or models."""
import ast,copy,hashlib,os,unittest
from pathlib import Path
from typing import Any
from test_independent12_executor import make
from graph_credit.executor import canonical
from independent12_executor import Independent12ExecutionError

QUESTION={'id':'synthetic_cost_reconciliation','question':'Return the synthetic marker.','context':'Synthetic logging-fault fixture only.'}


def event_failure_case(transform,at=1,return_generation=False):
    exe,router,state=make()
    # Target Qwen calls 1..3, so a failing request at 3 retains two paid predecessors.
    backend=router.backends['qwen'];original=backend.generate
    def generate(messages,seed,max_new_tokens,config):
        result=original(messages,seed,max_new_tokens,config)
        if len(state['calls'])==at:
            base=copy.deepcopy(backend.physical_calls[-1])
            replacement=transform(base)
            backend.physical_calls[-1:]=replacement
            if not return_generation:raise OSError('Synthetic post-generation logging failure')
        return result
    backend.generate=generate
    try:result=exe.execute(QUESTION,99191)
    except Independent12ExecutionError as error:result=error.partial_result
    return result,router,state


class CostReconciliationTests(unittest.TestCase):
    def test_identical_metrics_same_id_statuses_charged_once(self):
        def events(base):return [base,{**base,'status':'FAILED','error_type':'OSError'}]
        result,_,state=event_failure_case(events)
        cost=result.cost();event=result.slots[0].physical_events[0]
        self.assertEqual(cost['calls'],1)
        self.assertEqual(cost['input_tokens'],event['input_tokens'])
        self.assertEqual(cost['output_tokens'],event['output_tokens'])
        self.assertEqual(cost['model_elapsed_seconds'],event['elapsed_seconds'])
        self.assertTrue(cost['cost_fields_complete'])
        self.assertEqual(cost['request_cost_evidence'][0]['groups'][0]['event_statuses'],['SUCCESS','FAILED'])
        self.assertEqual(len(result.slots[0].physical_events),2)

    def test_conflicting_costs_unknown_and_predecessor_subtotal_retained(self):
        def events(base):return [base,{**base,'status':'FAILED','input_tokens':base['input_tokens']+7,'elapsed_seconds':.9}]
        result,_,_=event_failure_case(events,at=3)
        cost=result.cost();group=cost['request_cost_evidence'][2]['groups'][0]
        expected=sum(slot.generation.input_tokens for slot in result.slots[:2])
        self.assertIsNone(cost['input_tokens']);self.assertIsNone(cost['model_elapsed_seconds'])
        self.assertEqual(cost['known_cost_components']['input_tokens'],expected)
        self.assertEqual(cost['known_cost_ranges']['input_tokens'],{'lower':expected,'upper':None,'scope':'nonnegative subtotal lower bound; null upper means unresolved'})
        evidence=group['metrics']['input_tokens']
        self.assertTrue(evidence['conflict']);self.assertEqual(evidence['observed_max']-evidence['observed_min'],7)
        self.assertTrue(evidence['observed_range_is_not_a_truth_bound'])
        self.assertEqual(cost['output_tokens'],sum(slot.generation.output_tokens for slot in result.slots[:2])+result.slots[2].physical_events[0]['output_tokens'])

    def test_incomplete_same_id_does_not_fill_from_complete_success(self):
        def events(base):return [base,{'call_id':base['call_id'],'seed':base['seed'],'status':'FAILED','input_tokens':base['input_tokens'],'output_tokens':None,'elapsed_seconds':base['elapsed_seconds']}]
        result,_,_=event_failure_case(events)
        cost=result.cost();group=cost['request_cost_evidence'][0]['groups'][0]
        self.assertEqual(cost['input_tokens'],result.slots[0].physical_events[0]['input_tokens'])
        self.assertIsNone(cost['output_tokens']);self.assertIsNone(cost['gpu_seconds'])
        self.assertEqual(group['metrics']['output_tokens']['missing_count'],1)

    def test_missing_call_ids_are_not_merged_or_assumed_separate_charges(self):
        def events(base):
            base.pop('call_id');return [base,dict(base)]
        result,_,_=event_failure_case(events)
        cost=result.cost();groups=cost['request_cost_evidence'][0]['groups']
        self.assertEqual(len(groups),2)
        self.assertTrue(all(g['call_id'] is None for g in groups))
        self.assertIsNone(cost['input_tokens'])
        self.assertEqual(cost['known_cost_components']['input_tokens'],0)
        self.assertTrue(all('missing_call_id_cannot_establish_event_identity' in g['identity_issues'] for g in groups))

    def test_conflicting_seed_identity_makes_all_fees_unknown(self):
        def events(base):return [base,{**base,'seed':base['seed']+1}]
        result,_,_=event_failure_case(events)
        cost=result.cost();group=cost['request_cost_evidence'][0]['groups'][0]
        self.assertIsNone(cost['input_tokens']);self.assertIsNone(cost['output_tokens'])
        self.assertIn('conflicting_identity_field:seed',group['identity_issues'])

    def test_multiple_call_ids_for_one_request_are_identity_ambiguous(self):
        def events(base):return [base,{**base,'call_id':'different-id'}]
        result,_,_=event_failure_case(events)
        cost=result.cost()
        self.assertEqual(cost['calls'],1)
        self.assertEqual(len(cost['request_cost_evidence'][0]['groups']),2)
        self.assertIsNone(cost['input_tokens'])

    def test_returned_generation_is_not_preferred_over_conflicting_event(self):
        def events(base):return [{**base,'input_tokens':base['input_tokens']+9}]
        result,_,_=event_failure_case(events,return_generation=True)
        self.assertEqual(result.status,'COMPLETE');self.assertEqual(result.cost()['returned_generations'],13)
        self.assertIsNone(result.cost()['input_tokens'])
        self.assertTrue(result.cost()['request_cost_evidence'][0]['groups'][0]['metrics']['input_tokens']['conflict'])

    def test_call_id_reuse_across_backend_requests_is_not_silently_reconciled(self):
        exe,router,state=make();backend=router.backends['qwen'];original=backend.generate
        def generate(*args,**kwargs):
            result=original(*args,**kwargs)
            if len(state['calls'])<=3:
                backend.physical_calls[-1]['call_id']='reused-qwen-call'
                import dataclasses
                result=dataclasses.replace(result,call_id='reused-qwen-call')
            return result
        backend.generate=generate
        result=exe.execute(QUESTION,99191);cost=result.cost()
        self.assertIsNone(cost['input_tokens'])
        self.assertTrue(all('call_id_reused_across_backend_requests' in cost['request_cost_evidence'][i]['groups'][0]['identity_issues'] for i in range(3)))

    def test_exact_frozen_record_ast_append_then_oserror_same_id_failure_tail(self):
        # Execute only the exact _record method AST; never import/construct HFBackend.
        path=Path(__file__).resolve().parent/'frozen_dependencies/src/graph_credit/backend.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'284cfd5acc8c970b7525e498174466e4662bfe1b35a1235ae230bdf2e7e0e2ed')
        tree=ast.parse(path.read_text('utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HFBackend')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_record')
        generate_ast=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
        records=[n for n in ast.walk(generate_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_record']
        self.assertEqual(len(records),2)
        for call in records:
            fields={k.value:v for k,v in zip(call.args[0].keys,call.args[0].values)}
            self.assertIsInstance(fields['call_id'],ast.Name);self.assertEqual(fields['call_id'].id,'call_id')
        env={'Any':Any,'canonical':canonical,'os':os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),env)
        record=env['_record']
        class FailingLog:
            @property
            def parent(self):return self
            def mkdir(self,*a,**kw):pass
            def open(self,*a,**kw):raise OSError('Synthetic write-open failure after list append')
        exe,router,state=make();backend=router.backends['qwen'];original=backend.generate
        backend.physical_log=FailingLog()
        def generate(messages,seed,max_new_tokens,config):
            returned=original(messages,seed,max_new_tokens,config)
            success=backend.physical_calls.pop()
            try:
                record(backend,success)
            except Exception as error:
                # Exact relevant exception-tail semantics: same call_id, no new generation.
                record(backend,{'call_id':success['call_id'],'status':'FAILED','seed':seed,'input_tokens':success['input_tokens'],'output_tokens':None,'elapsed_seconds':.3,'error_type':type(error).__name__})
                raise
            return returned
        backend.generate=generate
        with self.assertRaises(Independent12ExecutionError) as caught:exe.execute(QUESTION,99191)
        result=caught.exception.partial_result;cost=result.cost();events=result.slots[0].physical_events
        self.assertEqual(len(state['calls']),1);self.assertEqual(len(events),2)
        self.assertEqual(events[0]['call_id'],events[1]['call_id'])
        self.assertEqual([e['status'] for e in events],['SUCCESS','FAILED'])
        self.assertEqual(cost['input_tokens'],events[0]['input_tokens'])
        self.assertIsNone(cost['output_tokens']);self.assertIsNone(cost['model_elapsed_seconds'])
        self.assertFalse(router._execution_lock.locked());self.assertIsNone(router._selected.get())
        self.assertTrue(all(s.status=='NOT_ATTEMPTED' for s in result.slots[1:]))


if __name__=='__main__':unittest.main(verbosity=2)
