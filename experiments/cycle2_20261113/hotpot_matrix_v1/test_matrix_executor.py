"""Synthetic CPU tests: direct frozen-operator equivalence and failure geometry."""
import collections
import copy
import dataclasses
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'frozen_dependencies/src'),str(ROOT/'frozen_dependencies/router'),
              str(ROOT/'frozen_dependencies/control'),str(ROOT)]
from graph_credit.executor import Executor,Generation,GraphConfig,affected_closure,canonical,derive_seed,digest,normalize_message
from routed_executor import RoutedBackend,RoutedExecutor
from independent12_executor import Independent12Executor
from matrix_geometry import A,B,NO_EDGE,B_EDGES,ROUTE,geometry,without
from matrix_executor import (HotpotApplicabilityMatrixExecutor,MatrixExecutionError,SingleMExecutor,
    SINGLE_SYSTEM_PROMPT,SINGLE_ID,SEED1_ORDER,SEED2_ORDER,COST_KEYS,classify_failure)


class DeterministicBackend:
    def __init__(self,name,state,repeat):
        self.name=name;self.state=state;self.repeat=repeat
        self.identity={'model':name,'mock_only':True,'tokenizer_repeat':repeat,'version':'seed_and_prompt_deterministic'}
        self.physical_calls=[]

    def encode(self,text):return [ord(c) for c in text for _ in range(self.repeat)]

    def decode(self,tokens):return ''.join(chr(tokens[i]) for i in range(0,len(tokens),self.repeat))

    def generate(self,messages,seed,max_new_tokens,config):
        index=len(self.state['requests'])+1
        prompt=digest(messages);input_tokens=len(self.encode(canonical(messages)))
        request={'model':self.name,'seed':seed,'cap':max_new_tokens,'messages':copy.deepcopy(messages),'index':index}
        self.state['requests'].append(request)
        if index in self.state.get('overflow_at',set()):
            raise ValueError('Prompt has 8193 tokens; frozen cap is 8192')
        if input_tokens>config.max_input_tokens:
            raise ValueError(f'Prompt has {input_tokens} tokens; frozen cap is {config.max_input_tokens}')
        if index in self.state.get('position_limit_at',set()):
            raise ValueError('Prompt plus output reservation exceeds model position limit')
        call_id=f'{self.name}-mock-{self.state.get("reuse_call_id_at",{}).get(index,index)}'
        if index in self.state.get('infra_at',set()):
            self.physical_calls.append({'call_id':call_id,'status':'FAILED','seed':seed,'prompt_sha256':prompt,
                'input_tokens':input_tokens,'output_tokens':None,'elapsed_seconds':.3,'error_type':'RuntimeError'})
            raise RuntimeError('synthetic CUDA out of memory')
        if index in self.state.get('other_value_error_at',set()):
            raise ValueError('synthetic tokenizer failure')
        token=digest([self.name,seed,messages])[:5]
        text=canonical({'answer':token,'evidence':'x','uncertainty':''})
        if index in self.state.get('format_at',set()):text='invalid-json'
        generation=Generation(text,input_tokens,len(self.encode(text)),.1,.02,1000*self.repeat,'eos',call_id)
        self.physical_calls.append({'call_id':call_id,'status':'SUCCESS','seed':seed,'prompt_sha256':prompt,
            'input_tokens':generation.input_tokens,'output_tokens':generation.output_tokens,'elapsed_seconds':.1,
            'gpu_seconds':.02,'gpu_peak_allocated_bytes':1000*self.repeat,'finish_reason':'eos',
            'generation_config':{'max_new_tokens':max_new_tokens}})
        request['generation']=generation
        if index in self.state.get('logging_failure_at',set()):
            self.physical_calls.append({'call_id':call_id,'status':'FAILED','seed':seed,'prompt_sha256':prompt,
                'input_tokens':input_tokens,'output_tokens':None,'elapsed_seconds':.12,'error_type':'OSError'})
            raise OSError('synthetic append succeeded then disk write failed')
        return generation


def models(**flags):
    state={'requests':[],**flags}
    return {'qwen':DeterministicBackend('qwen',state,1),'mistral':DeterministicBackend('mistral',state,2)},state


def semantic_nodes(result):
    return {name:{'messages':r.messages,'seed':r.seed,'fingerprint':r.input_fingerprint,
                  'wire':r.wire_message,'parsed':r.parsed,'metadata':r.message_metadata,
                  'received':r.received_message_tokens,'self_state':r.self_state_tokens,
                  'text':r.generation.text,'input':r.generation.input_tokens,'output':r.generation.output_tokens,
                  'finish':r.generation.finish_reason} for name,r in result.nodes.items()}


def branch_semantics(branch):
    raw=branch.execution
    nodes=raw['nodes'] if 'nodes' in raw else {s['node']:s['record'] for s in raw['slots']}
    return {name:{'messages':r['messages'],'seed':r['seed'],'fingerprint':r['input_fingerprint'],
                  'wire':r['wire_message'],'parsed':r['parsed'],'metadata':r['message_metadata'],
                  'received':r['received_message_tokens'],'self_state':r['self_state_tokens'],
                  'text':r['generation']['text'],'input':r['generation']['input_tokens'],
                  'output':r['generation']['output_tokens'],'finish':r['generation']['finish_reason']} for name,r in nodes.items()}


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.question={'id':'synthetic_matrix_question','question':'Which marker does the passage identify?',
            'context':'This complete synthetic passage identifies cobalt as the marker. All context must remain present.'}
        self.seed1=26111301;self.seed2=26111302

    def make(self,**flags):
        backends,state=models(**flags)
        return HotpotApplicabilityMatrixExecutor(backends),backends,state

    def test_independent_geometry_matches_frozen_closure_for_all_four_edges(self):
        for edge in sorted(B_EDGES):
            g=geometry(edge)
            self.assertEqual((g['seed1_calls'],g['seed2_calls'],g['total_success_physical_calls'],g['total_success_logical_calls']),(55,32,87,131))
            for name,old,new in (('B',A,B),('NO_EDGE',A,NO_EDGE),('A_MINUS_E',A,without(A,edge)),('B_MINUS_E',B,without(B,edge))):
                self.assertEqual(set(g['closures'][name]),affected_closure(old,new,3))

    def test_matrix_order_counts_model_routes_and_logical_cost_for_all_edges(self):
        for edge in sorted(B_EDGES):
            matrix,_,state=self.make()
            result=matrix.execute(self.question,self.seed1,self.seed2,edge)
            self.assertEqual(result.status,'COMPLETE')
            self.assertEqual([c.name for c in result.conditions],list(SEED1_ORDER+SEED2_ORDER))
            self.assertEqual([c.physical_cost['calls'] for c in result.conditions],[13,9,9,6,4,13,1,13,9,6,4])
            self.assertEqual([c.logical_service_cost['calls'] for c in result.conditions],[13,13,13,13,13,13,1,13,13,13,13])
            output=result.to_dict()
            self.assertEqual(len(state['requests']),87)
            self.assertEqual(output['total_physical_cost']['calls'],87)
            self.assertEqual(output['total_logical_service_cost']['calls'],131)
            self.assertEqual(output['request_count_by_model'],geometry(edge)['total_success_model_calls'])
            self.assertFalse(output['scores_computed'] or output['scientific_admission'])

    def test_every_condition_matches_direct_frozen_operator_semantics_and_cost(self):
        for edge in sorted(B_EDGES):
            with self.subTest(probe_edge=edge):self._check_direct_equivalence(edge)

    def _check_direct_equivalence(self,edge):
        matrix,_,_=self.make();result=matrix.execute(self.question,self.seed1,self.seed2,edge)
        backends,state=models();router=RoutedBackend(backends,ROUTE,'mistral');config=GraphConfig(rounds=3)
        graph=RoutedExecutor(router,config);independent=Independent12Executor(router,config)
        self.assertEqual(matrix.router.identity,router.identity)
        self.assertEqual(matrix.graph.protocol_hash,graph.protocol_hash)
        self.assertEqual(matrix.independent.protocol_hash,independent.protocol_hash)
        bases={}
        for branch in result.conditions:
            if branch.mask is not None:
                desired={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(branch.name)
                base=bases.get((branch.seed_number,desired))
                direct=graph.execute(self.question,branch.mask,branch.seed,base)
                if branch.name in ('A','B'):bases[(branch.seed_number,branch.name)]=direct
                self.assertEqual(branch_semantics(branch),semantic_nodes(direct))
                self.assertEqual(branch.execution['physical_cost'],direct.cost())
                self.assertEqual(branch.execution['logical_service_cost'],direct.cost(False))
            elif branch.name=='INDEPENDENT12':
                direct=independent.execute(self.question,branch.seed)
                self.assertEqual(branch_semantics(branch),semantic_nodes(direct))
                for key in COST_KEYS:self.assertEqual(branch.physical_cost[key],direct.cost()[key])
            else:
                # A direct M call with the explicitly frozen new prompt, not a worker or candidate aggregator.
                router.select_node('aggregate')
                messages=[{'role':'system','content':SINGLE_SYSTEM_PROMPT},
                          {'role':'user','content':self.question['question']+'\n\nSupplied context:\n'+self.question['context']}]
                seed=derive_seed(self.question['id'],branch.seed,'aggregate')
                gen=router.generate(messages,seed,256,config)
                wire,parsed,metadata=normalize_message(gen.text,router,256)
                router.clear_node()
                slot=branch.execution['slots'][0]
                self.assertEqual(slot['messages'],messages)
                self.assertEqual(slot['generation']['text'],gen.text)
                self.assertEqual((slot['record']['wire_message'],slot['record']['parsed'],slot['record']['message_metadata']),(wire,parsed,metadata))
                self.assertEqual(branch.physical_cost['input_tokens'],gen.input_tokens)
                self.assertEqual(branch.physical_cost['output_tokens'],gen.output_tokens)
                self.assertEqual(branch.physical_cost['calls'],1)
        self.assertEqual(len(state['requests']),87)

    def test_graph_lineage_uses_only_declared_original_base_and_same_seed(self):
        matrix,_,_=self.make();result=matrix.execute(self.question,self.seed1,self.seed2,5)
        branches={c.key:c for c in result.conditions}
        for branch in result.conditions:
            if branch.mask is None:continue
            expected={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(branch.name)
            expected_key=f'S{branch.seed_number}/{expected}' if expected else None
            self.assertEqual(branch.cache_source_condition,expected_key)
            if expected_key:
                base=branches[expected_key].execution
                reused=[r for r in branch.execution['nodes'].values() if r['reused']]
                self.assertTrue(all(r['source_execution_id']==base['execution_id'] for r in reused))
            for node,record in branch.execution['nodes'].items():
                self.assertEqual(record['seed'],derive_seed(self.question['id'],branch.seed,node))
        one={r['generation']['call_id'] for r in branches['S1/A'].execution['nodes'].values()}
        two={r['generation']['call_id'] for r in branches['S2/A'].execution['nodes'].values()}
        self.assertTrue(one.isdisjoint(two))

    def test_independent_and_single_have_no_cross_control_candidates_or_cache(self):
        matrix,_,_=self.make();result=matrix.execute(self.question,self.seed1,self.seed2,7)
        branches={c.name:c for c in result.conditions if c.seed_number==1}
        independent=branches['INDEPENDENT12'].execution
        plain=Executor(matrix.router,GraphConfig(rounds=3))
        for slot in independent['slots'][:12]:
            expected,_,_=plain._messages(self.question,f'r1:w{slot["role"]}',(0,)*12,{})
            self.assertEqual(slot['messages'],expected)
            self.assertFalse(slot['record']['reused'])
        single=branches['SINGLE_M'].execution
        self.assertEqual(single['control_id'],SINGLE_ID)
        self.assertEqual(len(single['slots']),1)
        content=single['slots'][0]['messages']
        self.assertEqual(content[0]['content'],SINGLE_SYSTEM_PROMPT)
        self.assertEqual(content[1]['content'],self.question['question']+'\n\nSupplied context:\n'+self.question['context'])
        self.assertNotIn('worker',content[0]['content'].lower())
        self.assertEqual(single['slots'][0]['record']['received_message_tokens'],0)
        self.assertEqual(single['physical_cost']['planned_calls'],1)
        ids={branches[name].execution['protocol_hash'] for name in ('A','INDEPENDENT12','SINGLE_M')}
        self.assertEqual(len(ids),3)

    def test_single_format_failure_remains_paid_without_scoring(self):
        matrix,_,state=self.make(format_at={55})
        result=matrix.execute(self.question,self.seed1,self.seed2,1)
        single=next(c for c in result.conditions if c.name=='SINGLE_M')
        self.assertEqual(single.status,'COMPLETE')
        self.assertEqual(single.execution['format_failures'],1)
        self.assertEqual(single.execution['answer'],'')
        self.assertEqual(single.execution['slots'][0]['generation']['text'],'invalid-json')
        self.assertEqual(single.physical_cost['calls'],1)
        self.assertEqual(len(state['requests']),87)
        self.assertFalse(result.to_dict()['scores_computed'])

    def test_missing_A_base_fresh_continuation_can_exceed_87(self):
        matrix,_,state=self.make(overflow_at={1})
        result=matrix.execute(self.question,self.seed1,self.seed2,1)
        self.assertEqual(result.status,'COMPLETE_WITH_INPUT_REJECTIONS')
        branches={c.key:c for c in result.conditions}
        self.assertEqual(branches['S1/A'].status,'INPUT_REJECTED')
        self.assertIsNone(branches['S1/A'].physical_cost['input_tokens'])
        for name in ('B','NO_EDGE','A_MINUS_E'):
            branch=branches['S1/'+name]
            self.assertEqual(branch.replay_mode,'FRESH_BASE_UNAVAILABLE')
            self.assertEqual(branch.physical_cost['calls'],13)
            self.assertIsNone(branch.cache_source_condition)
        self.assertEqual(branches['S1/B_MINUS_E'].cache_source_condition,'S1/B')
        self.assertEqual(len(state['requests']),90)
        self.assertIsNone(result.to_dict()['total_physical_cost']['input_tokens'])
        self.assertIsNone(result.to_dict()['total_logical_service_cost'])
        self.assertTrue(result.to_dict()['condition_set_terminal'])

    def test_aggregate_overflow_preserves_12_partial_nodes_and_never_reuses_partial_base(self):
        matrix,_,state=self.make(overflow_at={13})
        result=matrix.execute(self.question,self.seed1,self.seed2,9)
        a=result.conditions[0];b=result.conditions[1]
        self.assertEqual(a.status,'INPUT_REJECTED')
        self.assertEqual(len(a.partial_execution['nodes']),12)
        self.assertEqual(sum(s['generation'] is not None for s in a.request_trace),12)
        self.assertEqual(a.physical_cost['calls'],13)
        self.assertIsNone(a.physical_cost['output_tokens'])
        self.assertEqual(b.replay_mode,'FRESH_BASE_UNAVAILABLE')
        self.assertEqual(b.execution['cache_reuse_count'],0)
        self.assertEqual(len(state['requests']),102)

    def test_missing_B_base_does_not_replay_B_minus_from_A_or_partial_B(self):
        matrix,_,state=self.make(overflow_at={14})
        result=matrix.execute(self.question,self.seed1,self.seed2,5)
        branches={c.key:c for c in result.conditions}
        self.assertEqual(branches['S1/B'].status,'INPUT_REJECTED')
        self.assertEqual(len(branches['S1/B'].partial_execution['nodes']),4)
        bm=branches['S1/B_MINUS_E']
        self.assertEqual(bm.replay_mode,'FRESH_BASE_UNAVAILABLE')
        self.assertIsNone(bm.cache_source_condition)
        self.assertEqual(bm.physical_cost['calls'],13)
        self.assertEqual(len(state['requests']),88)

    def test_infrastructure_failure_stops_preserves_raw_generations_and_cleans_locks(self):
        matrix,_,state=self.make(infra_at={6})
        with self.assertRaises(MatrixExecutionError) as caught:
            matrix.execute(self.question,self.seed1,self.seed2,1)
        result=caught.exception.partial_result
        self.assertEqual(result.status,'INCOMPLETE')
        self.assertEqual(len(state['requests']),6)
        self.assertEqual(len(result.conditions[0].partial_execution['nodes']),5)
        self.assertEqual(sum(s['generation'] is not None for s in result.conditions[0].request_trace),5)
        self.assertTrue(all(c.status=='NOT_RUN' for c in result.conditions[1:]))
        self.assertIsNone(result.conditions[0].physical_cost['output_tokens'])
        self.assertFalse(matrix._matrix_lock.locked() or matrix.router._execution_lock.locked())
        self.assertIsNone(matrix.router._selected.get())
        state['infra_at']=set()
        other=matrix.execute(self.question,self.seed1,self.seed2,1)
        self.assertEqual(other.to_dict()['total_physical_cost']['calls'],87)
        self.assertEqual(len(matrix._traces),87)

    def test_only_exact_input_preflight_errors_continue(self):
        self.assertEqual(classify_failure(ValueError('Prompt has 8193 tokens; frozen cap is 8192')),'INPUT_TOKEN_LIMIT')
        self.assertEqual(classify_failure(ValueError('Prompt plus output reservation exceeds model position limit')),'MODEL_POSITION_LIMIT')
        matrix,_,state=self.make(other_value_error_at={1})
        with self.assertRaises(MatrixExecutionError):matrix.execute(self.question,self.seed1,self.seed2,1)
        self.assertEqual(len(state['requests']),1)

    def test_cross_condition_call_id_reuse_is_unresolved_and_stops(self):
        matrix,_,state=self.make(reuse_call_id_at={14:1})
        with self.assertRaises(MatrixExecutionError) as caught:
            matrix.execute(self.question,self.seed1,self.seed2,1)
        result=caught.exception.partial_result
        self.assertEqual(len(state['requests']),22)
        self.assertEqual(result.conditions[1].failure_category,'INTEGRITY_FAILURE')
        self.assertTrue(all(c.status=='NOT_RUN' for c in result.conditions[2:]))
        costs=result.to_dict()['total_physical_cost']
        self.assertEqual(costs['calls'],22)
        self.assertIsNone(costs['input_tokens'])
        self.assertIsNone(costs['gpu_seconds'])

    def test_route_log_failure_keeps_returned_generation_missing_from_graph_nodes(self):
        matrix,_,state=self.make()
        original_record=matrix.router._record
        def record_then_fail(row):
            original_record(row)
            if len(matrix.router.route_events)==4:raise OSError('synthetic route fsync failure')
        matrix.router._record=record_then_fail
        with self.assertRaises(MatrixExecutionError) as caught:
            matrix.execute(self.question,self.seed1,self.seed2,1)
        branch=caught.exception.partial_result.conditions[0]
        self.assertEqual(len(state['requests']),4)
        self.assertEqual(len(branch.partial_execution['nodes']),3)
        self.assertEqual(len(branch.request_trace),4)
        self.assertTrue(all(row['generation'] is not None for row in branch.request_trace))
        self.assertEqual(branch.physical_cost['returned_generations'],4)
        self.assertTrue(branch.physical_cost['cost_fields_complete'])
        self.assertEqual(branch.route_events[-1]['status'],'SUCCESS')

    def test_backend_success_then_failure_same_id_is_one_request_with_unknown_cost(self):
        matrix,_,state=self.make(logging_failure_at={3})
        with self.assertRaises(MatrixExecutionError) as caught:
            matrix.execute(self.question,self.seed1,self.seed2,1)
        branch=caught.exception.partial_result.conditions[0]
        self.assertEqual(len(state['requests']),3)
        self.assertEqual(branch.physical_cost['calls'],3)
        self.assertEqual(len(branch.request_trace[-1]['physical_events']),2)
        self.assertIsNone(branch.request_trace[-1]['generation'])
        self.assertIsNone(branch.physical_cost['output_tokens'])
        self.assertIsNone(branch.physical_cost['model_elapsed_seconds'])

    def test_invalid_inputs_and_identity_changes_rejected_before_model_calls(self):
        matrix,_,state=self.make()
        for edge in (0,2,True,'1'):
            with self.assertRaises(ValueError):matrix.execute(self.question,self.seed1,self.seed2,edge)
        with self.assertRaises(ValueError):matrix.execute(self.question,self.seed1,self.seed1,1)
        with self.assertRaises(ValueError):matrix.execute({**self.question,'gold':'forbidden'},self.seed1,self.seed2,1)
        matrix.graph.config=GraphConfig(rounds=2)
        with self.assertRaises(RuntimeError):matrix.execute(self.question,self.seed1,self.seed2,1)
        self.assertEqual(state['requests'],[])


if __name__=='__main__':unittest.main(verbosity=2)
