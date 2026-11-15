import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
for name in ['exogenous_paths_v1','receiver_factorial_v1']:sys.path.insert(0,str(HERE.parent/name))
import closure_operator as c
import exogenous_path_operator as base
from test_operator import Backend,payload

class Tests(unittest.TestCase):
    def matrix(self,shock='U',seeds=(1,2)):
        return c.ClosureOperator(Backend('q'),Backend('m')).run_matrix(payload(),list(seeds),shock)
    def test_exact_coverage_cost_and_cache(self):
        for n in [2,5]:
            r=self.matrix(seeds=range(n));self.assertEqual(len(r['conditions']),9*n)
            self.assertEqual(sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values()),31*n)
            self.assertEqual(len(r['request_trace']),15*n)
    def test_dead_node_elimination_symmetric(self):
        for shock in ['U','V']:
            for row in self.matrix(shock)['conditions']:
                if row['condition'] not in base.GRAPH_CONDITIONS:continue
                expected=[shock] if row['condition']=='G1_MINUS_E'+shock else []
                self.assertEqual(row['skipped_dead_nodes'],expected)
                self.assertEqual(set(row['nodes']),set(base.NODES)-set(expected))
    def test_all_active_nodes_get_all_raw_context(self):
        for row in self.matrix()['conditions']:
            for v in row['nodes'].values():self.assertEqual(len(json.loads(v['request']['messages'][1]['content'])['documents']),10)
    def test_equal_information_and_model_allocation_baseline(self):
        rows={v['key']:v for v in self.matrix()['conditions']}
        for s in [1,2]:
            for n in ['U','V']:
                self.assertEqual(rows[f'{s}/G0']['nodes'][n]['request'],rows[f'{s}/INDEPENDENT3_Q']['nodes'][n]['request'])
            self.assertEqual(rows[f'{s}/G0']['nodes']['M']['model'],'qwen')
    def test_disconnected_output_cannot_reach_terminal(self):
        # Run each graph directly to avoid whole-matrix call-count assertion for uncompiled control.
        for shock in ['U','V']:
            graph='G1_MINUS_E'+shock;op=c.ClosureOperator(Backend('q'),Backend('m'));reference=c.ClosureOperator(Backend('q'),Backend('m'))
            for x in [op,reference]:x._cache={};x._call_ids=set();x._traces=[];x._conditions=[];x._matrix_id='synthetic'
            op._condition(payload(),1,graph,shock,{})
            with patch.object(c,'live_nodes',return_value=base.NODES):reference._condition(payload(),1,graph,shock,{})
            a,b=op._conditions[0],reference._conditions[0]
            self.assertEqual(a['answer'],b['answer'])
            self.assertEqual(a['nodes']['M']['request'],b['nodes']['M']['request'])
            self.assertEqual(sum(v['calls'] for v in a['logical_service_cost_by_model'].values())+1,sum(v['calls'] for v in b['logical_service_cost_by_model'].values()))
    def test_label_and_seed_rejection(self):
        p=payload();p['answer']='forbidden'
        with self.assertRaises(ValueError):c.ClosureOperator(Backend('q'),Backend('m')).run_matrix(p,[1,2],'U')
        with self.assertRaises(ValueError):self.matrix(seeds=(1,2,3))
    def test_overflow_before_backend(self):
        q,m=Backend('q'),Backend('m');op=c.ClosureOperator(q,m,base.PathConfig(max_input_tokens=9))
        with self.assertRaises(base.PathExecutionError):op.run_matrix(payload(),[1,2],'U')
        self.assertEqual(len(q.physical_calls)+len(m.physical_calls),0)
        with self.assertRaises(RuntimeError):op.run_matrix(payload(),[1,2],'U')
if __name__=='__main__':unittest.main()
