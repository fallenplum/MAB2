import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exogenous_paths_v1'))
import exogenous_path_operator as base
from test_operator import Backend,payload
from factorial_operator import FactorialPathOperator,SCHEDULE,vote_choice
class Tests(unittest.TestCase):
    def matrix(self):return FactorialPathOperator(Backend('q'),Backend('m')).run_matrix(payload(),[1,2],'U')
    def test_coverage_and_full_cost(self):
        r=self.matrix();self.assertEqual(r['complete_terminal_conditions'],58)
        self.assertEqual(len(r['request_trace']),74);self.assertEqual(sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values()),218)
        self.assertEqual([c['key'] for c in r['conditions']],[f'{s}/{n}' for s in [1,2] for n in SCHEDULE])
    def test_upstream_is_exactly_shared_across_treatments(self):
        r=self.matrix();rows={c['key']:c for c in r['conditions']}
        for s in [1,2]:
            for g in base.GRAPH_CONDITIONS:
                original=rows[f'{s}/{g}']
                for variant in ['Q_PARTIAL','M_FULL','Q_FULL']:
                    changed=rows[f'{s}/{g}__{variant}']
                    self.assertEqual(original['edges'],changed['edges'])
                    for n in ['U','V','R']:
                        self.assertEqual(original['nodes'][n]['request'],changed['nodes'][n]['request'])
                        self.assertEqual(original['nodes'][n]['generation'],changed['nodes'][n]['generation'])
                        self.assertTrue(changed['nodes'][n]['reused'])
    def test_raw_context_only_changes_receiver(self):
        for row in self.matrix()['conditions']:
            if '__' not in row['condition']:continue
            variant=row['condition'].split('__')[1]
            for node,record in row['nodes'].items():
                data=json.loads(record['request']['messages'][1]['content'])
                expected=5 if node in ['U','V'] else 10 if node=='M' and variant.endswith('FULL') else 0
                self.assertEqual(len(data['documents']),expected)
            self.assertEqual(row['nodes']['M']['model'],'qwen' if variant.startswith('Q') else 'mistral')
    def test_matched_independent_model_allocation(self):
        r=self.matrix();rows={c['key']:c for c in r['conditions']}
        for s in [1,2]:
            reference=rows[f'{s}/INDEPENDENT3_Q']
            self.assertEqual([reference['logical_service_cost_by_model'][n]['calls'] for n in ['qwen','mistral']],[3,1])
            for n in ['U','V','R']:self.assertEqual(reference['nodes'][n]['request'],rows[f'{s}/INDEPENDENT3_M']['nodes'][n]['request'])
            self.assertEqual(len(json.loads(reference['nodes']['M']['request']['messages'][1]['content'])['documents']),10)
    def test_original_conditions_keep_original_requests(self):
        old=base.PathOperator(Backend('q'),Backend('m')).run_matrix(payload(),[1,2],'U')
        rows={c['key']:c for c in self.matrix()['conditions']}
        for c in old['conditions']:
            for n,v in c['nodes'].items():
                other=rows[c['key']]['nodes'][n]
                for f in ['messages','seed','model','max_new_tokens','config']:self.assertEqual(v['request'][f],other['request'][f])
                self.assertEqual(v['generation']['text'],other['generation']['text'])
    def test_overflow_fail_closed_and_no_retry(self):
        q=Backend('q');m=Backend('m');op=FactorialPathOperator(q,m,base.PathConfig(max_input_tokens=9))
        with self.assertRaises(base.PathExecutionError) as error:op.run_matrix(payload(),[1,2],'U')
        self.assertEqual(error.exception.partial_result['status'],'INPUT_REJECTED');self.assertEqual(len(q.physical_calls)+len(m.physical_calls),0)
        with self.assertRaises(RuntimeError):op.run_matrix(payload(),[1,2],'U')
    def test_vote_has_no_gold_and_preserves_raw_selected_answer(self):
        self.assertEqual(vote_choice(['The Cedar.','cedar','Birch']),0)
        self.assertEqual(vote_choice(['Birch','Cedar','Cedar.']),1)
        self.assertEqual(vote_choice(['Birch','Cedar','Oak']),0)
        self.assertEqual(vote_choice(['','', 'Cedar']),0)
        rows={c['key']:c for c in self.matrix()['conditions']}
        for s in [1,2]:
            row=rows[f'{s}/VOTE_Q3'];self.assertEqual(row['logical_service_cost_by_model']['qwen']['calls'],3)
            self.assertEqual(row['logical_service_cost_by_model']['mistral']['calls'],0)
            self.assertEqual(row['nodes']['M']['request'],rows[f'{s}/SINGLE_Q']['nodes']['M']['request'])
            self.assertEqual(row['answer'],row['nodes'][row['selected_node']]['generation']['text'].strip())
if __name__=='__main__':unittest.main()
