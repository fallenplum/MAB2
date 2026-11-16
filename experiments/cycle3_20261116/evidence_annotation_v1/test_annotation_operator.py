import json,unittest
import exogenous_path_operator as base
from test_operator import Backend,payload
from annotation_operator import AnnotationOperator

class Tests(unittest.TestCase):
    def test_reachability_notes_and_raw_text(self):
        p=payload()
        for shock in ['U','V']:
            result=AnnotationOperator(Backend('q'),Backend('m')).run_matrix(p,[1,2],shock)
            self.assertEqual(result['complete_terminal_conditions'],22)
            self.assertEqual(sum(c['calls'] for c in result['logical_completed_node_cost_by_model'].values()),72)
            for row in result['conditions']:
                if row['condition'] in ['INDEPENDENT3_Q','VOTE_Q3']:continue
                for node,rec in row['nodes'].items():
                    data=json.loads(rec['request']['messages'][1]['content'])
                    for doc in data['documents']:
                        i=int(doc['document_id'][1:]);self.assertEqual(doc['title'],p['context']['title'][i]);self.assertEqual(doc['sentences'],p['context']['sentences'][i])
                    expected=[dict(source=n,text=row['nodes'][n]['wire_message']) for n in base.NODES if [n,node] in row['edges']]
                    self.assertEqual(data['evidence_notes'],expected)
                    if node!='M':self.assertEqual(rec['evidence_ids'],rec['available_evidence_ids'])
                if row['condition'] in ['G0','G1','FLAT_ANNOTATION','SINGLE_Q','SINGLE_M']:self.assertEqual(row['terminal_evidence_ids'],[f'D{i:02d}' for i in range(10)])
                if row['condition']=='G1_MINUS_E'+shock:
                    self.assertNotIn(shock,row['nodes']);self.assertEqual(len(row['terminal_evidence_ids']),7)

    def test_evidence_ids_alone_do_not_certify_annotations(self):
        result=AnnotationOperator(Backend('q'),Backend('m')).run_matrix(payload(),[1,2],'U')
        rows={(c['execution_seed'],c['condition']):c for c in result['conditions']}
        a,b=rows[1,'G0'],rows[1,'G0_MINUS_EU']
        self.assertEqual(a['terminal_evidence_ids'],b['terminal_evidence_ids'])
        self.assertNotEqual(a['nodes']['M']['request_fingerprint'],b['nodes']['M']['request_fingerprint'])

    def test_dead_node_and_observed_prefix_cost(self):
        result=AnnotationOperator(Backend('q'),Backend('m')).run_matrix(payload(),[1,2],'U')
        rows={(c['execution_seed'],c['condition']):c for c in result['conditions']}
        self.assertEqual(sum(x['calls'] for x in rows[1,'G1_MINUS_EU']['logical_service_cost_by_model'].values()),3)
        self.assertEqual(sum(x['calls'] for x in rows[1,'G1_MINUS_EV']['logical_service_cost_by_model'].values()),4)
        self.assertLess(len(result['request_trace']),72)

    def test_labels_and_capacity_fail_closed(self):
        p=payload();p['answer']='forbidden'
        with self.assertRaises(ValueError):AnnotationOperator(Backend('q'),Backend('m')).run_matrix(p,[1,2],'U')
        q,m=Backend('q'),Backend('m')
        with self.assertRaises(base.PathExecutionError):AnnotationOperator(q,m,base.PathConfig(max_input_tokens=9)).run_matrix(payload(),[1,2],'U')
        self.assertEqual(len(q.physical_calls)+len(m.physical_calls),0)

if __name__=='__main__':unittest.main()
