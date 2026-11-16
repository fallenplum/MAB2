import dataclasses,json,unittest
import exogenous_path_operator as base
from test_operator import Backend,Generation,payload
import evidence_operator as ev

class SelectorBackend(Backend):
    def generate(self,messages,seed,cap,config):
        if messages[0]['content']==ev.SELECT:
            docs=json.loads(messages[1]['content'])['documents']
            # Input-dependent ranking makes relay interventions capable of changing selection.
            ids=sorted((d['document_id'] for d in docs),key=lambda x:base.digest([x,seed]))[:2]
            text=json.dumps(ids)
        else:text=base.digest([messages,seed])[:12]
        g=Generation(text,10,len(text),.2,.1,100,'eos',self.identity['model']+str(len(self.physical_calls)))
        self.physical_calls.append({**dataclasses.asdict(g),'status':'SUCCESS','seed':seed,'prompt_sha256':base.digest(messages)})
        return g

class Tests(unittest.TestCase):
    def test_strict_selection(self):
        for text in ['D00','```json\n["D00"]\n```','["D99"]','["D00","D00"]','["D00","D01","D02"]','[1]','{}']:
            self.assertEqual(ev.parse_selection(text,['D00','D01','D02']),([],False))
        self.assertEqual(ev.parse_selection('["D01","D00"]',['D00','D01']),(['D00','D01'],True))
        self.assertEqual(ev.parse_selection('[]',[]),([],True))

    def test_partition_permissions_cost_and_certificates(self):
        p=payload()
        for shock in ['U','V']:
            r=ev.EvidenceOperator(SelectorBackend('q'),SelectorBackend('m')).run_matrix(p,[1,2],shock)
            self.assertEqual(r['complete_terminal_conditions'],22)
            self.assertEqual(sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values()),72)
            parts=r['document_partition'];self.assertEqual([len(parts[n]) for n in ['U','V','R']],[3,3,4])
            self.assertEqual(sorted(parts['U']+parts['V']+parts['R']),list(range(10)))
            seen={}
            for row in r['conditions']:
                if row['condition'] in ('INDEPENDENT3_Q','VOTE_Q3'):continue
                for n,rec in row['nodes'].items():
                    data=json.loads(rec['request']['messages'][1]['content'])
                    allowed=set(parts[n]) if n!='M' else set()
                    if row['condition'].startswith('SINGLE'):allowed=set(range(10))
                    for parent,node in row['edges']:
                        if node==n:allowed.update(int(x[1:]) for x in row['nodes'][parent]['evidence_ids'])
                    self.assertEqual(rec['available_evidence_ids'],[f'D{i:02d}' for i in sorted(allowed)])
                    for doc in data['documents']:
                        i=int(doc['document_id'][1:]);self.assertEqual(doc['sentences'],p['context']['sentences'][i])
                    if n!='M':self.assertTrue(rec['selection_valid'])
                if row['condition']!='SINGLE_M':
                    key=(row['execution_seed'],tuple(row['terminal_evidence_ids']))
                    value=(row['nodes']['M']['request_fingerprint'],row['answer'])
                    if key in seen:self.assertEqual(value,seen[key])
                    seen[key]=value
            dead=[c for c in r['conditions'] if c['condition']=='G1_MINUS_E'+shock]
            self.assertTrue(all(shock not in c['nodes'] for c in dead))

    def test_invalid_is_empty_not_retry(self):
        q=Backend('q');m=Backend('m');r=ev.EvidenceOperator(q,m).run_matrix(payload(),[1,2],'U')
        for c in r['conditions']:
            if c['condition'] in ('INDEPENDENT3_Q','VOTE_Q3'):continue
            for n,v in c['nodes'].items():
                if n!='M':self.assertFalse(v['selection_valid']);self.assertEqual(v['evidence_ids'],[])

    def test_labels_and_overflow(self):
        p=payload();p['answer']='forbidden'
        with self.assertRaises(ValueError):ev.EvidenceOperator(Backend('q'),Backend('m')).run_matrix(p,[1,2],'U')
        q=Backend('q');m=Backend('m')
        with self.assertRaises(base.PathExecutionError):ev.EvidenceOperator(q,m,base.PathConfig(max_input_tokens=9)).run_matrix(payload(),[1,2],'U')
        self.assertEqual(len(q.physical_calls)+len(m.physical_calls),0)

if __name__=='__main__':unittest.main()
