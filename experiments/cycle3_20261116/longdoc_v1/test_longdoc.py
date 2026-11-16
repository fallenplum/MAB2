import dataclasses,json,unittest,copy
import exogenous_path_operator as base
from test_operator import Backend,Generation
import longdoc_operator as ld

class Selector(Backend):
    def generate(self,messages,seed,cap,config):
        passages=json.loads(messages[1]['content'])['passages']
        text=json.dumps([x['id'] for x in sorted(passages,key=lambda x:base.digest([x['id'],seed]))[:6]]) if messages[0]['content']==ld.SELECT else base.digest([messages,seed])[:12]
        g=Generation(text,10,min(100,len(text)),.2,.1,100,'eos',self.identity['model']+str(len(self.physical_calls)))
        self.physical_calls.append({**dataclasses.asdict(g),'status':'SUCCESS','seed':seed,'prompt_sha256':base.digest(messages)})
        return g
def payload():
    ids=[f'P{i:04d}' for i in range(45)]
    return dict(id='artificial',question='Artificial?',segments=[{'id':x,'text':x+' original text \u03b1\u4e2d\u6587'} for x in ids],retrieval={'BM25':ids[:18],'DENSE':ids[-18:]})
class Tests(unittest.TestCase):
    def test_permissions_cost_and_cache(self):
        p=payload()
        for shock in 'UV':
            r=ld.LongdocOperator(Selector('q'),Selector('m')).run_matrix(p,[1,2],shock)
            self.assertEqual(r['complete_terminal_conditions'],24);self.assertEqual(sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values()),68)
            seen={}
            for row in r['conditions']:
                for node,rec in row['nodes'].items():
                    available=set(rec['available_ids']);c=row['condition']
                    if c.startswith('G') or c=='FLAT':
                        allowed=set(r['document_partition'][node]) if node!='M' else set()
                        parents=[u for u,v in row['edges'] if v==node] if c.startswith('G') else list('UVR') if node=='M' else []
                        for n in parents:allowed.update(row['nodes'][n]['selected_ids'])
                    elif c in ('BM25','DENSE'):allowed=set(p['retrieval'][c])
                    else:allowed={x['id'] for x in p['segments']}
                    self.assertEqual(available,allowed)
                    docs=json.loads(rec['request']['messages'][1]['content'])['passages'];self.assertEqual(docs,[x for x in p['segments'] if x['id'] in available])
                    fp=rec['request_fingerprint']
                    if fp in seen:self.assertEqual(rec['generation'],seen[fp]);self.assertTrue(rec['reused'])
                    seen[fp]=rec['generation']
                if row['condition']=='G1_MINUS_E'+shock:self.assertNotIn(shock,row['nodes'])
            self.assertLess(len(r['request_trace']),68)
    def test_invalid_selection_and_labels(self):
        for t in ['["P9999"]','["P0001","P0001"]','{}','[1]','```json\n[]\n```']:
            self.assertEqual(ld.select(t,['P0001']),([],False))
        p=payload();p['answer']='not allowed'
        with self.assertRaises(AssertionError):ld.validate(p)
    def test_changed_evidence_invalidates_cache(self):
        p=payload();a=ld.LongdocOperator(Selector('q'),Selector('m')).run_matrix(p,[1,2],'U');p['segments'][0]['text']+=' changed';b=ld.LongdocOperator(Selector('q'),Selector('m')).run_matrix(p,[1,2],'U')
        self.assertNotEqual(a['question_fingerprint'],b['question_fingerprint'])
        self.assertFalse(set(v['request_fingerprint'] for v in a['request_trace'])&set(v['request_fingerprint'] for v in b['request_trace']))
    def test_vote(self):self.assertEqual(ld.vote(['Yes','No','yes']), 'Yes');self.assertEqual(ld.vote(['a','b','c']),'a')
if __name__=='__main__':unittest.main()
