"""Artificial tests for leakage, grouping and pre-quality capacity selection."""
import importlib.util, pathlib, sys, unittest
import input_selection as m

def row(i, question=None):
    return {'id':f'artificial_{i:04d}','question':question or f'Artificial question {i}?',
        'context':{'title':[f'Artificial document {j}' for j in range(10)],
                   'sentences':[[f'Fabricated sentence {j}.'] for j in range(10)]},
        'type':'bridge','level':'easy'}

class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod=m.legacy(sys.argv_legacy)
    def exposure(self):
        return {'schema_version':'1.0','id_namespace':'hotpot_raw','ids':[],
            'question_group_hashes':[],'question_normalizer':self.mod.NORMALIZER,
            'content_hashes':[],'content_hash_algorithm':self.mod.CONTENT_HASH,
            'known_records_scope':'artificial tests only','unknown_cross_project_exposure':'UNKNOWN'}
    def test_label_fields_rejected(self):
        for field in ('answer','supporting_facts'):
            r=row(0);r[field]='forbidden'
            with self.assertRaises(ValueError):m.build_pool([r],self.exposure(),self.mod,size=1)
    def test_exposed_normalized_group_excluded(self):
        rows=[row(0,'  ARTIFICIAL   duplicate?'),row(1,'artificial duplicate?')]+[row(i) for i in range(2,8)]
        exp=self.exposure();exp['ids']=[rows[0]['id']]
        pool,stats=m.build_pool(rows,exp,self.mod,size=6)
        self.assertEqual({r['id'] for r in pool},{r['id'] for r in rows[2:]})
        self.assertEqual(stats['excluded_question_groups'],1)
    def test_repeatability_no_mutation_and_duplicate_id_failure(self):
        rows=[row(i) for i in range(10)]; before=m.canonical(rows)
        a,_=m.build_pool(rows,self.exposure(),self.mod,size=8)
        b,_=m.build_pool(rows,self.exposure(),self.mod,size=8)
        self.assertEqual(a,b);self.assertEqual(before,m.canonical(rows))
        with self.assertRaises(ValueError):m.build_pool(rows+[rows[0]],self.exposure(),self.mod,size=8)
    def measures(self,rows):
        names=['SINGLE_Q','SINGLE_M','INDEPENDENT3_U','INDEPENDENT3_V','INDEPENDENT3_R','INDEPENDENT3_M']
        return [{'index':i,'input_sha256':m.digest(r),'ten_documents':True,
                 'lengths':{k:100 for k in names}} for i,r in enumerate(rows)]
    def test_capacity_first_eligible_and_balanced_shock(self):
        rows=[row(i) for i in range(12)];ms=self.measures(rows)
        ms[0]['lengths']['SINGLE_Q']=7169;ms[3]['ten_documents']=False
        ids,shocks,_=m.select_capacity(rows,ms,n=8)
        self.assertEqual(ids,[1,2,4,5,6,7,8,9]);self.assertEqual(shocks.count('U'),4)
        self.assertEqual(shocks.count('V'),4)
    def test_no_auto_expansion_and_pin_or_coverage_failure(self):
        rows=[row(i) for i in range(4)];ms=self.measures(rows)
        ms[0]['lengths']['SINGLE_M']=7169
        with self.assertRaises(ValueError):m.select_capacity(rows,ms,n=4)
        ms=self.measures(rows);ms[0]['input_sha256']='0'*64
        with self.assertRaises(ValueError):m.select_capacity(rows,ms,n=4)
        ms=self.measures(rows);del ms[0]['lengths']['SINGLE_Q']
        with self.assertRaises(ValueError):m.select_capacity(rows,ms,n=4)

if __name__=='__main__':
    sys.argv_legacy=sys.argv.pop(1)
    unittest.main()
