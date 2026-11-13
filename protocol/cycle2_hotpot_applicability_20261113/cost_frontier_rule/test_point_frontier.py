import copy,unittest
from point_frontier import review_frontier,METHODS

def data():
    quality={m:100 for m in METHODS}
    costs={m:{'qwen':{'calls':2304,'input_tokens':100000,'output_tokens':10000,'CUDA_event_seconds':100.},
              'mistral':{'calls':2688,'input_tokens':200000,'output_tokens':20000,'CUDA_event_seconds':200.}} for m in METHODS}
    costs['SINGLE_M']={'qwen':{'calls':0,'input_tokens':0,'output_tokens':0,'CUDA_event_seconds':0.},
                       'mistral':{'calls':384,'input_tokens':30000,'output_tokens':3000,'CUDA_event_seconds':30.}}
    audit=dict(archive_integrity_passed=True,unique_questions=384,complete_conditions=4224,input_rejections=0,runtime_failures=0)
    return quality,costs,audit

class Tests(unittest.TestCase):
    def test_equal_quality_cheaper_single_witness(self):
        q,c,a=data();r=review_frontier(q,c,**a);self.assertEqual(r['witnesses'],['SINGLE_M'])
    def test_quality_tradeoff_not_rescued_by_cheaper_cost(self):
        q,c,a=data();q['SINGLE_M']=99;self.assertEqual(review_frontier(q,c,**a)['status'],'NOT_DOMINATED')
    def test_per_model_cost_tradeoff_not_hidden_by_sum(self):
        q,c,a=data();q['SINGLE_M']=99;q['NO_EDGE']=101;c['NO_EDGE']['qwen']['input_tokens']=10000;c['NO_EDGE']['mistral']['input_tokens']+=1
        self.assertEqual(review_frontier(q,c,**a)['status'],'NOT_DOMINATED')
    def test_exact_equality_not_strict_dominance(self):
        q,c,a=data();q['SINGLE_M']=0;self.assertEqual(review_frontier(q,c,**a)['witnesses'],[])
    def test_unknown_and_incomplete_not_free(self):
        q,c,a=data();c['B']['qwen']['CUDA_event_seconds']=None
        self.assertEqual(review_frontier(q,c,**a)['status'],'NOT_ASSESSED')
        q,c,a=data();a['input_rejections']=1;self.assertEqual(review_frontier(q,c,**a)['status'],'NOT_ASSESSED')
    def test_cache_discounted_call_geometry_rejected(self):
        q,c,a=data();c['B']['qwen']['calls']=1536
        with self.assertRaises(ValueError):review_frontier(q,c,**a)
    def test_noninteger_quality_or_malformed_model_rejected(self):
        q,c,a=data();q['B']=100.0
        with self.assertRaises(ValueError):review_frontier(q,c,**a)
        q,c,a=data();c['B']['qwen']['input_tokens']=-1
        with self.assertRaises(ValueError):review_frontier(q,c,**a)
    def test_no_rounding_of_time(self):
        q,c,a=data();q['SINGLE_M']=99;c['NO_EDGE']['mistral']['CUDA_event_seconds']=199.999999999
        self.assertEqual(review_frontier(q,c,**a)['witnesses'],['NO_EDGE'])
    def test_nonzero_calls_cannot_have_zero_inputs(self):
        q,c,a=data();c['B']['qwen']['input_tokens']=0
        with self.assertRaises(ValueError):review_frontier(q,c,**a)
    def test_zero_calls_cannot_hide_positive_cost(self):
        for key in ('input_tokens','output_tokens','CUDA_event_seconds'):
            q,c,a=data();c['SINGLE_M']['qwen'][key]=1
            with self.assertRaises(ValueError):review_frontier(q,c,**a)

if __name__=='__main__':unittest.main()
