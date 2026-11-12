"""All examples synthetic; does not load any experiment arrays or labels."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import confirmation_calculator as c

class Tests(unittest.TestCase):
    def test_binary_moments_enumeration(self):
        for delta,d in ((0,.1),(.02,.1),(-.05,.2),(1,1)):
            plus,minus=(d+delta)/2,(d-delta)/2
            self.assertAlmostEqual(plus-minus,delta)
            self.assertAlmostEqual(plus*(1-delta)**2+minus*(-1-delta)**2+(1-d)*delta**2,c.paired_variance(delta,d))
    def test_invalid_binary_moments(self):
        for x in ((.2,.1),(0,-.1),(0,1.1),(float('nan'),.1),(0,float('inf'))):
            with self.assertRaises(ValueError): c.paired_variance(*x)
    def test_null_normal_power(self):
        self.assertAlmostEqual(c.conditional_pair(192,0,.1)['two_sided_normal_power'],.05,places=12)
    def test_degenerate_is_not_power_claim(self):
        self.assertIsNone(c.conditional_pair(100,0,0)['two_sided_normal_power'])
        self.assertIsNone(c.conditional_pair(100,1,1)['two_sided_normal_power'])
    def test_halfwidth_reference(self):
        self.assertAlmostEqual(c.conditional_pair(192,0,.1)['normal_95_or_alpha_halfwidth'],1.959963984540054*math.sqrt(.1/192),places=12)
    def test_required_n_minimal(self):
        n=c.required_questions_normal(.02,.1)
        self.assertGreaterEqual(c.conditional_pair(n,.02,.1)['two_sided_normal_power'],.8)
        self.assertLess(c.conditional_pair(n-1,.02,.1)['two_sided_normal_power'],.8)
    def test_no_power_for_null(self):
        self.assertIsNone(c.required_questions_normal(0,.1))
    def test_alpha_and_power_guards(self):
        for alpha in (0,1,-1,float('nan')):
            with self.assertRaises(ValueError): c.conditional_pair(10,.02,.1,alpha)
        with self.assertRaises(ValueError): c.required_questions_normal(.02,.1,.01)
    def test_episode_zero_clustering_matches_iid(self):
        a=c.episode_design(8,24,.02,.1,0)
        b=c.conditional_pair(192,.02,.1)
        self.assertAlmostEqual(a['se'],b['se'])
    def test_episode_all_cluster_has_floor(self):
        v=c.paired_variance(.02,.1)
        a=c.episode_design(8,24,.02,.1,math.sqrt(v))
        b=c.episode_design(8,24000,.02,.1,math.sqrt(v))
        self.assertAlmostEqual(a['se'],b['se'])
        self.assertAlmostEqual(a['effective_iid_question_count'],8)
    def test_more_episodes_reduces_floor(self):
        a=c.episode_design(8,24,.02,.1,.1)
        b=c.episode_design(16,24,.02,.1,.1)
        self.assertAlmostEqual(a['mean_variance'],2*b['mean_variance'])
    def test_episode_invalid_variance(self):
        with self.assertRaises(ValueError): c.episode_design(8,24,.02,.1,.9)
    def test_seed_question_only_no_gain(self):
        a=c.crossed_seed_design(400,1,.02,.1,1,0)
        b=c.crossed_seed_design(400,5,.02,.1,1,0)
        self.assertAlmostEqual(a['se'],b['se'])
        self.assertEqual(b['unique_questions'],400)
        self.assertEqual(b['total_paired_evaluations'],2000)
    def test_seed_residual_only_gain(self):
        a=c.crossed_seed_design(400,1,.02,.1,0,0)
        b=c.crossed_seed_design(400,5,.02,.1,0,0)
        self.assertAlmostEqual(a['mean_variance'],5*b['mean_variance'])
    def test_global_seed_floor(self):
        a=c.crossed_seed_design(400,5,.02,.1,0,1)
        b=c.crossed_seed_design(40000,5,.02,.1,0,1)
        self.assertAlmostEqual(a['se'],b['se'])
    def test_crossed_components_guard(self):
        with self.assertRaises(ValueError): c.crossed_seed_design(400,5,.02,.1,.9,.2)
    def test_condition_covariance_not_double_n(self):
        a=c.condition_interaction(192,0,.1,.1,1)
        b=c.condition_interaction(192,.02,.1,.1,0)
        self.assertEqual(a['se'],0)
        self.assertAlmostEqual(b['se']**2,.2/192)
        self.assertEqual(b['unique_questions'],192)
    def test_negative_condition_correlation(self):
        a=c.condition_interaction(10,.1,.1,.1,-1)
        self.assertAlmostEqual(a['interaction_pair_variance'],.4)
    def test_capacity_union_not_sum(self):
        a=c.role_capacity(10,[{'a','b'},{'b','c'}],1)
        self.assertEqual(a['available_upper_after_known_exclusions'],6)
    def test_capacity_fail_closed(self):
        with self.assertRaises(ValueError): c.role_capacity(2,[{'a','b','c'}])
    def test_official_upper_is_not_seed_multiplied(self):
        a=c.gsm8k_test_capacity(requested=1320)
        self.assertEqual(a['source_upper'],1319)
        self.assertFalse(a['fits_known_upper_bound'])
    def test_hash_role_dedup(self):
        a,b='a'*64,'b'*64
        self.assertEqual(c.validate_role_hashes({'dev':[a],'test':[b]})['counts']['dev'],1)
        with self.assertRaises(ValueError): c.validate_role_hashes({'dev':[a],'test':[a]})
        with self.assertRaises(ValueError): c.validate_role_hashes({'test':[b]},[b])
        with self.assertRaises(ValueError): c.validate_role_hashes({'test':[a,a]})
    def test_role_bad_hash(self):
        with self.assertRaises(ValueError): c.validate_role_hashes({'test':['raw-question-is-forbidden']})
    def test_budget_all_disjoint_charges(self):
        a=c.budget(40,15,25,80)
        self.assertEqual(a['available_before_proposal_gpu_hours'],80)
        self.assertTrue(a['fits_assumed_ledger'])
        self.assertFalse(c.budget(40,15,25,80.01)['fits_assumed_ledger'])
    def test_overcommit(self):
        a=c.budget(161,0,0,0)
        self.assertTrue(a['overcommitted_before_proposal'])
        self.assertFalse(a['fits_assumed_ledger'])
    def test_failure_and_load_cost(self):
        a=c.call_budget(2,400,5,37,5,696,1,1,.15)
        self.assertEqual(a['physical_service_calls'],148000)
        self.assertEqual(a['physical_learning_calls'],6960)
        self.assertAlmostEqual(a['estimated_total_gpu_hours'],(154960/3600+1)*1.15)
    def test_cost_not_divided_by_cards(self):
        a=c.call_budget(2,400,5,37,5,696,1)
        self.assertAlmostEqual(a['inference_gpu_hours'],154960/3600)
    def test_numeric_bad_inputs(self):
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError): c.conditional_pair(n,.02,.1)
        with self.assertRaises(ValueError): c.budget(-1,0,0,0)
        with self.assertRaises(ValueError): c.call_budget(2,400,5,37,5,696,0)
    def test_scenario_no_observations(self):
        x=c.scenario_grid()
        self.assertFalse(x['observed_experiment_data'])
        self.assertFalse(x['final_confirmation_numbers_frozen'])
        self.assertIsNone(x['current_budget_snapshot'])
        self.assertEqual(len(x['conditional_binary_pairs']),64)
        json.dumps(x,allow_nan=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args()
    if args.receipt.exists(): raise SystemExit('Refuse overwrite')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={'schema':'mab2_confirmation_calculator_synthetic_tests_v1','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
            'real_experiment_data_read':False,
            'calculator_sha256':hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
