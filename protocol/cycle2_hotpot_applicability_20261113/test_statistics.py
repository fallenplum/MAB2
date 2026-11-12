"""Synthetic contract tests only. No files, question IDs, models or labels."""
import copy
import unittest
import numpy as np
import applicability_statistics as analysis


def audit():
    return dict(integrity_passed=True, unique_questions=384, complete_conditions=4224,
                runtime_failure_count=0, input_overflow_conditions=0,
                cost_frontier_review="NOT_DOMINATED")


def zeros():
    return dict(graphs=np.zeros((384, 2, 4)), no_edge=np.zeros(384),
                independent12=np.zeros(384), single=np.zeros(384))


EDGE = np.repeat([1, 5, 7, 9], 96)


class ContractTests(unittest.TestCase):
    def test_seedwise_product_not_product_of_grand_means(self):
        m = zeros()
        m["graphs"][0, 0, 1] = 1
        m["graphs"][0, 1, 0] = 1
        m["graphs"][1, 0, 0] = 1
        m["graphs"][1, 1, 1] = 1
        p, _ = analysis._question_statistics(m)
        self.assertEqual(p[:, 2].mean(), 0)
        self.assertEqual(p[:, 3].mean(), -2 / 384)

    def test_all_zero_drift_cannot_establish_no_effect(self):
        r = analysis.analyze(zeros(), zeros(), EDGE, audit())
        self.assertEqual(r["primary_EM"][analysis.NAMES[3]]["interval"], [0., 0.])
        self.assertEqual(r["decision"]["L2"], "INCONCLUSIVE")
        self.assertEqual(r["decision"]["theta_small_effect_exclusion"],
                         "NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT")
        self.assertEqual(r["bootstrap"]["primary_quantiles"], [.00625, .99375])

    def test_supported_sub_five_pp_effect_not_automatically_rejected(self):
        m = zeros()
        # Twenty positive and one negative quality pair, spread across edge strata.
        pos = np.concatenate([np.arange(k*96, k*96+5) for k in range(4)])
        m["graphs"][pos, :, 1] = 1
        m["no_edge"][95] = 1
        m["independent12"][95] = 1
        r = analysis.analyze(m, copy.deepcopy(m), EDGE, audit())
        self.assertLess(r["primary_EM"][analysis.NAMES[0]]["estimate"], .05)
        self.assertEqual(r["decision"]["L1"], "SUPPORTED_BOTH")
        self.assertEqual(r["decision"]["L2"], "GRAPH_CONDITIONAL_SIGNAL_SUPPORTED")
        self.assertEqual(r["decision"]["action"], "CONSIDER_INDEPENDENT_MECHANISM_PROTOCOL")
        self.assertFalse(r["decision"]["GPU_admission_granted"])
        a = audit(); a["input_overflow_conditions"] = 1
        blocked = analysis.analyze(m, copy.deepcopy(m), EDGE, a)
        self.assertEqual(blocked["decision"]["action"], "NO_AUTOMATIC_UPGRADE_CAPACITY_FAILURES")
        self.assertEqual(r["bootstrap"]["draw_index_sha256"], blocked["bootstrap"]["draw_index_sha256"])

    def test_f1_cannot_rescue_zero_em(self):
        f = zeros(); f["graphs"][::2, :, 1] = 1
        r = analysis.analyze(zeros(), f, EDGE, audit())
        self.assertEqual(r["decision"]["L1"], "INCONCLUSIVE")
        self.assertGreater(r["secondary_descriptive_only"]["F1_main_forms"]
                           ["B_MINUS_NO_EDGE_F1"]["estimate"], 0)

    def test_within_stratum_constants_cannot_get_false_bootstrap_precision(self):
        m = zeros(); m["graphs"][:96, :, 1] = 1
        r = analysis.analyze(m, copy.deepcopy(m), EDGE, audit())
        for name in analysis.NAMES:
            item = r["primary_EM"][name]
            self.assertGreater(item["sample_variance"], 0)
            self.assertEqual(item["interval"], [.25, .25])
            self.assertFalse(item["bootstrap_nondegenerate"])
            self.assertFalse(item["support_guard_pass"])
        self.assertEqual(r["decision"]["L1"], "INCONCLUSIVE")
        self.assertEqual(r["decision"]["L2"], "INCONCLUSIVE")

    def test_incomplete_and_runtime_failed_archives_rejected(self):
        for field, value in (("complete_conditions", 4223), ("runtime_failure_count", 1),
                             ("unique_questions", 383), ("integrity_passed", False)):
            a = audit(); a[field] = value
            with self.assertRaises(ValueError):
                analysis.analyze(zeros(), zeros(), EDGE, a)

    def test_bad_edge_or_nonbinary_em_rejected(self):
        m = zeros(); m["graphs"][0, 0, 1] = .5
        with self.assertRaises(ValueError):
            analysis.analyze(m, zeros(), EDGE, audit())
        bad = EDGE.copy(); bad[0] = 5
        with self.assertRaises(ValueError):
            analysis.analyze(zeros(), zeros(), bad, audit())
        with self.assertRaises(ValueError):
            analysis.analyze(zeros(), zeros(), EDGE[:-1], audit())

    def test_reversed_signed_mechanism_is_not_positive_redundancy(self):
        m = zeros()
        m["graphs"][::2, :, 0] = 1  # gamma_A = 1
        m["graphs"][::2, :, 1] = 1
        m["graphs"][::2, :, 3] = 1  # gamma_B = 0
        r = analysis.analyze(m, copy.deepcopy(m), EDGE, audit())
        self.assertEqual(r["decision"]["signed_direction"], "REDUNDANCY_DIRECTION_REVERSED")
        self.assertTrue(r["decision"]["theta_nonzero_supported"])


if __name__ == "__main__":
    unittest.main()
