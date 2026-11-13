"""Small arithmetic counterexamples, entirely artificial; no data access."""
import unittest
import numpy as np
import independent_statistics_review as review


class IndependentArithmeticTests(unittest.TestCase):
    def test_credit_direction_and_within_question_product(self):
        metric = {
            'graphs': [[[0, 1, 0, 0], [1, 0, 0, 0]],
                       [[1, 1, 0, 0], [0, 1, 0, 0]]],
            'no_edge': [0, 1], 'independent12': [1, 0], 'single': [1, 0],
        }
        actual = review.form_columns(metric)
        np.testing.assert_array_equal(actual[:, :4], [[1, 0, 0, -1], [0, 1, .5, 0]])
        self.assertEqual(actual[:, 3].mean(), -.5)
        self.assertEqual(actual[:, 9].mean() * actual[:, 10].mean(), 0)

    def test_fixed_strata_frequency_resampling(self):
        edges = np.repeat(review.EDGES, 96)
        weights, sha = review.resampling_weights(edges, repetitions=3)
        self.assertEqual(len(sha), 64)
        self.assertTrue((weights.sum(axis=1) == 384).all())
        self.assertTrue(all((weights[:, edges == e].sum(axis=1) == 96).all() for e in review.EDGES))
        value = np.tile(np.arange(96), 4)
        manual_draws = np.random.Generator(np.random.PCG64(261113103)).integers(0, 96, size=(3, 4, 96))
        np.testing.assert_array_equal(weights @ value / 384, manual_draws.mean(axis=(1, 2)))

    def test_pooled_variation_cannot_override_degenerate_bootstrap(self):
        values = np.repeat([0., 1., 2., 3.], 96)
        result = review.summarize(values, np.repeat(1.5, 100), (.00625, .99375))
        self.assertGreater(result['sample_variance'], 0)
        self.assertFalse(result['bootstrap_nondegenerate'])
        self.assertFalse(result['support_guard_pass'])

    def test_support_count_and_negative_theta_are_not_rescued(self):
        values = np.zeros(384)
        values[:19] = 1
        s = review.summarize(values, np.linspace(.01, .09, 100), (.00625, .99375))
        self.assertFalse(s['support_guard_pass'])
        base = {'estimate': 0., 'interval': [-.1, .1], 'interval_width': .2,
                'support_guard_pass': True}
        primary = {k: dict(base) for k in review.PRIMARY}
        primary[review.PRIMARY[0]].update(estimate=-.2, interval=[-.3, -.1])
        primary[review.PRIMARY[3]].update(estimate=-.03, interval=[-.1, .03])
        result = review.classify(primary, 0, 'NOT_DOMINATED')
        self.assertEqual(result['L1'], 'B_DISFAVORED')
        self.assertEqual(result['L2'], 'INCONCLUSIVE')
        self.assertEqual(result['action'], 'STOP_CURRENT_B_DIRECTION')
        self.assertTrue(result['theta_negative_point_retained'])
        self.assertFalse(result['GPU_admission_granted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
