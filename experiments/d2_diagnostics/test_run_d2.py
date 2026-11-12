"""Independent synthetic checks of D2 decisions/noise. Never calls execute/read_inputs."""

import itertools
import unittest

import numpy as np

import run_d2 as run


class AdapterDecisionTests(unittest.TestCase):
    def test_action_sign_add_delete_stay_strict_zero_and_lower_edge_tie(self):
        pred = np.zeros((5, 2, 12))
        bits = np.zeros((5, 2, 12), dtype=int)
        targets = np.zeros((5, 2, 12, 5))
        # Addition: positive conditional credit for an absent edge.
        pred[0] = -.1; pred[0, :, 3] = .5; targets[0, :, 3] = .4
        # Deletion: negative conditional credit for a present edge.
        bits[1] = 1; pred[1] = .1; pred[1, :, 5] = -.6; targets[1, :, 5] = -.25
        # Zero predictions must strictly stay, even if realized labels favor edits.
        targets[2] = 1
        # All signed predicted edit gains are negative.
        pred[3] = -.2
        # Equal positive maxima use the lower edge index, not observed labels.
        pred[4, :, [2, 7]] = .9
        targets[4, :, 2] = -.2; targets[4, :, 7] = 1
        gain, actions, uniform = run.action_values(pred, targets, bits)
        np.testing.assert_array_equal(actions, np.repeat([[3], [5], [12], [12], [2]], 2, axis=1))
        np.testing.assert_allclose(gain, [.4, .25, 0, 0, -.2], atol=1e-15)
        np.testing.assert_allclose(uniform, [.4/13, .25/13, 12/13, 0, .8/13], atol=1e-15)
        metrics, _ = run.metrics(pred, targets, bits)
        self.assertEqual(metrics['addition_count'], 4)
        self.assertEqual(metrics['deletion_count'], 2)
        self.assertEqual(metrics['stay_count'], 4)

    def test_signed_max_uses_presence_for_each_edge(self):
        pred = np.zeros((1, 2, 12)); bits = np.zeros((1, 2, 12), dtype=int)
        targets = np.zeros((1, 2, 12, 5))
        # The largest positive raw credit is an existing edge, hence not an add.
        pred[:, :, 0] = 10; bits[:, :, 0] = 1
        pred[:, :, 4] = -2; bits[:, :, 4] = 1; targets[:, :, 4] = -1
        pred[:, :, 7] = 1; targets[:, :, 7] = .3
        gain, selected, _ = run.action_values(pred, targets, bits)
        np.testing.assert_array_equal(selected, [[4, 4]])
        np.testing.assert_allclose(gain, [1])

    @staticmethod
    def result(gain, mse):
        return {'tune': {'mean_action_gain': gain, 'mean_credit_mse': mse}}

    def test_nomination_excludes_r0_and_uses_gain_then_mse(self):
        results = {'C1_R0': self.result(100, 0), 'C2_R0': self.result(101, 0),
                   'C1_R1': self.result(.1, .02), 'C2_R1': self.result(.1, .01),
                   'C1_R2': self.result(.09, 0)}
        protocol = {'candidate_nomination': {'eligible': ['C1_R1', 'C2_R1', 'C1_R2']}}
        self.assertEqual(run.choose_nominee(results, protocol), 'C2_R1')

    def test_nomination_tolerances_then_lexical_tie(self):
        results = {'C1_R1': self.result(.1, .02),
                   'C1_R1b': self.result(.1+5e-13, .03),
                   'C1_R2': self.result(.1-2e-12, .001),
                   'C2_R1': self.result(.1, .02-5e-13)}
        protocol = {'candidate_nomination': {'eligible': list(results)}}
        # R1b's gain advantage is inside tolerance; R2 is outside gain tolerance.
        # C2_R1's MSE advantage is inside tolerance, leaving lexical C1_R1.
        self.assertEqual(run.choose_nominee(results, protocol), 'C1_R1')


class SeedNoiseTests(unittest.TestCase):
    @staticmethod
    def manual_loso(targets, bits):
        qn, anchors, edges, seeds = targets.shape
        per_question = np.zeros(qn)
        # Scalar, independently written action loop to detect array-axis leakage.
        for q in range(qn):
            held_values = []
            for held in range(seeds):
                training = [s for s in range(seeds) if s != held]
                for a in range(anchors):
                    best_gain, best_edge = 0., None
                    for e in range(edges):
                        sign = 1-2*int(bits[q, a, e])
                        estimated = sign*sum(float(targets[q, a, e, s]) for s in training)/len(training)
                        if estimated > best_gain:
                            best_gain, best_edge = estimated, e
                    held_values.append(0. if best_edge is None else
                                       (1-2*int(bits[q, a, best_edge]))*targets[q, a, best_edge, held])
            per_question[q] = np.mean(held_values)
        return per_question

    def test_loso_excludes_heldout_seed_from_selection(self):
        targets = np.zeros((2, 2, 12, 5))
        bits = np.zeros((2, 2, 12), dtype=int)
        # Each seed uniquely rewards a different edge. Selecting from all five
        # yields optimistic .2; selecting from the other four yields heldout 0.
        for s in range(5):
            targets[:, :, s, s] = 1
        report = run.noise_report(targets, bits)
        self.assertAlmostEqual(report['naive_observed_menu_oracle']['estimate'], .2)
        self.assertEqual(report['leave_one_seed_out_selection_gain']['estimate'], 0.)
        self.assertFalse(report['five_folds_independent'])
        self.assertFalse(report['true_oracle_claimed'])
        # Inference unit remains a question, not q*5 dependent seed folds.
        self.assertEqual(report['leave_one_seed_out_selection_gain']['question_count'], 2)

    def test_loso_signed_actions_match_independent_scalar_reference(self):
        rng = np.random.default_rng(241)
        targets = rng.integers(-1, 2, size=(4, 2, 12, 5)).astype(float)
        bits = rng.integers(0, 2, size=(4, 2, 12))
        expected = self.manual_loso(targets, bits)
        report = run.noise_report(targets, bits)
        self.assertAlmostEqual(report['leave_one_seed_out_selection_gain']['estimate'], expected.mean(), places=14)
        self.assertEqual(report['leave_one_seed_out_selection_gain']['samples'], 10000)
        self.assertEqual(report['leave_one_seed_out_selection_gain']['multiplicity'], 'unadjusted')

    def test_loso_zero_train_mean_strictly_stays(self):
        targets = np.zeros((1, 2, 12, 5)); bits = np.zeros((1, 2, 12), dtype=int)
        # Only edge zero has feedback: three seeds zero, one +1, one -1.
        targets[:, :, 0, 3] = 1; targets[:, :, 0, 4] = -1
        # For heldout -1, remaining +1 induces a bad add; for heldout +1,
        # remaining -1 means stay. Other heldout seeds give exactly zero -> stay.
        report = run.noise_report(targets, bits)
        self.assertAlmostEqual(report['leave_one_seed_out_selection_gain']['estimate'], -.2)
        self.assertEqual(report['naive_observed_menu_oracle']['estimate'], 0.)

    def test_crossseed_square_noise_identity(self):
        rng = np.random.default_rng(242)
        target = rng.integers(-1, 2, size=(3, 2, 12, 5)).astype(float)
        bits = rng.integers(0, 2, size=(3, 2, 12))
        # Explicit off-diagonal products, independent of sum-of-squares formula.
        cross = np.mean([target[..., i]*target[..., j] for i, j in itertools.combinations(range(5), 2)], axis=0)
        mean_sq = target.mean(axis=-1)**2
        noise = target.var(axis=-1, ddof=1)/5
        np.testing.assert_allclose(mean_sq-noise, cross, atol=2e-16)
        report = run.noise_report(target, bits)
        self.assertAlmostEqual(report['mean_cross_seed_credit_squared'], cross.mean(), places=14)
        self.assertAlmostEqual(report['observed_mean_squared'], mean_sq.mean(), places=14)
        self.assertLess(report['noise_identity_max_abs'], 2e-16)

    def test_crossseed_prediction_risk_is_mse_minus_seed_noise(self):
        rng = np.random.default_rng(243)
        target = rng.integers(-1, 2, size=(3, 2, 12, 5)).astype(float)
        pred = rng.normal(size=(3, 2, 12))
        bits = rng.integers(0, 2, size=(3, 2, 12))
        errors = pred[..., None]-target
        explicit_risk = np.mean([errors[..., i]*errors[..., j] for i, j in itertools.combinations(range(5), 2)])
        expected = ((pred-target.mean(axis=-1))**2).mean()-target.var(axis=-1, ddof=1).mean()/5
        report, _ = run.metrics(pred, target, bits)
        self.assertAlmostEqual(report['mean_credit_risk'], explicit_risk, places=13)
        self.assertAlmostEqual(report['mean_credit_risk'], expected, places=13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
