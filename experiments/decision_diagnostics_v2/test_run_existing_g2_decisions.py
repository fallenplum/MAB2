import unittest
import io
import hashlib
import numpy as np

from run_existing_g2_decisions import prepare_arrays, ridge_direction_leverage, decode_frozen_features


class AdapterRegressionTests(unittest.TestCase):
    def test_frozen_feature_bytes_are_checked_before_decoding(self):
        with self.assertRaisesRegex(AssertionError, "PCA feature bytes"):
            decode_frozen_features(b"not an npz", "0" * 64)

    def test_feature_decoder_uses_the_exact_verified_buffer(self):
        stream = io.BytesIO()
        np.savez(stream, question_ids=np.array(["synthetic"]), phi=np.array([[1., 2., 1.]]))
        raw = stream.getvalue()
        result = decode_frozen_features(raw, hashlib.sha256(raw).hexdigest())
        np.testing.assert_array_equal(result["synthetic"], [1., 2., 1.])

    def test_weighted_ridge_geometry_matches_direct_inverse(self):
        matrix = np.array([[1., 2.], [2., -1.], [0., 3.]])
        weights = np.array([.25, .75, 1.])
        directions = np.array([[1., 0.], [0., 1.], [-2., 3.], [0., 0.]])
        alpha = 2.
        gram = matrix.T @ np.diag(weights) @ matrix + alpha * np.eye(2)
        expected = np.sqrt(np.diag(directions @ np.linalg.inv(gram) @ directions.T))
        np.testing.assert_allclose(ridge_direction_leverage(matrix, weights, directions, alpha), expected, atol=1e-14)

    def test_zero_training_design_still_penalizes_constant_direction(self):
        value = ridge_direction_leverage(np.zeros((3, 2)), np.ones(3), np.array([[3., 4.]]), 4.)
        np.testing.assert_array_equal(value, [2.5])

    def test_array_mapping_retains_anchor_and_model_axes(self):
        qids = [f"synthetic_{q:03}" for q in range(120)]
        designs, predictions, coordinates = {}, [], []
        for q, qid in enumerate(qids):
            designs[qid] = {"anchors": [{"anchor_id": "a", "mask": [0] * 12},
                                          {"anchor_id": "b", "mask": [1] * 12}]}
            for anchor, family in enumerate(["a", "b"]):
                for edge in range(12):
                    value = q + anchor / 10 + edge / 1000
                    predictions.append({"question_id": qid, "anchor_id": family, "edge": edge,
                                        "predictions": {"B0": value, "C1": value + 1, "C2": value + 2}})
                    coordinates.append({"qid": qid, "family": family, "edge": edge, "targets": [anchor] * 5})
        context = {"plan": {"partitions": dict.fromkeys(qids, "test"), "designs": designs},
                   "families": ["b", "a"], "predictions": list(reversed(predictions))}
        actual_qids, families, fitted, targets, bits = prepare_arrays(context, list(reversed(coordinates)))
        self.assertEqual(actual_qids, qids)
        self.assertEqual(families, ["a", "b"])
        np.testing.assert_allclose(fitted[119, 1, 11], [119.111, 120.111, 121.111])
        np.testing.assert_array_equal(targets[:, 0], np.zeros((120, 12, 5)))
        np.testing.assert_array_equal(targets[:, 1], np.ones((120, 12, 5)))
        np.testing.assert_array_equal(bits[:, 0], np.zeros((120, 12)))
        np.testing.assert_array_equal(bits[:, 1], np.ones((120, 12)))


if __name__ == "__main__":
    unittest.main()
