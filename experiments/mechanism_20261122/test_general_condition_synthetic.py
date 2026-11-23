from __future__ import annotations

import unittest

from general_channel_synthetic import run


class GeneralConditionSyntheticTest(unittest.TestCase):
    def test_source_only_degenerates_to_zero_query_transport(self) -> None:
        result = run(seed=11, n_per_cell=30)
        source = next(cell for cell in result["cells"] if cell["drift_type"] == "source_only")
        self.assertEqual(source["methods"]["ctdr_transport"]["mean_queries"], 0.0)
        self.assertEqual(source["methods"]["ctdr_transport"]["mean_tokens"], 0.0)

    def test_conditional_drift_requires_some_revalidation(self) -> None:
        result = run(seed=13, n_per_cell=100)
        conditional = next(cell for cell in result["cells"] if cell["drift_type"] == "conditional")
        self.assertGreater(conditional["methods"]["ctdr_transport"]["mean_queries"], 0.0)
        self.assertEqual(conditional["methods"]["full_restart"]["action_error"], 0.0)


if __name__ == "__main__":
    unittest.main()

