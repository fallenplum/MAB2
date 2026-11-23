from pathlib import Path
import unittest
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conditional_real_score import decision_from_history
from m60_strong_baseline_matrix import _row_refresh_prediction, oracle_row_refresh

class PermissionOracleTest(unittest.TestCase):
  def test_stale_and_oracle_paths_do_not_return_target_directly(self):
    source = (HERE / "conditional_real_score.py").read_text(encoding="utf-8")
    matrix = (HERE / "m60_strong_baseline_matrix.py").read_text(encoding="utf-8")
    self.assertIn('decision_from_history(old_source, base, history, context="stale")', matrix)
    self.assertIn('"oracle_prediction": "independent_row_refresh_replay"', matrix)
    self.assertNotIn('"oracle_row_refresh": (set(target)', matrix)
    self.assertIn('return decision_set(audit, context=context)', source)

  def test_post_edit_replay_uses_new_relay_values(self):
    matrix = (HERE / "m60_strong_baseline_matrix.py").read_text(encoding="utf-8")
    scorer = (HERE / "conditional_real_score.py").read_text(encoding="utf-8")
    self.assertIn('"reweighted_old_history": optional_controller(lambda: replay(old_source, new_source, base, new_relay', matrix)
    self.assertIn('"source_only_ctdr": replay(old_source, new_source, base, new_relay', scorer)


  def test_oracle_row_refresh_prediction_rejects_unfinished_plan(self):
    source = np.asarray([0.5, 0.5, 0.0, 0.0])
    base = np.full(4, 0.25)
    relay = np.tile(base, (4, 1))
    labels = np.arange(16, dtype=int).reshape(4, 4) % 2
    self.assertIsNone(_row_refresh_prediction(source, base, relay, [], [(0, None)]))


  def test_oracle_entrypoint_unpacks_warm_cache_pair(self):
    """The full oracle entrypoint must run past warm-cache setup."""
    source = np.asarray([0.25, 0.25, 0.25, 0.25])
    base = np.full(4, 0.25)
    relay = np.tile(base, (4, 1))
    labels = np.arange(16, dtype=int).reshape(4, 4) % 2
    tokens = np.ones(27, dtype=int)
    rows, spent, lower_bound = oracle_row_refresh(
      source, base, relay, labels, [], [0], tokens, set()
    )
    self.assertIsInstance(rows, tuple)
    self.assertIsInstance(spent, int)
    self.assertIsInstance(lower_bound, int)


  def test_oracle_entrypoint_handles_no_changed_rows(self):
    source = np.asarray([0.25, 0.25, 0.25, 0.25])
    base = np.full(4, 0.25)
    relay = np.tile(base, (4, 1))
    labels = np.arange(16, dtype=int).reshape(4, 4) % 2
    tokens = np.ones(27, dtype=int)
    rows, spent, lower_bound = oracle_row_refresh(
      source, base, relay, labels, [], [], tokens, set()
    )
    self.assertEqual(rows, ())
    self.assertEqual(spent, 0)
    self.assertEqual(lower_bound, 0)


  def test_decision_from_history_rejects_unfinished_transcript(self):
    with self.assertRaisesRegex(RuntimeError, "unfinished audit"):
      decision_from_history(np.full(4, 0.25), np.full(4, 0.25), [], context="test")


if __name__ == "__main__":
  unittest.main()
