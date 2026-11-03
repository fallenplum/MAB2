import copy
import unittest
from test_watch_g1 import fixture, watch


class ElasticMonitor(unittest.TestCase):
    def running(self):
        data = fixture()
        data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
        data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
        return data

    def test_eighth_worker_is_monitored_independently_of_three_logical_shards(self):
        data = self.running()
        data["remote"]["processes"] = [{"role": "child", "shard_index": 7, "pid": 123,
                                         "scheduler_worker": True, "completed": False,
                                         "alive": False, "matches_project": True}]
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_completed_elastic_worker_may_exit_without_false_failure(self):
        data = self.running()
        data["remote"]["processes"] = [{"role": "child", "shard_index": 0, "pid": 123,
                                         "scheduler_worker": True, "completed": True,
                                         "alive": False, "matches_project": False}]
        self.assertEqual(watch.classify(data)["state"], "RUNNING")

    def test_active_elastic_worker_identity_mismatch_is_not_ignored(self):
        data = self.running()
        data["remote"]["processes"] = [{"role": "child", "shard_index": 5, "pid": 123,
                                         "scheduler_worker": True, "completed": False,
                                         "alive": True, "matches_project": False}]
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_remote_probe_remains_valid_python(self):
        compile(watch.REMOTE_PROBE, "remote_probe", "exec")


if __name__ == "__main__":
    unittest.main()
