from pathlib import Path
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import local_archive_transport as transport


class LocalArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.spool = self.dir / "spool"
        self.spool.mkdir()
        self.archive = self.dir / "archive"
        self.archive.mkdir()
        self.plan = {"output": str(self.spool), "collector_run_id": "test", "minimum_disk_bytes": 8 * 1024**3}
        self.lease = self.dir / "spool.collector.json"

    def lease_write(self, capacity, run_id="test"):
        self.lease.write_text(json.dumps({"run_id": run_id, "local_free_bytes": capacity}), encoding="utf-8")

    def test_lease(self):
        self.assertFalse(transport.ready(self.plan, self.spool))
        self.lease_write(12 * 1024**3)
        self.assertTrue(transport.ready(self.plan, self.spool))
        old = time.time() - transport.LEASE_SECONDS - 5
        os.utime(self.lease, (old, old))
        self.assertFalse(transport.ready(self.plan, self.spool))
        self.lease_write(1)
        self.assertFalse(transport.ready(self.plan, self.spool))
        self.lease_write(12 * 1024**3, "other")
        self.assertFalse(transport.ready(self.plan, self.spool))
        self.lease_write(12 * 1024**3)
        with patch.object(transport.shutil, "disk_usage") as space:
            space.return_value.free = 100
            self.assertFalse(transport.ready(self.plan, self.spool))

    def test_inflight_append_and_close(self):
        f = self.spool / "requests.jsonl"
        f.write_bytes(b"start\n")
        first = transport.inventory(self.spool)
        f.write_bytes(b"start\nreturned\n")
        prior = {}
        transport.pull(self.archive, str(self.spool), first, prior, open)
        self.assertEqual((self.archive / f.name).read_bytes(), b"start\n")
        transport.pull(self.archive, str(self.spool), transport.inventory(self.spool), prior, open)
        self.assertEqual((self.archive / f.name).read_bytes(), f.read_bytes())

    def test_bad_hash_and_path(self):
        (self.spool / "a").write_bytes(b"x")
        with self.assertRaises(ValueError):
            transport.pull(self.archive, str(self.spool), {"a": [1, "0" * 64]}, {}, open)
        self.assertFalse((self.archive / "a").exists())
        with self.assertRaises(ValueError):
            transport.pull(self.archive, str(self.spool), {"../escape": [1, "0" * 64]}, {}, open)


if __name__ == "__main__":
    unittest.main()
