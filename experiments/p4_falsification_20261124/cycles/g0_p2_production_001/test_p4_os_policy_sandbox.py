from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from p4_online_runtime import run_policy_step


@unittest.skipUnless(os.name == "posix", "OS namespace probe runs on the server POSIX policy boundary")
class PolicySandboxTests(unittest.TestCase):
    def test_policy_cannot_read_host_tmp_sentinel_or_model_weights(self):
        bwrap = shutil.which("bwrap")
        self.assertIsNotNone(bwrap, "POSIX policy execution must fail closed without bubblewrap")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="mab2_p4_secret_", dir="/tmp", delete=False) as secret:
            secret.write("sealed-sentinel-value")
            secret_path = secret.name
        try:
            self.assertTrue(Path(secret_path).is_file(), "sentinel must exist in the host namespace before launch")
            model_path = "/home/chen/KDBPT/KD/models/Qwen-0.5B/model.safetensors"
            script = Path(__file__).with_name("sandbox_probe_worker_temp.py")
            script.write_text(
                "import json, pathlib, sys\n"
                "state=json.loads(sys.stdin.read())\n"
                f"sentinel=pathlib.Path({secret_path!r})\n"
                f"model=pathlib.Path({model_path!r})\n"
                "json.dump({'type':'abstain','sentinel_visible':sentinel.exists(),'model_visible':model.exists(),'public_state_seen':state.get('probe')=='public-only'},sys.stdout)\n",
                encoding="utf-8",
            )
            try:
                result = run_policy_step(script, {"probe": "public-only"}, timeout_seconds=10)
            finally:
                script.unlink(missing_ok=True)
            self.assertEqual(result["policy_isolation"], "BWRAP_USER_PID_NET_IPC_UTS_MOUNT")
            self.assertFalse(result["sentinel_visible"])
            self.assertFalse(result["model_visible"])
            self.assertTrue(result["public_state_seen"])
        finally:
            Path(secret_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
