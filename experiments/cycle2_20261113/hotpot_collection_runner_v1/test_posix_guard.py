"""Real CPU-only Linux process guards. Skipped on non-POSIX development hosts."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from functools import partial
from production_guard import arm_before_exec,kill_owned_group


@unittest.skipUnless(sys.platform.startswith('linux'),'Production guards require Linux; Windows test is not equivalent')
class PosixGuardTests(unittest.TestCase):
    def spawn(self,code,seconds):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        return subprocess.Popen([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True,
            preexec_fn=partial(arm_before_exec,os.getpid(),time.monotonic()+seconds))

    def test_real_deadline_alarm_survives_exec(self):
        child=self.spawn('import time; time.sleep(20)',.35)
        try:
            child.communicate(timeout=5)
            self.assertEqual(child.returncode,-signal.SIGALRM)
        finally:
            if child.poll() is None:kill_owned_group(child,child.pid);child.communicate(timeout=5)

    def test_real_parent_death_configuration_and_owned_group_kill(self):
        code=('import os,time; from production_guard import verify_child_guard; '
              'verify_child_guard(os.getppid(),time.monotonic()+10); print("ARMED",flush=True); time.sleep(20)')
        child=self.spawn(code,10)
        try:
            self.assertEqual(child.stdout.readline().strip(),b'ARMED')
            self.assertTrue(kill_owned_group(child,child.pid))
            child.communicate(timeout=5);self.assertEqual(child.returncode,-signal.SIGKILL)
        finally:
            if child.poll() is None:kill_owned_group(child,child.pid);child.communicate(timeout=5)


if __name__=='__main__':unittest.main()
