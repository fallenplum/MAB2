import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import path_runtime_v3 as m
def plan():return {'mode':'formal','source_pins':{},'gpus':[2,3,4,5],'prefix_per_worker':2,
 'worker_cap_seconds':6750,'assignments':{str(i):list(range(i,96,4)) for i in range(4)},
 'seeds':[26111311,26111312],'minimum_gpu_free_MiB':40000,'minimum_disk_bytes':1}
class Tests(unittest.TestCase):
 def test_fixed_coverage(self):
  m.verify(plan())
  for mutate in [lambda p:p['gpus'].__setitem__(0,3),lambda p:p.update(worker_cap_seconds=6751),
   lambda p:p.update(prefix_per_worker=3),lambda p:p['assignments']['0'].__setitem__(0,1),lambda p:p.update(seeds=[1,2])]:
   p=plan();mutate(p)
   with self.assertRaises(ValueError):m.verify(p)
 def test_partial_launch_cleans_owned_child(self):
  d=Path(tempfile.mkdtemp(prefix='path_guard_',dir=Path(__file__).parent));p=plan();p['output']=str(d/'out')
  f=d/'PLAN.json';f.write_text(json.dumps(p),encoding='utf-8')
  class Child:
   pid=123456789;returncode=None
   def poll(self):return self.returncode
   def wait(self,timeout=None):self.returncode=-15;return self.returncode
  child=Child();signals=[]
  with patch.object(m.subprocess,'check_output',return_value='2, 48000, 0\n3, 48000, 0\n4, 48000, 0\n5, 48000, 0'),\
       patch.object(m.subprocess,'Popen',side_effect=[child,OSError('artificial launch failure')]),\
       patch.object(m.os,'killpg',side_effect=lambda pid,sig:signals.append(pid),create=True):
   with self.assertRaises(OSError):m.supervisor(str(f),m.sha(f))
  closure=m.read(d/'out/CLOSURE.json')
  self.assertEqual(signals,[child.pid]);self.assertTrue(closure['all_workers_exited'])
  self.assertEqual(len(closure['residence_seconds_by_worker']),1)
  self.assertTrue((d/'out/ARCHIVE_MANIFEST.json').exists())
if __name__=='__main__':unittest.main()
