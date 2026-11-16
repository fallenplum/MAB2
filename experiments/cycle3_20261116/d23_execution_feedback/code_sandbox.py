"""Linux-only fail-closed STDIO evaluator; no dataset solution execution on host."""
import argparse,hashlib,json,os,signal,subprocess,tempfile,time
from pathlib import Path

HARNESS=r'''
import ctypes,errno,os,resource,sys
source=open('/candidate.py',encoding='utf-8').read()
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
resource.setrlimit(resource.RLIMIT_NPROC,(1,1))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=lib.seccomp_init(0x7fff0000)
if not ctx:raise RuntimeError('seccomp init failed')
for name in ['clone','clone3','fork','vfork','execve','execveat','unshare','setns','ptrace','mount','umount2','pivot_root','chroot','keyctl','bpf','io_uring_setup','open_by_handle_at','process_vm_readv','process_vm_writev','kexec_load','reboot','swapon','swapoff']:
 n=lib.seccomp_syscall_resolve_name(name.encode())
 if n>=0 and lib.seccomp_rule_add(ctx,0x50000|errno.EPERM,n,0)!=0:raise RuntimeError('seccomp rule failed')
if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp load failed')
lib.seccomp_release(ctx)
sys.argv=['candidate.py']
exec(compile(source,'candidate.py','exec'),{'__name__':'__main__','__file__':'/candidate.py'})
'''

def run_code(source,stdin,root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 assert isinstance(source,str) and len(source.encode())<=131072 and len(stdin.encode())<=1048576
 with tempfile.TemporaryDirectory(prefix='case-',dir=root) as tmp:
  p=Path(tmp);(p/'candidate.py').write_text(source,encoding='utf-8');(p/'harness.py').write_text(HARNESS,encoding='utf-8')
  cmd=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--dir','/usr','--dir','/usr/bin','--dir','/usr/lib','--ro-bind','/usr/bin/python3.8','/usr/bin/python3.8','--ro-bind','/usr/lib/python3.8','/usr/lib/python3.8','--ro-bind','/usr/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu','--ro-bind','/usr/lib64','/usr/lib64','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache','--proc','/proc','--dev','/dev','--dir','/tmp','--ro-bind',str(p/'candidate.py'),'/candidate.py','--ro-bind',str(p/'harness.py'),'/harness.py','--chdir','/tmp','--remount-ro','/','/usr/bin/python3.8','-I','-B','/harness.py']
  # Device/proc submounts must also be read-only, not just the root mount.
  pos=cmd.index('--remount-ro');cmd[pos:pos]=['--remount-ro','/dev','--remount-ro','/proc']
  with (p/'stdout').open('w+b') as out,(p/'stderr').open('w+b') as err:
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,env={'PATH':'/usr/bin','LANG':'C.UTF-8'},start_new_session=True)
   timeout=False
   try:proc.communicate(stdin.encode(),timeout=4)
   except subprocess.TimeoutExpired:
    timeout=True
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.communicate()
   out.seek(0);err.seek(0);stdout=out.read(65537);stderr=err.read(65537)
  status='TIMEOUT' if timeout else ('OUTPUT_LIMIT' if len(stdout)>65536 or len(stderr)>65536 else ('OK' if proc.returncode==0 else 'RUNTIME_ERROR'))
  return dict(status=status,returncode=proc.returncode,stdout=stdout[:65536].decode('utf-8','replace'),stderr=stderr[:65536].decode('utf-8','replace'),elapsed_seconds=time.monotonic()-start)

def qualify(root):
 tests=[('stdio','import sys;print(sum(map(int,sys.stdin.read().split())))','2 3\n','5\n','OK'),
 ('filesystem',"import os;assert not os.path.exists('/data1');assert not os.path.exists('/home');assert not os.path.exists('/etc/passwd');assert not os.path.exists('/proc/1/root/data1');print('ISOLATED')",'', 'ISOLATED\n','OK'),
 ('environment',"import os;assert set(os.environ)<=set(['PATH','LANG','LC_CTYPE','PWD']);print('CLEAN')",'','CLEAN\n','OK'),
 ('network',"import socket\ns=socket.socket();s.settimeout(.2)\ntry:s.connect(('192.0.2.1',80));raise AssertionError('network permitted')\nexcept OSError:print('NETWORK_BLOCKED')",'','NETWORK_BLOCKED\n','OK'),
 ('process',"import os\ntry:os.fork();raise AssertionError('fork permitted')\nexcept OSError:print('FORK_BLOCKED')",'','FORK_BLOCKED\n','OK'),
 ('readonly',"try:open('/tmp/new','w');raise AssertionError('write permitted')\nexcept OSError:print('READONLY')",'','READONLY\n','OK'),
 ('stdlib','import collections,math,itertools,bisect,heapq,json,sys;print(math.isqrt(81))','','9\n','OK'),
 ('memory',"a=bytearray(800*1024**2)",'',None,'RUNTIME_ERROR'),
 ('cpu','while True:pass','',None,'RUNTIME_ERROR'),
 ('wall','import time;time.sleep(20)','',None,'TIMEOUT'),
 ('output',"print('x'*2000000)",'',None,'OUTPUT_LIMIT')]
 results=[]
 for name,source,stdin,expected,status in tests:
  r=run_code(source,stdin,root);r.update(test=name,pass_check=r['status']==status and (expected is None or r['stdout']==expected));results.append(r)
 return dict(status='PASS_ARTIFICIAL_EXECUTION_ISOLATION' if all(r['pass_check'] for r in results) else 'FAIL_ISOLATION',tests=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_calls=0,GPU_hours=0,writable_sandbox_paths=[],python='system3.8',limits=dict(cpu_seconds=2,wall_seconds=4,virtual_memory_MiB=384,processes=1,output_file_MiB=1,retained_output_KiB=64),scope='STDIO Python standard library only; no host project mounts; all namespaces unshared; seccomp blocks child processes/exec and namespace changes; no claim against kernel vulnerabilities')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--qualify-root',required=True);a.add_argument('--output',required=True);a=a.parse_args();result=qualify(a.qualify_root)
 with open(a.output,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='tests'}));print(json.dumps([dict(test=r['test'],passed=r['pass_check'],status=r['status'],stderr=r['stderr'][:150]) for r in result['tests']]))
