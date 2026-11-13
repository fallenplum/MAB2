"""Linux single-process GPU child guards; no new privileges or cgroups.

Sources: Linux man-pages PR_SET_PDEATHSIG(2const), setitimer(2). This covers the
direct model process and its threads, not independently spawned GPU descendants.
The supported qualification backend starts no extra model processes.
"""
import ctypes
import math
import os
import signal
import time


def _linux():
    import sys
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Parent-death supervision requires Linux')
    return ctypes.CDLL(None,use_errno=True)


def arm_child_deadline(supervisor_pid: int,deadline_monotonic: float):
    libc=_linux()
    if type(supervisor_pid) is not int or supervisor_pid<=1 or os.getppid()!=supervisor_pid:
        raise RuntimeError('Missing or mismatched live qualification supervisor')
    actual=ctypes.c_int()
    if libc.prctl(2,ctypes.byref(actual),0,0,0)!=0 or actual.value!=signal.SIGKILL:
        raise RuntimeError('Parent-death SIGKILL is not armed')
    if not math.isfinite(deadline_monotonic):raise ValueError('Finite monotonic deadline required')
    remaining=deadline_monotonic-time.monotonic()
    if remaining<=0:os._exit(124)
    if remaining>1760.5:raise ValueError('Child deadline exceeds frozen resource window')
    signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL,remaining)


def guard_before_exec(supervisor_pid: int,deadline_monotonic: float):
    libc=_linux()
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot arm parent-death SIGKILL')
    # Binding after parent death would bind the reaper instead. Compare against
    # the captured pre-fork supervisor PID after prctl to close that race.
    if os.getppid()!=supervisor_pid:os._exit(125)
    arm_child_deadline(supervisor_pid,deadline_monotonic)
