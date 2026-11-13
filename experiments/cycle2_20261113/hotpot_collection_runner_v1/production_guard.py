"""New formal-attempt Linux guards; old qualification guards are unchanged."""
import ctypes
import math
import os
import signal
import sys
import time


def arm_before_exec(supervisor_pid,deadline_monotonic):
    if not sys.platform.startswith('linux'):raise RuntimeError('Production guard requires Linux')
    if type(supervisor_pid) is not int or supervisor_pid<=1 or not math.isfinite(deadline_monotonic):raise ValueError('Invalid parent/deadline')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot arm parent-death SIGKILL')
    if os.getppid()!=supervisor_pid:os._exit(125)
    remaining=deadline_monotonic-time.monotonic()
    if remaining<=0:os._exit(124)
    signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL,remaining)


def verify_child_guard(supervisor_pid,deadline_monotonic):
    if not sys.platform.startswith('linux'):raise RuntimeError('Production worker requires Linux')
    libc=ctypes.CDLL(None,use_errno=True);actual=ctypes.c_int()
    if os.getppid()!=supervisor_pid or libc.prctl(2,ctypes.byref(actual),0,0,0)!=0 or actual.value!=signal.SIGKILL:
        raise RuntimeError('Missing live owner or parent-death guard')
    remaining=deadline_monotonic-time.monotonic()
    if remaining<=0:os._exit(124)
    signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL,remaining)


def kill_owned_group(process,pgid,*,system=os):
    if process.poll() is not None:return False
    if pgid!=process.pid or pgid<=1:raise RuntimeError('Refuse unowned process group')
    try:actual=system.getpgid(process.pid)
    except ProcessLookupError:return False
    if actual!=pgid or pgid==system.getpgrp():raise RuntimeError('Process group identity changed; refuse broad kill')
    try:system.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError:return False
    return True
