"""CPU-only Linux watchdog primitive probe. No SSH, CUDA, model, or data access.

Run with: python -B linux_cpu_watchdog_probe.py
Prints a JSON receipt to stdout. Does not certify any deployment-source version.
Only subprocesses created by this invocation are signalled. Every helper has a
short independent default SIGALRM limit to bound cleanup after test failure.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback

SELF = str(Path(__file__).resolve())
LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform.startswith('linux') else None
PRODUCTION_GUARD = None
GUARD_FILE = None


def load_guard(path):
    global PRODUCTION_GUARD, GUARD_FILE
    GUARD_FILE = str(Path(path).resolve(strict=True))
    spec = importlib.util.spec_from_file_location('audited_gpu_job_guard', GUARD_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(callable(getattr(module, name, None)) for name in ('guard_before_exec', 'arm_child_deadline')):
        raise ValueError('Production guard interface missing')
    PRODUCTION_GUARD = module


def arm_deadline(deadline, expected_parent):
    if PRODUCTION_GUARD is not None:
        return PRODUCTION_GUARD.arm_child_deadline(expected_parent, deadline)
    # A real default action is required: no Python-level callback is involved.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        os._exit(124)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    if os.getppid() != expected_parent:
        os._exit(125)


def guard(expected_parent, deadline):
    def preexec():
        if PRODUCTION_GUARD is not None:
            return PRODUCTION_GUARD.guard_before_exec(expected_parent, deadline)
        if LIBC.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            os._exit(126)
        if os.getppid() != expected_parent:  # closes fork/prctl parent-death race
            os._exit(125)
        arm_deadline(deadline, expected_parent)
    return preexec


def launch(mode, seconds=8, *, expected_parent=None, new_session=True):
    expected = os.getpid() if expected_parent is None else expected_parent
    deadline = time.monotonic() + seconds
    extra = ['--guard-file', GUARD_FILE] if GUARD_FILE is not None else []
    return subprocess.Popen(
        [sys.executable, '-B', SELF, '--helper', mode, '--deadline', str(deadline),
         '--expected-parent', str(expected)] + extra,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=new_session, preexec_fn=guard(expected, deadline))


def read_event(proc, timeout=3):
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise RuntimeError('Owned helper did not report readiness')
        line = proc.stdout.readline()
    if not line:
        raise RuntimeError('Owned helper exited before readiness')
    return json.loads(line)


def proc_state(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    # The comm field may contain spaces/parentheses; parse after its last ')'.
    return text[text.rfind(')') + 2:].split()[0]


def wait_not_running(pid, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = proc_state(pid)
        if state in (None, 'Z', 'X'):
            return state or 'absent'
        time.sleep(.01)
    raise AssertionError(f'Owned process {pid} remained running')


def stop_owned(proc):
    if proc.poll() is None:
        # All tracked handles are direct subprocesses and own their new session.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()


def helper(args):
    arm_deadline(args.deadline, args.expected_parent)
    if args.helper == 'supervisor':
        child = launch('busy', seconds=6)
        ready = read_event(child)
        print(json.dumps({'pid': os.getpid(), 'child_pid': child.pid,
                          'child_ready': ready, 'child_separate_pgid': os.getpgid(child.pid)}), flush=True)
        while True:
            time.sleep(.1)
    if args.helper == 'group':
        child = launch('busy', seconds=6, new_session=False)
        ready = read_event(child)
        print(json.dumps({'pid': os.getpid(), 'grandchild_pid': child.pid,
                          'grandchild_pgid': os.getpgid(child.pid), 'ready': ready}), flush=True)
        while True:
            time.sleep(.1)
    print(json.dumps({'pid': os.getpid(), 'ppid': os.getppid(),
                      'alarm_default': signal.getsignal(signal.SIGALRM) == signal.SIG_DFL,
                      'deadline_monotonic': args.deadline}), flush=True)
    if args.helper == 'normal':
        return 0
    while True:  # no sleeps or Python signal handler: hard default SIGALRM test
        pass


def run_tests():
    results = []
    handles = []
    try:
        # An unrelated owned sentinel must survive every tested termination.
        sentinel = launch('busy', seconds=20)
        handles.append(sentinel)
        read_event(sentinel)

        supervisor = launch('supervisor', seconds=8)
        handles.append(supervisor)
        event = read_event(supervisor)
        assert event['child_pid'] == event['child_separate_pgid']
        child_pid = event['child_pid']
        os.kill(supervisor.pid, signal.SIGKILL)
        assert supervisor.wait(timeout=2) == -signal.SIGKILL
        state = wait_not_running(child_pid)
        assert sentinel.poll() is None
        results.append({'test': 'SIGKILL_supervisor_terminates_separate_session_child',
                        'status': 'PASS', 'child_final_observed_state': state,
                        'zombie_means_no_execution_not_reaped': state == 'Z'})

        busy = launch('busy', seconds=.4)
        handles.append(busy)
        event = read_event(busy)
        assert event['alarm_default'] is True
        assert busy.wait(timeout=3) == -signal.SIGALRM
        assert sentinel.poll() is None
        results.append({'test': 'independent_default_SIGALRM_terminates_busy_loop', 'status': 'PASS'})

        normal = launch('normal', seconds=3)
        handles.append(normal)
        read_event(normal)
        assert normal.wait(timeout=2) == 0 and sentinel.poll() is None
        results.append({'test': 'normal_exit_preserves_unrelated_process', 'status': 'PASS'})

        group = launch('group', seconds=5)
        handles.append(group)
        event = read_event(group)
        assert event['grandchild_pgid'] == group.pid
        os.killpg(group.pid, signal.SIGTERM)
        assert group.wait(timeout=2) == -signal.SIGTERM
        state = wait_not_running(event['grandchild_pid'])
        assert sentinel.poll() is None
        results.append({'test': 'own_group_timeout_stops_own_descendants_only',
                        'status': 'PASS', 'grandchild_final_observed_state': state})

        wrong_parent = launch('busy', expected_parent=-1)
        handles.append(wrong_parent)
        assert wrong_parent.wait(timeout=2) == 125 and sentinel.poll() is None
        results.append({'test': 'post_prctl_expected_parent_mismatch_fails_closed', 'status': 'PASS'})

        expired = launch('busy', seconds=-.1)
        handles.append(expired)
        assert expired.wait(timeout=2) == 124 and sentinel.poll() is None
        results.append({'test': 'expired_absolute_deadline_exits_before_work', 'status': 'PASS'})

        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            blocked_alarm = launch('busy', seconds=.4)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        handles.append(blocked_alarm)
        read_event(blocked_alarm)
        assert blocked_alarm.wait(timeout=3) == -signal.SIGALRM and sentinel.poll() is None
        results.append({'test': 'inherited_blocked_SIGALRM_is_unblocked', 'status': 'PASS'})
        return results
    finally:
        for proc in reversed(handles):
            stop_owned(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper', choices=('busy', 'normal', 'supervisor', 'group'))
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--expected-parent', type=int)
    parser.add_argument('--guard-file', type=Path, help='Also test these actual production guard functions')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        print(json.dumps({'status': 'NOT_RUN_LINUX_REQUIRED', 'gpu_access': False}))
        return 2
    if args.helper:
        if args.guard_file is not None:
            load_guard(args.guard_file)
        return helper(args)
    started = time.monotonic()
    receipt = {'scope': 'independent_OS_primitives_not_production_qualification',
               'script_sha256': hashlib.sha256(Path(SELF).read_bytes()).hexdigest(),
               'python': sys.version, 'gpu_access': False, 'real_data_read': False}
    try:
        receipt['independent_primitive_tests'] = run_tests()
        if args.guard_file is not None:
            load_guard(args.guard_file)
            receipt['production_guard_sha256'] = hashlib.sha256(Path(GUARD_FILE).read_bytes()).hexdigest()
            receipt['production_guard_tests'] = run_tests()
            assert hashlib.sha256(Path(GUARD_FILE).read_bytes()).hexdigest() == receipt['production_guard_sha256']
        receipt['status'] = 'PASS_CPU_WATCHDOG_PRIMITIVES'
        code = 0
    except BaseException as exc:
        receipt.update(status='FAIL_CPU_WATCHDOG_PRIMITIVES', error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc())
        code = 1
    receipt['wall_seconds'] = time.monotonic() - started
    print(json.dumps(receipt, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
