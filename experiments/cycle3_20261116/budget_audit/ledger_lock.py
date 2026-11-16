"""OS-released local lock for serializing concurrent experiment cost closure."""
from pathlib import Path
from contextlib import contextmanager
import os,time

@contextmanager
def budget_lock(root,timeout=120):
    path=Path(root)/'GLOBAL_AUTORESEARCH_BUDGET.lock'
    stream=path.open('a+b')
    if stream.seek(0,2)==0:stream.write(b'0');stream.flush()
    deadline=time.monotonic()+timeout
    acquired=False
    try:
        while True:
            try:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired=True;break
            except (OSError,BlockingIOError):
                if time.monotonic()>=deadline:raise TimeoutError('Budget ledger busy; no mutation attempted')
                time.sleep(.1)
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
