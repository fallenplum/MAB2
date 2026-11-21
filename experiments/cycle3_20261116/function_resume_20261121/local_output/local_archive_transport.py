"""Pull a temporary compute spool over SFTP into a local durable archive."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time

MIN_SPOOL_FREE = 512 * 1024**2
MAX_SPOOL_BYTES = 256 * 1024**2
LEASE_SECONDS = 180


def ready(plan, spool):
    try:
        heartbeat = Path(plan["output"] + ".collector.json")
        with heartbeat.open("rb") as stream:
            age = time.time() - os.fstat(stream.fileno()).st_mtime
            value = json.load(stream)
        return (0 <= age <= LEASE_SECONDS and value["run_id"] == plan["collector_run_id"]
                and value["local_free_bytes"] >= plan["minimum_disk_bytes"]
                and shutil.disk_usage(spool).free >= MIN_SPOOL_FREE)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def inventory(spool):
    result = {}
    total = 0
    closed = (spool / "ARCHIVE_MANIFEST.json").is_file()
    for directory, subdirs, names in os.walk(spool, followlinks=False):
        if not closed and "sandbox_cases" in subdirs:
            subdirs.remove("sandbox_cases")
        for name in names:
            file = Path(directory) / name
            meta = file.lstat()
            if not stat.S_ISREG(meta.st_mode):
                raise ValueError("Unexpected spool file type")
            total += meta.st_size
            if total > MAX_SPOOL_BYTES:
                raise ValueError("Spool transfer ceiling exceeded")
            with file.open("rb") as stream:
                data = stream.read(meta.st_size)
            result[file.relative_to(spool).as_posix()] = [len(data), hashlib.sha256(data).hexdigest()]
    return result


def pull(root, remote_root, entries, previous, fetch):
    root = root.resolve()
    if sum(n for n, _ in entries.values()) > MAX_SPOOL_BYTES:
        raise ValueError("Spool transfer ceiling exceeded")
    for key, (length, expected) in sorted(entries.items()):
        parts = PurePosixPath(key).parts
        if not parts or any(not p or p in (".", "..") or ":" in p or "\\" in p for p in parts):
            raise ValueError("Invalid archive member")
        target = root.joinpath(*parts).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("Archive path escaped")
        if previous.get(key) == [length, expected] and target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + ".syncpart")
        with pending.open("wb") as dest:
            h = hashlib.sha256()
            with fetch(remote_root + "/" + key, "rb") as src:
                if hasattr(src, "prefetch") and length:
                    src.prefetch(length, max_concurrent_requests=16)
                remaining = length
                while remaining:
                    chunk = src.read(min(256 * 1024, remaining))
                    if not chunk:
                        raise OSError("Incomplete snapshot")
                    dest.write(chunk)
                    h.update(chunk)
                    remaining -= len(chunk)
            dest.flush()
            os.fsync(dest.fileno())
        if h.hexdigest() != expected:
            raise ValueError("Transferred prefix differs")
        os.replace(pending, target)
        previous[key] = [length, expected]


if __name__ == "__main__":
    import sys
    print(json.dumps(inventory(Path(sys.argv[1]))))
