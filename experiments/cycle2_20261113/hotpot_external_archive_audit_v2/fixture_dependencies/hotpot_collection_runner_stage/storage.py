"""Private durable writes and read-only disk guards; never delete data."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from orchestration_core import canonical,require


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_load(path):
    def unique(pairs):
        d={}
        for k,v in pairs:
            require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique)


def ensure_disk(path,minimum):
    if shutil.disk_usage(Path(path)).free<minimum:raise OSError('DISK_GUARD_LOW_SPACE')


def sync_directory(path):
    if os.name=='posix':
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def write_new(path,value):
    path=Path(path);raw=canonical(value)
    # A unique temporary file is fully durable before its final name becomes
    # visible. Hard-link publication is atomic and refuses an existing target.
    temporary=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex)
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)
    sync_directory(path.parent)
    # Remove only our redundant link after the immutable final link is durable.
    # This is publication bookkeeping, never disk-pressure cleanup of evidence.
    temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def progress(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.next')
    with temporary.open('wb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);sync_directory(path.parent)


def pinned_json(path,pin):
    require(file_sha(path)==pin,'Pinned JSON identity mismatch')
    value=json_load(path);require(file_sha(path)==pin,'Pinned JSON changed during read')
    return value
