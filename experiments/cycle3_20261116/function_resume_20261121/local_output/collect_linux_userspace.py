"""Write a tar stream of explicitly scoped Linux userspace dependencies to stdout."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

stdlib=Path("/usr/lib/python3.8")
lib=Path("/usr/lib/x86_64-linux-gnu")
assert stdlib.is_dir() and lib.is_dir()
bins=[Path("/usr/bin/python3.8"),Path("/usr/bin/bwrap")]
allowed_roots=[Path("/usr/lib"),Path("/lib"),Path("/usr/lib64"),Path("/lib64")]
dependencies={lib/"libseccomp.so.2"}
for binary in bins+list(stdlib.glob("lib-dynload/*.so"))+[lib/"libseccomp.so.2"]:
    proc=subprocess.run(["ldd",str(binary)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assert proc.returncode==0 and "not found" not in proc.stdout
    for match in re.findall(r"(/[^\s()]+)\s+\(",proc.stdout):
        path=Path(match)
        assert any(path==root or root in path.parents for root in allowed_roots)
        dependencies.add(path)
paths={p.relative_to("/").as_posix():p for p in bins+[Path("/etc/ld.so.cache")]}
for p in sorted(stdlib.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc":
        # libpython config directories may link to the distribution's static development library.
        if p.stat().st_size < 40*1024**2:
            paths[p.relative_to("/").as_posix()]=p
for path in dependencies:
    if len(path.parts)>2 and path.parts[1] in ("lib","lib64"):
        alias="usr/"+path.relative_to("/").as_posix()
    else:alias=path.relative_to("/").as_posix()
    paths[alias]=path
assert sum(p.stat().st_size for p in paths.values())<384*1024**2
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz",dereference=True) as tar:
    for name,p in sorted(paths.items()):
        info=tarfile.TarInfo(name)
        info.mode=0o755 if os.access(p,os.X_OK) else 0o644
        info.size=p.stat().st_size
        with p.open("rb") as f:tar.addfile(info,f)
