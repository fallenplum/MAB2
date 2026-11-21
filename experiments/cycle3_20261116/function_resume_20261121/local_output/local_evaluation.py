"""Unprivileged local Linux evaluator using a hash-verified private userspace bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args):
    bundle=args.env/"native_userspace.tar.gz"
    record=json.loads((args.env/"PACKAGE.json").read_bytes())
    if digest(bundle)!=record["sha256"]:
        raise ValueError("Userspace bundle differs")
    # All work stays on this workstation. Use Linux-native storage for sandbox timings.
    with tempfile.TemporaryDirectory(prefix="mab2-formal10-") as d:
        scratch=Path(d); root=scratch/"root"; root.mkdir(); work=scratch/"work";work.mkdir()
        with tarfile.open(bundle) as tar:
            members=tar.getmembers()
            if len(members)!=record["members"] or not all(x.isreg() and not x.name.startswith("/") and ".." not in Path(x.name).parts for x in members):
                raise ValueError("Unexpected distribution bundle")
            tar.extractall(root, members=members, filter="data")
        for n in ("dev","proc","tmp","work","scorer","helper","input","labels","lib64"):
            if n!="lib64": (root/n).mkdir(exist_ok=True)
        (root/"lib").symlink_to("usr/lib")
        (root/"lib64").symlink_to("usr/lib64")
        (root/"labels/pilot.json").touch()
        opts=[str(root/"usr/bin/bwrap"),"--unshare-all","--die-with-parent","--new-session",
              "--cap-drop","ALL","--ro-bind",str(root),"/","--dev","/dev","--proc","/proc","--tmpfs","/tmp",
              "--bind",str(work),"/work","--ro-bind",str(args.scorer),"/scorer","--ro-bind",str(args.helper),"/helper"]
        def call(arguments):
            subprocess.run(opts+["--chdir","/work","/usr/bin/python3.8","-B"]+arguments,
                           check=True,env={"PATH":"/usr/bin","LANG":"C.UTF-8","PYTHONPATH":"/scorer:/helper"})
        if args.mode=="qualify":
            call(["/scorer/code_sandbox.py","--qualify-root","/work/isolation","--output","/work/isolation.json"])
            call(["/scorer/function_adapter.py","--root","/work/adapter","--output","/work/adapter.json"])
            isolation=json.loads((work/"isolation.json").read_bytes())
            adapter=json.loads((work/"adapter.json").read_bytes())
            if not all(x["pass_check"] for x in isolation["tests"]) or not all(x["check"] for x in adapter["tests"]):
                raise RuntimeError("Local isolation or adapter qualification failed")
            args.out.mkdir(exist_ok=False)
            for n in ("isolation.json","adapter.json"): shutil.copyfile(work/n,args.out/n)
            print("LOCAL_QUALIFICATION_PASS",len(isolation["tests"]),len(adapter["tests"]),adapter["comparison_checks"],flush=True)
            return
        if args.mode=="replay":
            opts+=["--ro-bind",str(args.view),"/input"]
            call(["/helper/replay_visible.py"])
            args.out.mkdir(exist_ok=False)
            shutil.copyfile(work/"replay.json",args.out/"replay.json")
            print((work/"replay.json").read_text(),flush=True)
            return
        release=json.loads(args.release.read_bytes())
        if release["status"]!="PILOT_HIDDEN_SCORING_RELEASED" or release["confirmation_labels_released"]:
            raise ValueError("Scoring release absent")
        if digest(args.view/"ARCHIVE_MANIFEST.json")!=release["archive_sha256"] or digest(args.labels)!=release["hidden_sha256"]:
            raise ValueError("Scoring pins differ")
        if not (args.qualification/"isolation.json").is_file() or not (args.qualification/"adapter.json").is_file():
            raise ValueError("Local qualification missing")
        opts+=["--ro-bind",str(args.view),"/input","--ro-bind",str(args.labels),"/labels/pilot.json"]
        try:
            call(["/helper/evaluate_joined.py","--archive","/input","--archive-sha",release["archive_sha256"],
                  "--hidden","/labels/pilot.json","--hidden-sha",release["hidden_sha256"],"--output","/work/hidden"])
        finally:
            if (work/"hidden").exists():
                shutil.copytree(work/"hidden",args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["qualify","replay","evaluate"])
    for key in ("env","scorer","helper","out","release","view","labels","qualification"):
        p.add_argument("--"+key,type=Path,required=key in ("env","scorer","helper","out"))
    execute(p.parse_args())
