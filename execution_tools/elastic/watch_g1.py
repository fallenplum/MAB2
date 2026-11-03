"""Read-only v3 G1 observer. Writes only artifacts/monitoring, never experiments.

Normal checks are silent. Credentials are read only from the runtime environment.
An operational REVIEW_READY is deliberately distinct from a scientific gate PASS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

RUN_ID = "motivation_v3_20261103"
SHARDS = 3
REVIEW_STAGE = "G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED"
COMPLETE_PHASE = "GENERATION_COMPLETE_AWAITING_INDEPENDENT_GATE"
STOP_STATES = {"REVIEW_READY", "ATTENTION_REQUIRED", "STOPPED"}
SHARD_FILES = ("STATUS.json", "frozen_plan.json", "run_manifest.json", "OUTPUTS_FROZEN.json", "SCORING.json",
               "physical_calls.jsonl", "raw_executions.jsonl", "observations.jsonl")


REMOTE_PROBE = r'''
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve(); run="motivation_v3_20261103"; count=3
errors=[]
def obj(value):return value if isinstance(value,dict) else {}
def read(p):
 try: return obj(json.loads(p.read_text(encoding="utf-8")))
 except FileNotFoundError: return {}
 except (OSError,ValueError) as e: errors.append({"file":p.relative_to(r).as_posix(),"error_type":type(e).__name__}); return {}
def meta(p,hashed=False):
 try:
  s=p.stat(); d={"exists":True,"bytes":s.st_size,"mtime":s.st_mtime}
  if hashed:
   h=hashlib.sha256()
   with p.open("rb") as f:
    for block in iter(lambda:f.read(1048576),b""):h.update(block)
   d["sha256"]=h.hexdigest()
  return d
 except FileNotFoundError:return {"exists":False,"bytes":0}
def tail(p):
 try:
  with p.open("rb") as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-131072));lines=f.read().splitlines()
  for line in reversed(lines):
   try:
    x=obj(json.loads(line));return {k:x.get(k) for k in ["status","call_id","error_type","started_unix","elapsed_seconds","input_tokens","output_tokens"]}
   except ValueError:continue
 except FileNotFoundError:pass
 return {}
def proc(pid,role,index=None):
 d={"pid":pid,"role":role,"shard_index":index,"alive":False}
 try:
  p=Path("/proc")/str(int(pid)); raw=(p/"stat").read_text(); fields=raw[raw.rfind(")")+2:].split()
  d.update(alive=fields[0]!="Z",state=fields[0],cpu_ticks=int(fields[11])+int(fields[12]),start_ticks=int(fields[19]))
  argv=[x.decode("utf-8","replace") for x in (p/"cmdline").read_bytes().split(b"\0") if x]
  cwd=os.readlink(p/"cwd");d.update(argv=argv,cwd=cwd)
  status=(p/"status").read_text()
  for line in status.splitlines():
   if line.startswith("VmRSS:"):d["rss_kib"]=int(line.split()[1])
   if line.startswith("Uid:"):d["uid"]=int(line.split()[1])
  def flag(name):
   try:return argv[argv.index(name)+1]
   except (ValueError,IndexError):return None
  elastic=main.get("schema")=="graph_credit_elastic_supervisor_v1"
  script=("elastic_supervisor.py" if role=="main" else "elastic_worker.py") if elastic else ("server_supervisor.py" if role=="main" else "run_pilot.py")
  script_dir=r/"artifacts/private/performance" if elastic else r/"scripts"
  candidates=[x for x in argv if x.endswith("/"+script) or x==script or x.endswith("\\"+script)]
  script_ok=any((Path(cwd)/x).resolve()==script_dir/script for x in candidates)
  d["matches_project"]=script_ok and Path(cwd).resolve()==r and flag("--root") is not None and Path(flag("--root")).resolve()==r and d.get("uid")==os.getuid()
  if role=="child" and elastic:
   workers=main.get("active_workers") or [];worker=workers[index] if index is not None and index<len(workers) else {}
   d["scheduler_worker"]=True
   d["completed"]=False  # The elastic supervisor publishes only live formal workers here.
   d["matches_project"]=d["matches_project"] and flag("--worker-id")==worker.get("worker_id") and flag("--task-manifest") is not None and Path(flag("--task-manifest")).resolve()==r/worker.get("task_manifest","") and worker.get("pid")==pid
  elif role=="child":d["matches_project"]=d["matches_project"] and flag("--run-id")==run and flag("--num-shards")==str(count) and flag("--shard-index")==str(index)
 except (OSError,ValueError,IndexError) as e:d["inspection_error"]=type(e).__name__
 return d
main=read(r/"RUN_STATUS.json");terminal=main.get("stage")=="G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED"
base=r/"results/pilot"/run;shards=[];source_errors=[];first_plan={}
names=["STATUS.json","frozen_plan.json","run_manifest.json","OUTPUTS_FROZEN.json","SCORING.json","physical_calls.jsonl","raw_executions.jsonl","observations.jsonl"]
for i in range(count):
 p=base/("shard_%03d"%i);plan=read(p/"frozen_plan.json");manifest=read(p/"run_manifest.json")
 if plan and not first_plan:first_plan=plan
 files={n:meta(p/n,n in ["frozen_plan.json","run_manifest.json"] or terminal and n=="observations.jsonl") for n in names}
 shards.append({"index":i,"status":read(p/"STATUS.json"),"files":files,"last_physical_call":tail(p/"physical_calls.jsonl"),"outputs_frozen":read(p/"OUTPUTS_FROZEN.json"),"scoring":read(p/"SCORING.json"),"protocol_hash":manifest.get("protocol_hash"),"source_manifest_sha256":manifest.get("source_manifest_sha256"),"plan_protocol_hash":plan.get("protocol_hash"),"plan_source_manifest_sha256":plan.get("source_manifest_sha256"),"executor_version":obj(manifest.get("executor_config")).get("executor_version")})
for name,item in obj(first_plan.get("source_manifest")).items():
 p=(r/name).resolve()
 try:
  if not p.is_relative_to(r) or hashlib.sha256(p.read_bytes()).hexdigest()!=obj(item).get("sha256"):source_errors.append(name)
 except OSError:source_errors.append(name)
if main.get("schema")=="graph_credit_elastic_supervisor_v1":
 allocation_path=r/"artifacts/private/performance/allocation.json";allocation=read(allocation_path)
 if meta(allocation_path,True).get("sha256")!=main.get("allocation_sha256"):source_errors.append("elastic_allocation")
 for name,h in obj(allocation.get("scheduler_sources")).items():
  p=(r/name).resolve()
  if not p.is_relative_to(r) or meta(p,True).get("sha256")!=h:source_errors.append("elastic_source:"+name)
 for worker in allocation.get("workers",[]):
  p=(r/worker["task_manifest"]).resolve()
  if not p.is_relative_to(r) or meta(p,True).get("sha256")!=worker.get("task_manifest_sha256"):source_errors.append("elastic_tasks:"+worker.get("worker_id","unknown"))
pids=main.get("active_pids") or [];processes=[]
if main.get("pid"):processes.append(proc(main["pid"],"main"))
for i,pid in enumerate(pids):processes.append(proc(pid,"child",i))
gpu={"selected_indices":main.get("gpus") or []}
try:
 raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=15)
 gpu["rows"]=[line.strip() for line in raw.splitlines() if int(line.split(",")[0]) in (main.get("gpus") or [])]
except (OSError,ValueError,subprocess.SubprocessError) as e:gpu["error_type"]=type(e).__name__
artifacts={"all_observations.jsonl":meta(base/"all_observations.jsonl",terminal),"G1_ANALYSIS.json":meta(base/"G1_ANALYSIS.json",terminal)}
analysis=read(base/"G1_ANALYSIS.json") if terminal else {}
analysis_summary={"scientific_gate":obj(analysis.get("scientific_gate")),"support":{k:obj(analysis.get("support")).get(k) for k in ["input_rows","question_count","incomplete_group_count"]},"plan_verified":obj(obj(analysis.get("provenance")).get("complete_frozen_design_coverage")).get("verified")}
space=shutil.disk_usage(r);memory={}
for line in Path("/proc/meminfo").read_text().splitlines():
 if line.startswith(("MemAvailable:","MemTotal:")):memory[line.split(":")[0]+"_kib"]=int(line.split()[1])
print(json.dumps({"observed_unix":time.time(),"main":main,"shards":shards,"processes":processes,"gpu":gpu,"artifacts":artifacts,"analysis":analysis_summary,"source_mismatches":source_errors,"read_errors":errors,"disk":{"total_bytes":space.total,"free_bytes":space.free},"host_memory":memory}))
'''


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"_read_error": "not_object"}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        return {"_read_error": type(exc).__name__}


def object_value(value) -> dict:
    """Optional JSON object fields are often null before analysis exists."""
    return value if isinstance(value, dict) else {}


def file_meta(path: Path, hashed=False) -> dict:
    try:
        st = path.stat()
        value = {"exists": True, "bytes": st.st_size, "mtime": st.st_mtime}
        if hashed:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            value["sha256"] = h.hexdigest()
        return value
    except FileNotFoundError:
        return {"exists": False, "bytes": 0}


def local_snapshot(root: Path) -> dict:
    main, coord = read_json(root / "RUN_STATUS.json"), read_json(root / "COORDINATOR_STATUS.json")
    terminal = main.get("stage") == REVIEW_STAGE
    base = root / "results" / "pilot" / RUN_ID
    shards = []
    for index in range(SHARDS):
        folder = base / f"shard_{index:03d}"
        shards.append({"index": index, "status": read_json(folder / "STATUS.json"),
                       "files": {name: file_meta(folder / name, name == "frozen_plan.json" or terminal and name == "observations.jsonl") for name in SHARD_FILES},
                       "outputs_frozen": read_json(folder / "OUTPUTS_FROZEN.json"), "scoring": read_json(folder / "SCORING.json")})
    coordinator_process = {"pid": coord.get("pid"), "inspection_available": False}
    if coord.get("pid"):
        try:
            import psutil
            process = psutil.Process(int(coord["pid"]))
            argv = process.cmdline()
            coordinator_process.update(inspection_available=True, alive=process.is_running(),
                                       create_time=process.create_time(), rss_bytes=process.memory_info().rss)
            try:
                cli_root = Path(argv[argv.index("--root") + 1]).resolve()
            except (ValueError, IndexError):
                cli_root = None
            coordinator_process["matches_project"] = any(Path(x).name == "local_coordinator.py" for x in argv) and cli_root == root
        except ImportError:
            coordinator_process["inspection_error"] = "psutil_unavailable"
        except Exception as exc:
            coordinator_process.update(inspection_available=True, alive=False, inspection_error=type(exc).__name__)
    space = shutil.disk_usage(root)
    return {"main": main, "coordinator": coord, "coordinator_process": coordinator_process,
            "disk": {"total_bytes": space.total, "free_bytes": space.free},
            "shards": shards, "artifacts": {name: file_meta(base / name, terminal) for name in ("all_observations.jsonl", "G1_ANALYSIS.json")}}


def remote_snapshot(args) -> dict:
    import paramiko
    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        if not os.environ.get("GRAPH_CREDIT_SSH_PASSWORD"):
            return {"connection_status": "AUTH_OR_HOSTKEY_ERROR", "error_type": "MISSING_RUNTIME_CREDENTIAL"}
        client.connect(args.host, port=args.port, username=args.user,
                       password=os.environ["GRAPH_CREDIT_SSH_PASSWORD"], look_for_keys=False,
                       allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
        command = " ".join((shlex.quote(args.remote_python), "-c", shlex.quote(REMOTE_PROBE), shlex.quote(args.remote_root)))
        _, out, err = client.exec_command(command, timeout=60)
        data = out.read()
        err.read()  # Do not log remote exceptions that may contain private paths.
        code = out.channel.recv_exit_status()
        if code:
            return {"connection_status": "PROBE_FAILED", "exit_code": code}
        value = json.loads(data)
        value["connection_status"] = "OK"
        return value
    except (paramiko.BadHostKeyException, paramiko.AuthenticationException) as exc:
        return {"connection_status": "AUTH_OR_HOSTKEY_ERROR", "error_type": type(exc).__name__}
    except paramiko.SSHException as exc:
        state = "AUTH_OR_HOSTKEY_ERROR" if "not found in known_hosts" in str(exc) else "CONNECTION_RETRY"
        return {"connection_status": state, "error_type": type(exc).__name__}
    except Exception as exc:
        return {"connection_status": "CONNECTION_RETRY", "error_type": type(exc).__name__}
    finally:
        client.close()


def growth_key(snapshot: dict) -> dict:
    remote = object_value(snapshot.get("remote"))
    return {str(s["index"]): {"outputs": s.get("status", {}).get("outputs_frozen", 0),
                             "raw_bytes": s.get("files", {}).get("raw_executions.jsonl", {}).get("bytes", 0),
                             "calls_bytes": s.get("files", {}).get("physical_calls.jsonl", {}).get("bytes", 0),
                             "scored_bytes": s.get("files", {}).get("observations.jsonl", {}).get("bytes", 0)}
            for s in remote.get("shards", [])}


def classify(snapshot: dict, previous: dict | None = None, interval=1800) -> dict:
    """Pure operational decision; never upgrades a scientific claim."""
    previous = previous or {}
    remote, local = object_value(snapshot.get("remote")), object_value(snapshot.get("local"))
    connection = remote.get("connection_status")
    reasons = []
    state = "RUNNING"
    for side, data in (("local", local), ("remote", remote)):
        free_bytes = object_value(data.get("disk")).get("free_bytes")
        if isinstance(free_bytes, int) and free_bytes < 2 * 1024 ** 3:
            reasons.append(side + "_free_disk_below_2GiB")
    if reasons:
        return {"state": "ATTENTION_REQUIRED", "reasons": reasons, "scientific_completion": False}
    if connection in {"AUTH_OR_HOSTKEY_ERROR", "PROBE_FAILED"}:
        return {"state": "ATTENTION_REQUIRED", "reasons": [connection], "scientific_completion": False}
    if connection != "OK":
        return {"state": "CONNECTION_RETRY", "reasons": ["remote_connection_unavailable"], "scientific_completion": False}
    main, coord = object_value(remote.get("main")), object_value(local.get("coordinator"))
    stage = main.get("stage")
    if stage in {"FAILED", "G1_PARTIAL_FAILED", "FAILED_CHILDREN_SETTLING"}:
        reasons.append("remote_supervisor_failure")
    if coord.get("stage") == "FAILED":
        reasons.append("local_coordinator_failed")
    if remote.get("source_mismatches"):
        reasons.append("frozen_remote_source_mismatch")
    if remote.get("read_errors"):
        reasons.append("remote_status_or_provenance_unreadable")
    if not main or main.get("expected_graph_observations") not in (None, 60000):
        reasons.append("missing_or_unexpected_remote_main")
    shards = remote.get("shards") or []
    if len(shards) != SHARDS:
        reasons.append("unexpected_shard_count")
    plan_hashes, protocols = set(), set()
    all_done = len(shards) == SHARDS
    for shard in shards:
        status = object_value(shard.get("status"))
        if status.get("phase") == "FAILED" or shard.get("last_physical_call", {}).get("status") == "FAILED":
            reasons.append(f"shard_{shard['index']}_failed")
        if shard.get("executor_version") not in (None, "unrolled_closed_input_v3"):
            reasons.append(f"shard_{shard['index']}_executor_version_mismatch")
        if status.get("phase") not in {None, "LOADING_MODEL"}:
            if shard.get("executor_version") != "unrolled_closed_input_v3" or shard.get("protocol_hash") != shard.get("plan_protocol_hash") or shard.get("source_manifest_sha256") != shard.get("plan_source_manifest_sha256"):
                reasons.append(f"shard_{shard['index']}_plan_manifest_identity_mismatch")
        if shard.get("protocol_hash"):
            protocols.add(shard["protocol_hash"])
        plan_file = shard.get("files", {}).get("frozen_plan.json", {})
        if plan_file.get("sha256"):
            plan_hashes.add(plan_file["sha256"])
            if status.get("phase") != "LOADING_MODEL" and status.get("frozen_plan_sha256") != plan_file["sha256"]:
                reasons.append(f"shard_{shard['index']}_status_plan_hash_mismatch")
        done = (status.get("phase") == COMPLETE_PHASE and status.get("generation_complete") is True
                and status.get("scored") is True)
        all_done = all_done and done
        if done:
            count = status.get("outputs_frozen")
            if not count or count != shard.get("outputs_frozen", {}).get("outputs") or count != shard.get("scoring", {}).get("diagnostic_scoring_count"):
                reasons.append(f"shard_{shard['index']}_completion_evidence_mismatch")
    if len(plan_hashes) > 1 or len(protocols) > 1:
        reasons.append("cross_shard_plan_or_protocol_mismatch")
    if stage == "G1_REAL_GENERATION" and (len(shards) != SHARDS or any(not s.get("status") for s in shards)):
        reasons.append("missing_active_shard_status")
    for proc in remote.get("processes", []):
        active = proc.get("role") == "main" and stage != REVIEW_STAGE
        if proc.get("role") == "child":
            index = proc.get("shard_index")
            if proc.get("scheduler_worker"):
                active = stage == "G1_REAL_GENERATION" and not proc.get("completed", False)
            else:
                active = stage == "G1_REAL_GENERATION" and index is not None and index < len(shards) and not (
                    shards[index].get("status", {}).get("generation_complete") is True)
        if active and not proc.get("alive"):
            reasons.append(f"dead_remote_{proc.get('role')}_{proc.get('pid')}")
        elif active and proc.get("matches_project") is not True:
            reasons.append(f"remote_process_command_or_cwd_mismatch_{proc.get('pid')}")
    cp = object_value(local.get("coordinator_process"))
    if coord.get("stage") != "REMOTE_TERMINAL_MIRRORED" and cp.get("inspection_available") and (not cp.get("alive") or not cp.get("matches_project")):
        reasons.append("local_coordinator_process_dead_or_mismatched")
    # Compare immutable provenance, allowing normal lag for growing output files.
    for left, right in zip(local.get("shards", []), shards):
        local_hash = left.get("files", {}).get("frozen_plan.json", {}).get("sha256")
        remote_hash = right.get("files", {}).get("frozen_plan.json", {}).get("sha256")
        if local_hash and remote_hash and local_hash != remote_hash:
            reasons.append(f"local_remote_plan_mismatch_{right['index']}")
    growth = growth_key(snapshot)
    stagnant = (previous.get("consecutive_no_growth", 0) + 1) if growth and growth == previous.get("growth") else 0
    if stagnant >= 2 and stage == "G1_REAL_GENERATION":
        old_procs = {str(p.get("pid")): p for p in previous.get("remote", {}).get("processes", [])}
        abnormal = []
        for proc in remote.get("processes", []):
            if proc.get("role") != "child" or not proc.get("alive") or proc.get("completed", False):
                continue
            old = old_procs.get(str(proc.get("pid")), {})
            same_cpu = old.get("start_ticks") == proc.get("start_ticks") and old.get("cpu_ticks") == proc.get("cpu_ticks")
            index = proc.get("shard_index")
            shard = shards[index] if isinstance(index, int) and index < len(shards) else {}
            last = shard.get("last_physical_call", {})
            last_time = (last.get("started_unix") or 0) + (last.get("elapsed_seconds") or 0)
            stale_call = snapshot.get("observed_unix", time.time()) - last_time > 2 * interval
            if proc.get("state") in {"T", "t", "D", "Z"} or same_cpu and stale_call:
                abnormal.append(proc.get("pid"))
        if abnormal:
            reasons.append("two_checks_without_growth_and_abnormal_workers:" + ",".join(map(str, abnormal)))
    artifacts_ok = all(remote.get("artifacts", {}).get(name, {}).get("bytes", 0) > 0 and
                       local.get("artifacts", {}).get(name, {}).get("sha256") == remote.get("artifacts", {}).get(name, {}).get("sha256")
                       and remote.get("artifacts", {}).get(name, {}).get("sha256")
                       for name in ("all_observations.jsonl", "G1_ANALYSIS.json"))
    analysis = object_value(remote.get("analysis"))
    analysis_support = object_value(analysis.get("support"))
    scientific_gate = object_value(analysis.get("scientific_gate"))
    analysis_ok = (analysis_support.get("input_rows") == 60000 and analysis.get("plan_verified") is True
                   and scientific_gate.get("status") in {"PASS", "INSUFFICIENT_SIGNAL"})
    local_shards = local.get("shards", [])
    local_plan_hashes = {s.get("files", {}).get("frozen_plan.json", {}).get("sha256") for s in local_shards}
    plans_present = (len(local_shards) == SHARDS and len(shards) == SHARDS
                     and all(s.get("files", {}).get("frozen_plan.json", {}).get("exists") is True
                             and s.get("files", {}).get("frozen_plan.json", {}).get("sha256")
                             for s in local_shards + shards)
                     and len(plan_hashes) == 1 and local_plan_hashes == plan_hashes)
    local_done = (len(local_shards) == SHARDS and len(shards) == SHARDS
                  and all(s.get("status", {}).get("phase") == COMPLETE_PHASE
                          and s.get("status", {}).get("generation_complete") is True
                          and s.get("status", {}).get("scored") is True
                          and s.get("status", {}).get("outputs_frozen") == remote_shard.get("status", {}).get("outputs_frozen")
                          for s, remote_shard in zip(local_shards, shards)))
    if stage == REVIEW_STAGE and coord.get("stage") == "REMOTE_TERMINAL_MIRRORED" and not (all_done and local_done and analysis_ok and artifacts_ok and plans_present and len(protocols) == 1):
        reasons.append("terminal_completion_or_analysis_evidence_inconsistent")
    if reasons:
        state = "ATTENTION_REQUIRED"
    elif stage == REVIEW_STAGE and all_done and local_done and artifacts_ok and analysis_ok and plans_present and coord.get("stage") == "REMOTE_TERMINAL_MIRRORED":
        state = "REVIEW_READY"
        reasons = ["G1_generation_scoring_analysis_and_local_mirror_complete_requires_scientific_review"]
    elif stage == REVIEW_STAGE:
        state = "FINALIZING_MIRROR"
    return {"state": state, "reasons": reasons, "consecutive_no_growth": stagnant, "growth": growth,
            "remote_stage": stage, "coordinator_stage": coord.get("stage"),
            "scientific_gate_status": scientific_gate.get("status"),
            "scientific_completion": False}


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--known-hosts", required=True, type=Path)
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error("interval must be positive")
    root = args.root.resolve()
    folder = root / "artifacts" / "monitoring"
    folder.mkdir(parents=True, exist_ok=True)
    stop = folder / "STOP_MONITOR"
    status_file = folder / "MONITOR_STATUS.json"
    previous = read_json(status_file)
    while True:
        if stop.exists():
            atomic_json(status_file, {"state": "STOPPED", "run_id": RUN_ID, "reasons": ["intentional_scheduler_transition"], "scientific_completion": False})
            print(json.dumps({"state": "STOPPED", "run_id": RUN_ID}), flush=True)
            return 0
        started = time.monotonic()
        try:
            snapshot = {"observed_unix": time.time(), "run_id": RUN_ID,
                        "local": local_snapshot(root), "remote": remote_snapshot(args)}
            decision = classify(snapshot, previous, args.interval_seconds)
        except Exception as exc:
            snapshot = {"observed_unix": time.time(), "run_id": RUN_ID}
            decision = {"state": "ATTENTION_REQUIRED", "reasons": ["monitor_error:" + type(exc).__name__], "scientific_completion": False}
        record = {**snapshot, **decision, "monitor_pid": os.getpid(), "interval_seconds": args.interval_seconds}
        next_check = (time.time() + max(0, args.interval_seconds - (time.monotonic() - started))) if decision["state"] not in STOP_STATES else None
        record.update(timezone="Asia/Shanghai", next_check_unix=next_check,
                      next_check_local=dt.datetime.fromtimestamp(next_check, dt.timezone(dt.timedelta(hours=8))).isoformat() if next_check else None)
        with (folder / "snapshots.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(status_file, record)
        previous = record
        if args.once or decision["state"] in STOP_STATES:
            print(json.dumps({key: record.get(key) for key in ("state", "run_id", "remote_stage", "coordinator_stage", "scientific_gate_status", "reasons", "scientific_completion")}), flush=True)
            return 2 if decision["state"] == "ATTENTION_REQUIRED" else 0
        deadline = started + args.interval_seconds
        while time.monotonic() < deadline:
            if stop.exists():
                atomic_json(status_file, {"state": "STOPPED", "run_id": RUN_ID, "reasons": ["intentional_scheduler_transition"], "scientific_completion": False})
                print(json.dumps({"state": "STOPPED", "run_id": RUN_ID}), flush=True)
                return 0
            time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
