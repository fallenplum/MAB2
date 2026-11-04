"""Local data-owner coordination for the already frozen G2 experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import threading
import time
import uuid

import mirror as unit_mirror


RUN_ID = "g2_v1_20261104"
CONTROL_NAMES = ("STATUS.json", "WORKFLOW_STATUS.json", "GLOBAL_OUTPUTS_FROZEN.json",
                 "PREDICTIONS_FROZEN.json", "G2_ANALYSIS.json")
MAX_METADATA = 512 * 1024 * 1024
FINAL = "READY_FOR_INDEPENDENT_AUDIT"


class IntegrityError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def byte_sha(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (canonical(value) + "\n").encode("utf-8")


def decode(content):
    return json.loads(content.decode("utf-8-sig"), object_pairs_hook=unit_mirror.unique_object)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def control_state(value):
    return value.get("state", value.get("phase", value.get("stage")))


def exception_chain(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


def fatal_connection(exc):
    return any(unit_mirror.fatal_authentication(item) for item in exception_chain(exc))


def retryable_connection(exc):
    network_codes = {errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET, errno.ETIMEDOUT,
                     errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE}
    return any(isinstance(item, (ConnectionError, TimeoutError, EOFError))
               or type(item).__name__ in {"SSHException", "NoValidConnectionsError", "ChannelException"}
               or isinstance(item, OSError) and item.errno in network_codes for item in exception_chain(exc))


def relative(name):
    unit_mirror.relative_parts(name)
    return name


def strict_client_factory(args):
    """Credentials remain in this process and never enter commands or receipts."""
    import paramiko
    password = os.environ.get("GRAPH_CREDIT_SSH_PASSWORD")
    require(bool(password), "Runtime SSH authentication is unavailable")
    known_hosts = Path(args.known_hosts).resolve(strict=True)
    def connect():
        client = paramiko.SSHClient()
        client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(args.host, port=args.port, username=args.user, password=password,
                           timeout=20, auth_timeout=20, banner_timeout=20,
                           look_for_keys=False, allow_agent=False)
            return client
        except BaseException:
            unit_mirror.close_connection(client, None)
            raise
    return connect


@contextmanager
def connection(factory):
    client = sftp = None
    try:
        client, sftp = unit_mirror.connect(factory)
        yield client, sftp
    finally:
        unit_mirror.close_connection(client, sftp)


def remote_bytes(sftp, path, *, limit=MAX_METADATA):
    before = sftp.lstat(path)
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, "Remote artifact type/size is invalid")
    with sftp.open(path, "rb") as handle:
        opened = handle.stat()
        require(opened.st_size == before.st_size, "Remote artifact changed while opening")
        chunks, remaining = [], before.st_size
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise EOFError("Remote artifact ended before its fixed snapshot size")
            chunks.append(block)
            remaining -= len(block)
        require(not handle.read(1), "Remote artifact grew while mirroring a fixed snapshot")
    after = sftp.lstat(path)
    require(after.st_size == before.st_size and after.st_mtime == before.st_mtime,
            "Remote artifact changed before its snapshot completed")
    return b"".join(chunks)


def optional_remote(sftp, path, *, limit=MAX_METADATA):
    try:
        return remote_bytes(sftp, path, limit=limit)
    except FileNotFoundError:
        return None


def remote_exists(sftp, path):
    try:
        sftp.lstat(path)
        return True
    except FileNotFoundError:
        return False


def mkdir_remote(sftp, remote_root, parent):
    current = str(remote_root)
    for part in relative(parent).split("/"):
        current += "/" + part
        try:
            info = sftp.lstat(current)
            require(stat.S_ISDIR(info.st_mode), "Remote release parent is linked or is not a directory")
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_once(sftp, remote_root, name, content):
    """Publish only the caller's immutable label/release bytes, never replace history."""
    relative(name)
    target = str(remote_root / name)
    existing = optional_remote(sftp, target)
    if existing is not None:
        require(existing == content, "Existing remote release/input bytes differ from the owner receipt")
        return
    mkdir_remote(sftp, remote_root, str(PurePosixPath(name).parent))
    temporary = target + "." + uuid.uuid4().hex + ".uploading"
    created = False
    try:
        with sftp.open(temporary, "wx") as handle:
            created = True
            handle.write(content)
            handle.flush()
        require(optional_remote(sftp, target) is None, "Another publisher committed this release during transfer")
        sftp.posix_rename(temporary, target)
        created = False
        require(remote_bytes(sftp, target) == content, "Remote atomic release verification failed")
    finally:
        if created:
            try:
                sftp.remove(temporary)
            except OSError:
                pass


def split_original_labels(content, partitions):
    require(content.endswith(b"\n"), "Original calibration labels have an incomplete trailing record")
    groups = {name: [] for name in ("train", "tune", "test")}
    seen = set()
    for line in content.splitlines(keepends=True):
        require(bool(line.strip()), "Original calibration labels contain a blank record")
        row = decode(line)
        require(isinstance(row, dict) and isinstance(row.get("id"), str), "Malformed original calibration label ID")
        qid = row["id"]
        require(qid in partitions and qid not in seen, "Original labels duplicate or exceed the frozen 400 question IDs")
        seen.add(qid)
        groups[partitions[qid]].append(line)
    require(seen == set(partitions) and Counter(partitions.values()) == {"train": 200, "tune": 80, "test": 120},
            "Original labels do not cover the exact frozen 200/80/120 split")
    return {name: b"".join(lines) for name, lines in groups.items()}


def validate_global(marker, plan, plan_hash):
    expected = {"schema": "graph_credit_g2_outputs_frozen_v1", "state": "GLOBAL_OUTPUTS_FROZEN", "run_id": plan["run_id"],
                "plan_sha256": plan_hash, "source_manifest_sha256": digest(plan["source_manifest"]),
                "pipeline_source_manifest_sha256": digest(plan["pipeline_source_manifest"]), "protocol_hash": plan["protocol_hash"],
                "expected_units": 2000, "completed_units": 2000, "expected_outputs": 51970, "observations": 51970,
                "features_sha256": plan["features"]["sha256"], "features_receipt_sha256": plan["features"]["receipt_sha256"],
                "labels_read": False, "first_gold_opened_unix": None}
    require(plan["expected_units"] == 2000 and plan["expected_outputs"] == 51970, "Unexpected frozen G2 support")
    require(all(marker.get(name) == value for name, value in expected.items()), "Global output freeze is incomplete or foreign")
    hashes = marker.get("artifact_hashes", {})
    require(set(hashes) == {"observations", "physical_cost_audit", "features", "features_receipt"}, "Global artifact hash support changed")
    require(hashes["features"] == plan["features"]["sha256"] and hashes["features_receipt"] == plan["features"]["receipt_sha256"],
            "Global feature hashes differ from the plan")
    for name in ("observations", "physical_cost_audit"):
        item = marker["artifacts"][name]
        relative(item["path"])
        require(item["sha256"] == hashes[name] and item["bytes"] >= 0, "Global artifact descriptor/hash mismatch")


def validate_observations(content, plan, plan_hash):
    expected = {(unit["question_id"], unit["seed"], tuple(request["mask"]))
                for unit in plan["units"] for request in unit["mask_requests"]}
    seen = set()
    for line in content.splitlines(keepends=True):
        require(line.endswith(b"\n") and bool(line.strip()), "Canonical observations contain an incomplete record")
        row = decode(line)
        key = (row["question_id"], row["seed"], tuple(row["graph_mask"]))
        require(key in expected and key not in seen and row.get("plan_sha256") == plan_hash
                and row.get("source_manifest_sha256") == digest(plan["source_manifest"])
                and row.get("protocol_hash") == plan["protocol_hash"] and row.get("quality") is None
                and row.get("labels_read") is False and row.get("partition") == plan["partitions"][key[0]],
                "Canonical generation coverage or label-free provenance differs")
        seen.add(key)
    require(seen == expected and len(seen) == plan["expected_outputs"], "Canonical generation has missing unique outputs")


def independent_predictions(plan, plan_hash, global_hash, marker, parameters, predictions, phi):
    """Owner-side formula replay without importing the fitted predictor implementation."""
    import numpy as np
    expected = {"schema": "graph_credit_g2_predictions_frozen_v1", "state": "PREDICTIONS_FROZEN", "run_id": plan["run_id"],
                "plan_sha256": plan_hash, "source_manifest_sha256": digest(plan["source_manifest"]),
                "pipeline_source_manifest_sha256": digest(plan["pipeline_source_manifest"]),
                "global_outputs_frozen_sha256": global_hash, "feature_sha256": plan["features"]["sha256"],
                "features_receipt_sha256": plan["features"]["receipt_sha256"], "heldout_labels_read": False,
                "train_questions": 200, "tune_questions": 80, "heldout_questions": 120, "heldout_prediction_rows": 2880}
    require(all(marker.get(name) == value for name, value in expected.items()), "Prediction freeze provenance/support differs")
    require(marker["parameters_semantic_sha256"] == parameters.get("parameter_sha256")
            == digest({key: value for key, value in parameters.items() if key != "parameter_sha256"})
            and marker["predictions_semantic_sha256"] == digest(predictions), "Prediction/parameter semantic SHA mismatch")
    require(parameters.get("schema") == "graph_credit_g2_predictor_v1", "Unsupported frozen predictor schema")
    require(parameters.get("seeds") == [2701, 2702, 2703, 2704, 2705], "Predictor seed population differs from the frozen five seeds")
    train = {qid for qid, part in plan["partitions"].items() if part == "train"}
    tune = {qid for qid, part in plan["partitions"].items() if part == "tune"}
    test = {qid for qid, part in plan["partitions"].items() if part == "test"}
    require(set(parameters["train_question_ids"]) == train and len(parameters["train_question_ids"]) == 200
            and set(parameters["tune_question_ids"]) == tune and len(parameters["tune_question_ids"]) == 80
            and not (train | tune) & test, "Predictor fitting populations intersect or differ from the frozen partitions")
    baseline = np.asarray(parameters["B0"], dtype=np.float64)
    require(baseline.shape == (12,) and np.isfinite(baseline).all(), "Malformed B0 signed baseline")
    coefficients = {}
    for name, order, dimensions in (("C1", 1, 117), ("C2", 2, 711)):
        model = parameters["models"][name]
        theta = np.asarray(model["coefficients"], dtype=np.float64)
        require(model["order"] == order and model["dimensions"] == dimensions
                and theta.shape == (dimensions,) and np.isfinite(theta).all(), "Malformed frozen ridge representation")
        coefficients[name] = theta
    planned = {}
    for qid in test:
        for contrast in plan["designs"][qid]["contrasts"]:
            key = (qid, contrast["anchor_id"], contrast["edge"])
            require(key not in planned and contrast["question_id"] == qid, "Duplicate/malformed planned heldout coordinate")
            left, right = plan["masks"][contrast["mask0_id"]], plan["masks"][contrast["mask1_id"]]
            edge = contrast["edge"]
            require(left[edge] == 0 and right[edge] == 1 and sum(a != b for a, b in zip(left, right)) == 1,
                    "Heldout credit direction is not the frozen off-to-on edge contrast")
            planned[key] = (left, right)
    require(len(planned) == len(predictions) == 2880 and set(Counter(key[0] for key in planned).values()) == {24},
            "Heldout prediction support is not 120 questions by 24 coordinates")
    seen, max_error = set(), 0.0
    for row in predictions:
        key = (row["question_id"], row["anchor_id"], row["edge"])
        require(key in planned and key not in seen and row.get("parameter_sha256") == parameters["parameter_sha256"],
                "Heldout prediction coordinate is duplicated or outside the frozen design")
        seen.add(key)
        left, right = planned[key]
        require(row["mask0"] == left and row["mask1"] == right and set(row["predictions"]) == {"B0", "C1", "C2"},
                "Heldout prediction masks/models differ from the frozen contrast")
        question_phi = np.asarray(phi[key[0]], dtype=np.float64)
        require(question_phi.shape == (9,) and np.isfinite(question_phi).all() and question_phi[-1] == 1,
                "Heldout feature vector differs from the frozen nine-dimensional transform")
        reproduced = {"B0": float(baseline[key[2]])}
        for name, order in (("C1", 1), ("C2", 2)):
            features = []
            for mask in (left, right):
                h = [1.0, *mask]
                if order == 2:
                    h.extend(mask[a] * mask[b] for a, b in itertools.combinations(range(12), 2))
                features.append(np.kron(question_phi, np.asarray(h, dtype=np.float64)))
            reproduced[name] = float((features[1] - features[0]) @ coefficients[name])
        for name, expected_value in reproduced.items():
            value = row["predictions"][name]
            require(finite(value) and abs(value - expected_value) <= 1e-9, "Independent owner prediction replay differs")
            max_error = max(max_error, abs(value - expected_value))
    require(seen == set(planned), "Heldout predictions omit frozen coordinates")
    return {"status": "PASS", "coordinates": len(seen), "questions": 120, "maximum_absolute_error": max_error,
            "absolute_tolerance": 1e-9, "implementation": "independent_phi_major_kron_formula", "heldout_labels_read": False}


CPU_WRAPPER = r'''
import fcntl,json,os,resource,subprocess,sys,time,uuid
from pathlib import Path
import psutil
p=json.loads(sys.argv[1]); folder=Path(p['folder']); lock=open(folder/'launch.lock','a+b')
def read(path): return json.loads(path.read_text())
def save(path,value):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 with open(tmp,'x') as h: json.dump(value,h,sort_keys=True,separators=(',',':'));h.write('\n');h.flush();os.fsync(h.fileno())
 os.replace(tmp,path)
def identity(proc):
 return {'pid':proc.pid,'create_time':proc.create_time(),'owner':proc.username(),'argv':proc.cmdline(),'cwd':proc.cwd()}
active=folder/('active_'+p['mode']+'.json'); attempt=folder/(p['attempt_id']+'.json')
fcntl.flock(lock,fcntl.LOCK_EX)
record=read(active)
if record['attempt_id']!=p['attempt_id']: raise RuntimeError('CPU launch intent changed')
record['wrapper']=identity(psutil.Process());record['state']='WRAPPER_STARTING';save(active,record);save(attempt,record)
env=os.environ.copy();env.update(p['env']);started=time.time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
try:
 with open(p['log'],'ab',buffering=0) as out:
  child=subprocess.Popen(p['argv'],cwd=p['root'],env=env,stdout=out,stderr=out)
  record.update(state='RUNNING',child=identity(psutil.Process(child.pid)),started_unix=started)
  save(active,record);save(attempt,record);fcntl.flock(lock,fcntl.LOCK_UN)
  code=child.wait()
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 record.update(state='EXITED',exit_code=code,finished_unix=time.time(),wall_seconds=time.time()-started,
               child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))
except BaseException as exc:
 record.update(state='WRAPPER_FAILED',error_type=type(exc).__name__,finished_unix=time.time(),child_cpu_seconds=None)
finally:
 fcntl.flock(lock,fcntl.LOCK_EX);save(attempt,record)
 if read(active)['attempt_id']==p['attempt_id']: save(active,record)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
'''


CPU_ENGINE = r'''
import fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
import psutil
p=json.loads(sys.argv[1]);root=Path(p['root']).resolve();planpath=root/p['plan_relative']
def sha(b): return hashlib.sha256(b).hexdigest()
def read(path): return json.loads(path.read_text())
def save(path,value):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 with open(tmp,'x') as h: json.dump(value,h,sort_keys=True,separators=(',',':'));h.write('\n');h.flush();os.fsync(h.fileno())
 os.replace(tmp,path)
if sha(planpath.read_bytes())!=p['plan_sha256']: raise RuntimeError('CPU plan bytes changed')
plan=read(planpath)
for field in ('source_manifest','pipeline_source_manifest'):
 for rel,item in plan[field].items():
  path=(root/rel).resolve()
  if not path.is_relative_to(root) or sha(path.read_bytes())!=item['sha256']: raise RuntimeError('CPU frozen source changed')
folder=root/'artifacts/private/g2'/plan['run_id']/'owner_cpu';folder.mkdir(parents=True,exist_ok=True)
lock=open(folder/'launch.lock','a+b');fcntl.flock(lock,fcntl.LOCK_EX)
owner=psutil.Process().username()
def live(info,record):
 if not info:return False
 try:
  proc=psutil.Process(info['pid'])
  if abs(proc.create_time()-info['create_time'])>.001:return False
  if proc.status()==psutil.STATUS_ZOMBIE:return False
  if (proc.username()!=owner or proc.username()!=info['owner'] or proc.cwd()!=str(root)
      or proc.cmdline()!=info['argv']):raise RuntimeError('CPU process identity mismatch')
  env=proc.environ()
  if any(env.get(k)!=v for k,v in record['env'].items()):raise RuntimeError('CPU process environment mismatch')
  return proc.is_running()
 except psutil.NoSuchProcess:return False
records=[]
for mode in ('predict','evaluate'):
 active=folder/('active_'+mode+'.json')
 if not active.exists():continue
 rec=read(active)
 if rec['plan_sha256']!=p['plan_sha256'] or rec['coordinator_sha256']!=p['coordinator_sha256']:raise RuntimeError('CPU launcher receipt belongs to different inputs')
 if live(rec.get('child'),rec) or live(rec.get('wrapper'),rec):
  print(json.dumps({'state':'RUNNING','mode':mode,'attempt_id':rec['attempt_id']}));sys.exit(0)
 if rec['state'] in ('LAUNCH_INTENT','WRAPPER_STARTING','RUNNING'):
  matches=[]
  for proc in psutil.process_iter(['username']):
   if proc.info['username']!=owner:continue
   try:
    if proc.environ().get('GRAPH_CREDIT_G2_LAUNCH_ID')==rec['attempt_id'] and proc.cwd()==str(root):
     if proc.cmdline() not in (rec['argv'],rec['wrapper_argv']):raise RuntimeError('Unrecognized CPU launch-tagged process')
     matches.append(proc)
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  if matches:
   if len(matches)>2:raise RuntimeError('Multiple unresolved CPU launch processes')
   print(json.dumps({'state':'RUNNING_RECOVERED','mode':mode,'attempt_id':rec['attempt_id']}));sys.exit(0)
  if rec['state']=='LAUNCH_INTENT':
   print(json.dumps({'state':'RECOVERY_REQUIRED','reason':'INTENT_WITHOUT_IDENTIFIABLE_PROCESS','mode':mode}));sys.exit(0)
 records.append(rec)
if p['action']=='probe':
 print(json.dumps({'state':'IDLE','records':records}));sys.exit(0)
if p['mode']=='predict':
 for name in ('HELDOUT_LABEL_RELEASE.json','FIRST_HELDOUT_LABELS_OPENED.json','G2_ANALYSIS.json'):
  if (root/plan['output_dir']/name).exists() and not (root/plan['output_dir']/'PREDICTIONS_FROZEN.json').exists():raise RuntimeError('Cannot refit after heldout exposure')
attempts=[f for f in folder.glob('*.json') if f.stem.startswith(p['mode']+'_')]
if len(attempts)>=3:raise RuntimeError('CPU recovery attempt bound reached')
attempt_id=p['mode']+'_'+uuid.uuid4().hex[:20]
env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','GRAPH_CREDIT_G2_LAUNCH_ID':attempt_id}
argv=[p['python'],'-u',str(root/p['workflow_relative']),'--root',str(root),'--plan',str(planpath),'--mode',p['mode']]
payload={'folder':str(folder),'mode':p['mode'],'attempt_id':attempt_id,'root':str(root),'argv':argv,'env':env,'log':str(folder/(attempt_id+'.log'))}
wrapper_argv=[p['python'],'-u','-c',p['wrapper_code'],json.dumps(payload,separators=(',',':'))]
record={'schema':'graph_credit_g2_owner_cpu_launch_v1','state':'LAUNCH_INTENT','attempt_id':attempt_id,'mode':p['mode'],
 'plan_sha256':p['plan_sha256'],'coordinator_sha256':p['coordinator_sha256'],'launcher_sha256':p['launcher_sha256'],
 'workflow_sha256':p['workflow_sha256'],'argv':argv,'wrapper_argv':wrapper_argv,'env':env,'created_unix':time.time()}
active=folder/('active_'+p['mode']+'.json');save(active,record);save(folder/(attempt_id+'.json'),record)
wrapper_env=os.environ.copy();wrapper_env.update(env)
with open(folder/(attempt_id+'.wrapper.log'),'ab',buffering=0) as out:
 child=subprocess.Popen(wrapper_argv,cwd=root,env=wrapper_env,stdout=out,stderr=out,start_new_session=True)
 proc=psutil.Process(child.pid)
 record['wrapper']={'pid':proc.pid,'create_time':proc.create_time(),'owner':proc.username(),'argv':proc.cmdline(),'cwd':proc.cwd()}
 save(active,record);save(folder/(attempt_id+'.json'),record)
print(json.dumps({'state':'STARTED','mode':p['mode'],'attempt_id':attempt_id,'wrapper_pid':child.pid}))
'''


def cpu_command(remote_python, payload):
    require(isinstance(remote_python, str) and remote_python.startswith("/") and "\x00" not in remote_python,
            "Remote Python must be an explicit absolute executable path")
    return shlex.join([remote_python, "-u", "-c", CPU_ENGINE, canonical(payload)])


class Coordinator:
    def __init__(self, args, client_factory=None):
        import psutil
        self.args = args
        self.root = Path(args.root).resolve(strict=True)
        self.remote_root = PurePosixPath(args.remote_root)
        require(self.remote_root.is_absolute() and ".." not in self.remote_root.parts, "Remote project root is invalid")
        self.helpers = unit_mirror.load_helpers(self.root)
        info = self.root.lstat()
        self.root_identity = (info.st_dev, info.st_ino)
        self.output = "results/g2/" + RUN_ID
        self.private = "artifacts/private/g2/" + RUN_ID
        self.plan_relative = self.output + "/frozen_plan.json"
        self.plan_content = self.path(self.plan_relative).read_bytes()
        self.plan, self.plan_hash = decode(self.plan_content), byte_sha(self.plan_content)
        self.source_hash, self.pipeline_hash = digest(self.plan["source_manifest"]), digest(self.plan["pipeline_source_manifest"])
        require(self.plan.get("schema") == "graph_credit_g2_plan_v1" and self.plan.get("status") == "PLAN_FROZEN"
                and self.plan["run_id"] == RUN_ID and self.plan["output_dir"] == self.output and self.plan.get("labels_read") is False,
                "Coordinator requires the final canonical G2 plan")
        marker = decode(self.path(self.output + "/PLAN_FROZEN.json").read_bytes())
        require(marker.get("state") == "PLAN_FROZEN" and marker.get("plan_sha256") == self.plan_hash
                and marker.get("source_manifest_sha256") == self.source_hash
                and marker.get("pipeline_source_manifest_sha256") == self.pipeline_hash, "Coordinator plan marker binding failed")
        self.coordinator_sha = byte_sha(Path(__file__).read_bytes())
        self.factory = client_factory or strict_client_factory(args)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        process = psutil.Process()
        self.state = {"schema": "graph_credit_g2_local_coordinator_v1", "run_id": RUN_ID, "pid": process.pid,
                      "process_create_time": process.create_time(), "owner_username": process.username(),
                      "plan_sha256": self.plan_hash, "coordinator_sha256": self.coordinator_sha,
                      "stage": "STARTING", "final_full_verification_required": True, "scientific_completion": False,
                      "local_control_sources": {"coordinator.py": self.coordinator_sha,
                                                "mirror.py": byte_sha(Path(unit_mirror.__file__).read_bytes()),
                                                "mirror_only.py": unit_mirror.HELPER_SHA256},
                      "labels_read": False, "first_gold_opened_unix": None}
        self.global_hash = None
        self.prediction_hash = None
        self.global_marker = None
        self.prediction_marker = None
        self.heartbeat_error = None
        self.claims_ready = False
        self.control_snapshots = (("coordinator", Path(__file__).read_bytes()),
                                  ("mirror", Path(unit_mirror.__file__).read_bytes()),
                                  ("mirror_only", self.path(unit_mirror.HELPER_PATH).read_bytes()))

    def path(self, name):
        return self.helpers.local_path(self.root, relative(name), root_identity=self.root_identity)

    def publish(self, name, content, *, exclusive=False):
        target = self.path(name)
        if exclusive and target.exists():
            require(target.read_bytes() == content, "Immutable owner artifact already has different bytes")
            return
        self.helpers.atomic_bytes(target, content, root=self.root, root_identity=self.root_identity)

    def save(self, name, value, *, exclusive=False):
        self.publish(name, encoded(value), exclusive=exclusive)

    def persist_verification(self, name, value):
        """Retain the first completed proof and append each actual verification."""
        changing_fields = {"verified_unix", "coordinator_sha256", "maximum_absolute_error"}
        require(value.get("status") == "PASS" and finite(value.get("verified_unix")) and value["verified_unix"] > 0,
                "Current owner verification has an invalid completion time")
        if "maximum_absolute_error" in value:
            require(finite(value.get("maximum_absolute_error")) and finite(value.get("absolute_tolerance"))
                    and 0 <= value["maximum_absolute_error"] <= value["absolute_tolerance"],
                    "Current owner prediction replay violates its absolute tolerance")
        path = self.path(name)
        is_first = not path.exists()
        if is_first:
            self.save(name, value, exclusive=True)
        first_raw = path.read_bytes()
        first = decode(first_raw)
        stable = {key: item for key, item in value.items() if key not in changing_fields}
        original_stable = {key: item for key, item in first.items() if key not in changing_fields}
        require(stable == original_stable, "Existing owner verification proof binds different frozen inputs or gates")
        require(finite(first.get("verified_unix")) and first["verified_unix"] > 0,
                "Original owner verification completion time is invalid")
        for proof in (first, value):
            require(isinstance(proof.get("coordinator_sha256"), str) and bool(proof["coordinator_sha256"]),
                    "Owner verification proof has no control-source identity")
            if "maximum_absolute_error" in value:
                require(finite(proof.get("maximum_absolute_error"))
                        and 0 <= proof["maximum_absolute_error"] <= proof["absolute_tolerance"],
                        "Owner prediction replay proof violates its absolute tolerance")
        attempt = {"schema": "graph_credit_g2_owner_verification_attempt_v1", "plan_sha256": self.plan_hash,
                   "proof_path": name, "first_proof_sha256": byte_sha(first_raw), "is_first_verification": is_first,
                   "verified_unix": value["verified_unix"], "coordinator_sha256": self.coordinator_sha,
                   "new_verification": value}
        self.save(self.private + "/verification_attempts/" + uuid.uuid4().hex[:20] + ".json", attempt, exclusive=True)
        return first

    def status(self, stage=None, **values):
        with self.lock:
            if self.state["stage"] in (FINAL, "FAILED", "ATTENTION_REQUIRED") and stage not in (None, self.state["stage"]):
                return
            if stage is not None:
                self.state["stage"] = stage
            self.state.update(values, updated_unix=time.time())
            for attempt in range(5):
                try:
                    self.save("artifacts/monitoring/g2/COORDINATOR_STATUS.json", self.state)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(.02 * (attempt + 1))

    def heartbeat(self):
        while not self.stop.wait(5):
            try:
                self.status()
            except BaseException as exc:
                self.heartbeat_error = exc
                self.stop.set()
                return

    def remote_path(self, name):
        return str(self.remote_root / relative(name))

    def get(self, sftp, name, *, expected_sha=None, optional=False):
        content = optional_remote(sftp, self.remote_path(name)) if optional else remote_bytes(sftp, self.remote_path(name))
        if content is None:
            return None
        if expected_sha is not None:
            require(byte_sha(content) == expected_sha, "Remote artifact byte SHA differs from its frozen binding")
        self.publish(name, content)
        return content

    def verify_sources(self, sftp):
        require(remote_bytes(sftp, self.remote_path(self.plan_relative)) == self.plan_content, "Remote and local frozen plan bytes differ")
        for field in ("source_manifest", "pipeline_source_manifest"):
            for name, expected in self.plan[field].items():
                require(byte_sha(self.path(name).read_bytes()) == expected["sha256"], "Local frozen scientific source changed")
                require(byte_sha(remote_bytes(sftp, self.remote_path(name))) == expected["sha256"], "Remote frozen scientific source changed")

    def controls(self):
        values = {}
        with connection(self.factory) as (_, sftp):
            require(not remote_exists(sftp, self.remote_path("data/derived/calibration_labels.jsonl")),
                    "The complete calibration label file must remain local to the data owner")
            for name in CONTROL_NAMES:
                content = self.get(sftp, self.output + "/" + name, optional=True)
                if content is not None:
                    value = decode(content)
                    require(value.get("plan_sha256") == self.plan_hash, "Remote control artifact belongs to a different plan")
                    if control_state(value) in ("FAILED", "ATTENTION_REQUIRED", "STOPPING_AFTER_FAILURE"):
                        raise IntegrityError("Remote G2 process reported a failure")
                    values[name] = (value, byte_sha(content))
            if "GLOBAL_OUTPUTS_FROZEN.json" not in values:
                for name in self.plan["label_release"].values():
                    if isinstance(name, str) and name.endswith(".jsonl"):
                        require(not remote_exists(sftp, self.remote_path(name)),
                                "A partition label file appeared before the global output freeze")
            self.claims_ready = remote_exists(sftp, self.remote_path(self.output + "/claims"))
        if ("PREDICTIONS_FROZEN.json" not in values
                and any(name in values for name in ("G2_ANALYSIS.json",))):
            raise IntegrityError("Heldout analysis exists without the immutable prediction freeze")
        return values

    def verify_global_barrier(self, marker, marker_hash):
        validate_global(marker, self.plan, self.plan_hash)
        with connection(self.factory) as (_, sftp):
            self.verify_sources(sftp)
            artifacts = {}
            for name in ("observations", "physical_cost_audit"):
                item = marker["artifacts"][name]
                content = self.get(sftp, item["path"], expected_sha=item["sha256"])
                require(len(content) == item["bytes"], "Frozen canonical artifact size mismatch")
                artifacts[name] = content
            for field, hash_field in (("path", "sha256"), ("receipt", "receipt_sha256")):
                item = self.plan["features"]
                content = remote_bytes(sftp, self.remote_path(item[field]))
                require(byte_sha(content) == item[hash_field], "Remote label-free feature artifact changed")
                require(byte_sha(self.path(item[field]).read_bytes()) == item[hash_field], "Local label-free feature artifact changed")
            require(byte_sha(remote_bytes(sftp, self.remote_path(self.output + "/GLOBAL_OUTPUTS_FROZEN.json"))) == marker_hash,
                    "Global output marker changed while validating its artifacts")
        validate_observations(artifacts["observations"], self.plan, self.plan_hash)
        self.persist_verification(self.output + "/LOCAL_GLOBAL_BARRIER_VERIFIED.json", {"schema": "graph_credit_g2_owner_global_barrier_v1",
                  "status": "PASS", "plan_sha256": self.plan_hash, "global_outputs_frozen_sha256": marker_hash,
                  "source_manifest_sha256": self.source_hash, "pipeline_source_manifest_sha256": self.pipeline_hash,
                  "observations": 51970, "units": 2000, "verified_unix": time.time(), "labels_read": False,
                  "coordinator_sha256": self.coordinator_sha})
        self.global_marker, self.global_hash = marker, marker_hash

    def owner_labels(self):
        require(self.global_hash is not None, "Owner label access requires the verified global output barrier")
        split_name = self.output + "/LOCAL_LABEL_SPLIT_FROZEN.json"
        first_name = self.output + "/LOCAL_DATA_OWNER_FIRST_LABEL_READ.json"
        if self.path(split_name).exists():
            receipt = decode(self.path(split_name).read_bytes())
            require(receipt["plan_sha256"] == self.plan_hash and receipt["global_outputs_frozen_sha256"] == self.global_hash,
                    "Existing owner label split belongs to a different frozen generation")
            require(byte_sha(self.path(first_name).read_bytes()) == receipt["first_owner_label_read_sha256"], "First owner read event changed")
            contents = {part: self.path(self.plan["label_release"][part]).read_bytes() for part in ("train", "tune", "test")}
            require(all(byte_sha(content) == receipt["labels"][part]["sha256"] for part, content in contents.items()),
                    "A persisted local label partition changed")
            event = decode(self.path(first_name).read_bytes())
            self.status(labels_read=True, first_gold_opened_unix=event["first_gold_opened_unix"])
            return contents, receipt
        source_name = "data/derived/calibration_labels.jsonl"
        prior = decode(self.path(first_name).read_bytes()) if self.path(first_name).exists() else None
        with self.path(source_name).open("rb") as handle:
            info = os.fstat(handle.fileno())
            fingerprint = {"path": source_name, "bytes_at_open": info.st_size, "device": info.st_dev,
                           "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
            if prior is None:
                event = {"schema": "graph_credit_g2_data_owner_first_label_read_v1", "event": "HANDLE_OPENED_BEFORE_FIRST_READ",
                         "run_id": RUN_ID, "plan_sha256": self.plan_hash, "global_outputs_frozen_sha256": self.global_hash,
                         "source_manifest_sha256": self.source_hash, "pipeline_source_manifest_sha256": self.pipeline_hash,
                         "coordinator_sha256": self.coordinator_sha,
                         "source": fingerprint, "first_gold_opened_unix": time.time()}
                self.save(first_name, event, exclusive=True)
            else:
                require(prior.get("plan_sha256") == self.plan_hash and prior.get("global_outputs_frozen_sha256") == self.global_hash
                        and prior.get("source") == fingerprint and prior.get("source_manifest_sha256") == self.source_hash
                        and prior.get("pipeline_source_manifest_sha256") == self.pipeline_hash,
                        "Owner first-read recovery inputs changed")
                event = prior
            event_hash = byte_sha(self.path(first_name).read_bytes())
            self.save(self.private + "/owner_label_read_attempts/" + uuid.uuid4().hex[:20] + ".json",
                      {"first_event_sha256": event_hash, "plan_sha256": self.plan_hash, "source": fingerprint,
                       "handle_opened_before_read_unix": time.time(), "is_first_access": prior is None}, exclusive=True)
            content = handle.read()
            after = os.fstat(handle.fileno())
            require(after.st_size == info.st_size and after.st_mtime_ns == info.st_mtime_ns,
                    "Original calibration labels changed while the owner read the opened handle")
        contents = split_original_labels(content, self.plan["partitions"])
        for part, raw in contents.items():
            self.publish(self.plan["label_release"][part], raw, exclusive=True)
        receipt = {"schema": "graph_credit_g2_owner_label_split_v1", "run_id": RUN_ID, "plan_sha256": self.plan_hash,
                   "global_outputs_frozen_sha256": self.global_hash, "original_labels_sha256": byte_sha(content),
                   "first_owner_label_read_sha256": event_hash, "partitions_sha256": digest(self.plan["partitions"]),
                   "labels": {part: {"path": self.plan["label_release"][part], "sha256": byte_sha(raw), "bytes": len(raw),
                                     "questions": {"train": 200, "tune": 80, "test": 120}[part]} for part, raw in contents.items()},
                   "created_unix": time.time(), "source_rows_preserved_verbatim": True, "test_uploaded": False}
        receipt["coordinator_sha256"] = self.coordinator_sha
        self.save(split_name, receipt, exclusive=True)
        self.status(labels_read=True, first_gold_opened_unix=event["first_gold_opened_unix"])
        return contents, receipt

    def release(self, stage, labels, split):
        train = stage == "train_tune"
        parts = ("train", "tune") if train else ("test",)
        filename = "TRAIN_TUNE_LABEL_RELEASE.json" if train else "HELDOUT_LABEL_RELEASE.json"
        binding_key = "global_outputs_frozen_sha256" if train else "predictions_frozen_sha256"
        binding = self.global_hash if train else self.prediction_hash
        require(binding is not None, "Label release lacks its verified freeze barrier")
        expected = {"state": "TRAIN_TUNE_LABELS_RELEASED" if train else "HELDOUT_LABELS_RELEASED",
                    "plan_sha256": self.plan_hash, binding_key: binding}
        expected.update({part + "_labels_sha256": byte_sha(labels[part]) for part in parts} if train
                        else {"heldout_labels_sha256": byte_sha(labels["test"])})
        name = self.output + "/" + filename
        with connection(self.factory) as (_, sftp):
            remote_prior = optional_remote(sftp, self.remote_path(name))
            local_prior = self.path(name).read_bytes() if self.path(name).exists() else None
            require(remote_prior is None or local_prior is None or remote_prior == local_prior,
                    "Local and remote release history bytes differ")
            prior = local_prior if local_prior is not None else remote_prior
            if prior is not None:
                value = decode(prior)
                require(all(value.get(key) == item for key, item in expected.items()) and finite(value.get("released_unix")),
                        "Existing label release differs from the current owner inputs")
                content = prior
            else:
                content = encoded({"schema": "graph_credit_g2_owner_label_release_v1", "run_id": RUN_ID, **expected,
                                   "coordinator_sha256": self.coordinator_sha,
                                   "released_unix": time.time(), "owner_first_label_read_sha256": split["first_owner_label_read_sha256"],
                                   "local_label_split_sha256": byte_sha(self.path(self.output + "/LOCAL_LABEL_SPLIT_FROZEN.json").read_bytes())})
            for part in parts:
                upload_once(sftp, self.remote_root, self.plan["label_release"][part], labels[part])
            self.publish(name, content, exclusive=True)
            upload_once(sftp, self.remote_root, name, content)
        return byte_sha(content)

    def cpu(self, mode, *, action="launch"):
        scripts = [name for name in self.plan["pipeline_source_manifest"] if PurePosixPath(name).name == "workflow.py"]
        require(len(scripts) == 1, "Exactly one frozen workflow entry is required")
        payload = {"root": str(self.remote_root), "plan_relative": self.plan_relative, "plan_sha256": self.plan_hash,
                   "coordinator_sha256": self.coordinator_sha, "workflow_relative": scripts[0],
                   "workflow_sha256": self.plan["pipeline_source_manifest"][scripts[0]]["sha256"],
                   "launcher_sha256": byte_sha(CPU_ENGINE.encode()), "wrapper_code": CPU_WRAPPER,
                   "python": self.args.remote_python, "mode": mode, "action": action}
        with connection(self.factory) as (client, _):
            stdin, stdout, stderr = client.exec_command(cpu_command(self.args.remote_python, payload), timeout=30)
            stdin.close()
            stdout.channel.settimeout(30)
            try:
                content, errors = stdout.read(1024 * 1024 + 1), stderr.read(8193)
                require(len(content) <= 1024 * 1024 and len(errors) <= 8192, "Remote CPU launcher exceeded its response cap")
                require(stdout.channel.recv_exit_status() == 0, "Remote CPU launcher failed its identity/input check")
                result = decode(content)
            finally:
                for handle in (stdout, stderr):
                    try:
                        handle.close()
                    except Exception:
                        pass
        require(result.get("state") != "RECOVERY_REQUIRED", "CPU launch intent requires independent process review")
        self.save(self.private + "/cpu_controller_checks/" + uuid.uuid4().hex[:20] + ".json",
                  {"plan_sha256": self.plan_hash, "mode": mode, "action": action, "result": result,
                   "checked_unix": time.time(), "coordinator_sha256": self.coordinator_sha})
        return result

    def feature_map(self):
        import numpy as np
        descriptor = self.plan["features"]
        require(byte_sha(self.path(descriptor["path"]).read_bytes()) == descriptor["sha256"], "Local feature bytes changed")
        receipt_raw = self.path(descriptor["receipt"]).read_bytes()
        require(byte_sha(receipt_raw) == descriptor["receipt_sha256"], "Local feature receipt changed")
        receipt = decode(receipt_raw)
        require(receipt.get("status") == "PASS_LABEL_FREE" and receipt.get("labels_read") is False
                and receipt.get("pca_fit_questions") == 200 and receipt.get("transformed_questions") == 400
                and receipt.get("dimensions") == 9, "Frozen feature receipt does not establish train-only PCA")
        with np.load(self.path(descriptor["path"]), allow_pickle=False) as values:
            ids, phi = values["question_ids"].tolist(), np.asarray(values["phi"], dtype=np.float64)
        require(len(ids) == len(set(ids)) == 400 and set(ids) == set(self.plan["partitions"])
                and phi.shape == (400, 9) and np.isfinite(phi).all() and np.all(phi[:, -1] == 1), "Local feature coordinates changed")
        require(receipt["train_ids_sha256"] == digest([qid for qid in ids if self.plan["partitions"][qid] == "train"]),
                "PCA fitting population differs from the frozen train split")
        return dict(zip(ids, phi))

    def verify_prediction_barrier(self, marker, marker_hash):
        require(self.global_hash is not None, "Prediction verification requires the global barrier")
        with connection(self.factory) as (_, sftp):
            self.verify_sources(sftp)
            parameter_raw = self.get(sftp, self.output + "/parameters.json", expected_sha=marker["parameters_sha256"])
            prediction_raw = self.get(sftp, self.output + "/heldout_predictions.json", expected_sha=marker["predictions_sha256"])
            self.get(sftp, self.output + "/FIRST_TRAIN_TUNE_LABELS_OPENED.json", expected_sha=marker["first_train_tune_labels_opened_sha256"])
            release_raw = self.get(sftp, self.output + "/TRAIN_TUNE_LABEL_RELEASE.json", expected_sha=marker["train_tune_label_release_sha256"])
            require(byte_sha(remote_bytes(sftp, self.remote_path(self.output + "/PREDICTIONS_FROZEN.json"))) == marker_hash,
                    "Prediction marker changed during independent owner replay")
        release = decode(release_raw)
        require(release["global_outputs_frozen_sha256"] == self.global_hash and release["plan_sha256"] == self.plan_hash
                and marker["training_labels_sha256"] == {part: release[part + "_labels_sha256"] for part in ("train", "tune")},
                "Frozen predictor does not bind the owner train/tune release")
        replay = independent_predictions(self.plan, self.plan_hash, self.global_hash, marker,
                                         decode(parameter_raw), decode(prediction_raw), self.feature_map())
        self.persist_verification(self.output + "/LOCAL_PREDICTION_REPLAY.json", {"schema": "graph_credit_g2_owner_prediction_replay_v1", **replay,
                  "plan_sha256": self.plan_hash, "predictions_frozen_sha256": marker_hash,
                  "parameters_sha256": marker["parameters_sha256"], "predictions_sha256": marker["predictions_sha256"],
                  "verified_unix": time.time(), "coordinator_sha256": self.coordinator_sha})
        self.prediction_marker, self.prediction_hash = marker, marker_hash

    def final_metadata(self, analysis, analysis_hash):
        require(self.prediction_hash is not None, "Final mirror requires independently verified frozen predictions")
        require(analysis.get("schema") == "graph_credit_g2_heldout_evaluation_v1"
                and analysis.get("analysis_semantic_sha256") == digest({key: value for key, value in analysis.items() if key != "analysis_semantic_sha256"})
                and analysis.get("plan_sha256") == self.plan_hash and analysis.get("run_id") == RUN_ID
                and analysis.get("source_manifest_sha256") == self.source_hash
                and analysis.get("pipeline_source_manifest_sha256") == self.pipeline_hash
                and analysis.get("predictions_frozen_sha256") == self.prediction_hash
                and analysis.get("predictions_sha256") == self.prediction_marker["predictions_semantic_sha256"]
                and analysis.get("scientific_gate") in {"G2_PASS", "INSUFFICIENT_PREDICTIVE_GAIN"},
                "Completed G2 analysis does not bind the frozen prediction/protocol")
        coverage = analysis.get("coverage", {})
        require(coverage == {"unique_graph_outputs": 51970, "question_seed_units": 2000,
                             "train_questions": 200, "tune_questions": 80, "heldout_questions": 120,
                             "heldout_credit_targets": 2880, "complete": True}, "Analysis coverage is incomplete")
        require(analysis["primary"]["question_count"] == 120 and analysis["contrasts_per_question"] == 24
                and analysis["quality_seed_count"] == 5 and analysis["negative_risks_preserved"] is True,
                "Analysis omitted the frozen test support or signed risks")
        pinned = {self.output + "/G2_COST_SUMMARY.json": analysis["cost_summary_sha256"],
                  self.output + "/FIRST_HELDOUT_LABELS_OPENED.json": analysis["first_heldout_labels_opened_sha256"],
                  self.output + "/HELDOUT_LABEL_RELEASE.json": analysis["heldout_label_release_sha256"]}
        descriptors = {}
        def record(name, raw):
            descriptors[name] = {"path": name, "sha256": byte_sha(raw), "bytes": len(raw)}
        with connection(self.factory) as (_, sftp):
            self.verify_sources(sftp)
            for name, expected in pinned.items():
                record(name, self.get(sftp, name, expected_sha=expected))
            release = decode(self.path(self.output + "/HELDOUT_LABEL_RELEASE.json").read_bytes())
            split_receipt = decode(self.path(self.output + "/LOCAL_LABEL_SPLIT_FROZEN.json").read_bytes())
            require(release["heldout_labels_sha256"] == analysis["heldout_labels_sha256"]
                    == split_receipt["labels"]["test"]["sha256"]
                    and release["predictions_frozen_sha256"] == self.prediction_hash,
                    "Analysis heldout release differs from the owner split")
            names = set(self.output + "/" + name for name in (
                *CONTROL_NAMES, "PLAN_FROZEN.json", "frozen_plan.json", "GLOBAL_QUALIFICATION.json", "observations.jsonl",
                "PHYSICAL_COST_AUDIT.json", "parameters.json", "heldout_predictions.json", "TRAIN_TUNE_LABEL_RELEASE.json",
                "HELDOUT_LABEL_RELEASE.json", "FIRST_TRAIN_TUNE_LABELS_OPENED.json", "FIRST_HELDOUT_LABELS_OPENED.json", "G2_COST_SUMMARY.json"))
            names.update((self.plan["features"]["path"], self.plan["features"]["receipt"]))
            for worker in self.plan["qualification"]["worker_ids"]:
                names.update(self.output + "/segments/" + worker + "/" + name
                             for name in ("STATUS.json", "run_manifest.json", "physical_calls.jsonl"))
                formal_index = self.output + "/segments/" + worker + "/raw_executions.jsonl"
                if remote_exists(sftp, self.remote_path(formal_index)):
                    names.add(formal_index)
                else:
                    state = decode(remote_bytes(sftp, self.remote_path(self.output + "/segments/" + worker + "/STATUS.json")))
                    require(state.get("observations") == 0, "Nonempty formal worker has no raw execution ledger")
                base = self.output + "/qualification/" + worker
                names.update(base + "/" + filename for filename in ("run_manifest.json", "QUALIFICATION.json", "physical_calls.jsonl", "raw_executions.jsonl"))
                # Enumerate only the prespecified engineering outputs. Formal
                # unit trees use the completed-unit tar transport.
                for unit in self.plan["qualification"]["units"]:
                    uid = unit_mirror.unit_id(unit["question_id"], unit["seed"])
                    unit_base = base + "/units/" + uid
                    names.update(unit_base + "/" + filename for filename in ("UNIT_PLAN.json", "UNIT_OUTPUTS_FROZEN.json"))
                    for item in unit_mirror.schedule_for(unit):
                        mask_id = "".join(map(str, item["mask"]))
                        names.update(unit_base + "/" + folder + "/" + mask_id + ".json" for folder in ("raw", "executions"))
                    names.update(base + "/independent_full_replays/" + uid + "_" + str(index) + ".json"
                                 for index in self.plan["qualification"]["independent_full_replay_request_indices"])
            for folder in (self.output + "/cpu_attempts", self.output + "/label_access_attempts", self.private + "/owner_cpu"):
                if remote_exists(sftp, self.remote_path(folder)):
                    for item in sftp.listdir_attr(self.remote_path(folder)):
                        require("/" not in item.filename and "\\" not in item.filename, "Unsafe engineering metadata filename")
                        if item.filename.endswith(".json"):
                            require(stat.S_ISREG(item.st_mode), "Engineering event is linked or is not a regular file")
                            names.add(folder + "/" + relative(item.filename))
            for name in sorted(names):
                raw = self.get(sftp, name)
                record(name, raw)
            require(byte_sha(remote_bytes(sftp, self.remote_path(self.output + "/G2_ANALYSIS.json"))) == analysis_hash
                    and byte_sha(remote_bytes(sftp, self.remote_path(self.output + "/PREDICTIONS_FROZEN.json"))) == self.prediction_hash
                    and byte_sha(remote_bytes(sftp, self.remote_path(self.output + "/GLOBAL_OUTPUTS_FROZEN.json"))) == self.global_hash,
                    "A frozen phase marker changed while final metadata was mirrored")
        # Independently recheck all artifact bindings after the complete copy.
        for key, item in self.global_marker["artifacts"].items():
            require(byte_sha(self.path(item["path"]).read_bytes()) == item["sha256"], "Canonical global artifact changed during final mirror")
        for name, expected in (("parameters.json", self.prediction_marker["parameters_sha256"]),
                               ("heldout_predictions.json", self.prediction_marker["predictions_sha256"]),
                               ("G2_ANALYSIS.json", analysis_hash)):
            require(byte_sha(self.path(self.output + "/" + name).read_bytes()) == expected, "Frozen prediction/analysis bytes changed during final mirror")
        audit = decode(self.path(self.output + "/PHYSICAL_COST_AUDIT.json").read_bytes())
        for field in ("physical_ledgers", "qualification_ledgers", "unit_receipts", "qualification_receipts", "worker_runtime_manifests", "worker_terminal_states"):
            for item in audit[field]:
                raw = self.path(item["path"]).read_bytes()
                require(byte_sha(raw) == item["sha256"] and len(raw) == item["bytes"], "Final cost/engineering provenance bytes changed")
                record(item["path"], raw)
        for filename in ("LOCAL_GLOBAL_BARRIER_VERIFIED.json", "LOCAL_DATA_OWNER_FIRST_LABEL_READ.json", "LOCAL_LABEL_SPLIT_FROZEN.json", "LOCAL_PREDICTION_REPLAY.json"):
            name = self.output + "/" + filename
            record(name, self.path(name).read_bytes())
        for folder in (self.private + "/cpu_controller_checks", self.private + "/owner_label_read_attempts", self.private + "/verification_attempts"):
            path = self.path(folder + "/probe.json").parent
            for entry in path.glob("*.json"):
                name = entry.relative_to(self.root).as_posix()
                record(name, self.path(name).read_bytes())
        for entry in self.path(self.private + "/control_sources/probe.py").parent.glob("*.py"):
            name = entry.relative_to(self.root).as_posix()
            record(name, self.path(name).read_bytes())
        receipt = {"schema": "graph_credit_g2_private_metadata_mirror_v1", "run_id": RUN_ID, "plan_sha256": self.plan_hash,
                   "descriptors": [descriptors[name] for name in sorted(descriptors)], "snapshot_unix": time.time(),
                   "labels_read": False, "final_full_verification_required": True, "coordinator_sha256": self.coordinator_sha}
        receipt_name = self.output + "/PRIVATE_METADATA_MIRROR_RECEIPT.json"
        self.save(receipt_name, receipt)
        units = []
        for unit in self.plan["units"]:
            uid = unit_mirror.unit_id(unit["question_id"], unit["seed"])
            name = self.private + "/mirrored_units/" + uid + ".json"
            raw = self.path(name).read_bytes()
            value = decode(raw)
            require(value.get("unit_id") == uid and value.get("question_id") == unit["question_id"] and value.get("seed") == unit["seed"]
                    and value.get("plan_sha256") == self.plan_hash and value.get("run_id") == RUN_ID,
                    "Final mirror receipt has a missing or foreign unit")
            units.append({"path": name, "sha256": byte_sha(raw), "bytes": len(raw)})
        require(len(units) == 2000 and len({item["path"] for item in units}) == 2000, "Final local unit receipt coverage is incomplete")
        complete = {"schema": "graph_credit_g2_local_mirror_complete_v1", "state": "LOCAL_MIRROR_COMPLETE", "run_id": RUN_ID,
                    "plan_sha256": self.plan_hash, "source_manifest_sha256": self.source_hash, "pipeline_source_manifest_sha256": self.pipeline_hash,
                    "mirrored_units": 2000, "unit_receipts_sha256": digest(sorted(units, key=lambda row: row["path"])),
                    "global_outputs_frozen_sha256": self.global_hash, "predictions_frozen_sha256": self.prediction_hash,
                    "analysis_sha256": analysis_hash, "private_metadata_receipt_sha256": byte_sha(self.path(receipt_name).read_bytes()),
                    "completed_unix": time.time(), "final_full_verification_required": True, "scientific_completion": False}
        complete["coordinator_sha256"] = self.coordinator_sha
        self.save(self.output + "/LOCAL_MIRROR_COMPLETE.json", complete)
        return complete

    def cycle(self):
        controls = self.controls()
        self.status("REMOTE_RUNNING_AND_MIRRORING", remote_states={name: control_state(value) for name, (value, _) in controls.items()})
        mirrored = None
        if self.claims_ready:
            mirrored = unit_mirror.mirror_units(self.root, str(self.remote_root), self.plan, self.factory, max_workers=4,
                progress_callback=lambda value: self.status(current_operation="MIRRORING_COMPLETED_UNITS", mirror_progress=value))
            self.status(mirror=mirrored)
            if mirrored["status"] == "ATTENTION_REQUIRED":
                network_types = {"ConnectionError", "TimeoutError", "EOFError", "SSHException", "NoValidConnectionsError",
                                 "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError"}
                if (not mirrored.get("callback_errors") and mirrored["failures"]
                        and all(item["error_type"] in network_types and item["last_error_type"] in network_types
                                for item in mirrored["failures"])):
                    raise ConnectionError("Completed-unit transport exhausted its bounded network retries")
                raise IntegrityError("Completed-unit mirror encountered an integrity/authentication/publication failure")
        if "GLOBAL_OUTPUTS_FROZEN.json" not in controls:
            return False
        marker, marker_hash = controls["GLOBAL_OUTPUTS_FROZEN.json"]
        if self.global_hash is None:
            self.status("VERIFYING_GLOBAL_OUTPUTS")
            self.verify_global_barrier(marker, marker_hash)
        require(marker_hash == self.global_hash, "Global output freeze changed after owner verification")
        labels, split = self.owner_labels()
        self.release("train_tune", labels, split)
        if "PREDICTIONS_FROZEN.json" not in controls:
            with connection(self.factory) as (_, sftp):
                require(not remote_exists(sftp, self.remote_path(self.output + "/HELDOUT_LABEL_RELEASE.json"))
                        and not remote_exists(sftp, self.remote_path(self.output + "/FIRST_HELDOUT_LABELS_OPENED.json")),
                        "Heldout exposure prevents recovery without the frozen prediction marker")
            result = self.cpu("predict")
            self.status("PREDICT_CPU_RUNNING", cpu=result)
            return False
        prediction, prediction_hash = controls["PREDICTIONS_FROZEN.json"]
        if self.prediction_hash is None:
            self.status("INDEPENDENT_PREDICTION_REPLAY")
            self.verify_prediction_barrier(prediction, prediction_hash)
        require(prediction_hash == self.prediction_hash, "Prediction freeze changed after independent owner replay")
        self.release("heldout", labels, split)
        if "G2_ANALYSIS.json" not in controls:
            result = self.cpu("evaluate")
            self.status("EVALUATE_CPU_RUNNING", cpu=result)
            return False
        idle = self.cpu("evaluate", action="probe")
        if idle["state"] != "IDLE":
            self.status("WAITING_FINAL_CPU_RECEIPT", cpu=idle)
            return False
        require(mirrored is not None and mirrored["status"] == "PASS"
                and mirrored["mirrored_new"] + mirrored["skipped_cached"] == 2000,
                "Completed analysis awaits all 2000 local unit receipts")
        self.status("MIRRORING_FINAL_METADATA")
        analysis, analysis_hash = controls["G2_ANALYSIS.json"]
        complete = self.final_metadata(analysis, analysis_hash)
        self.status(FINAL, analysis_sha256=analysis_hash, local_mirror_complete_sha256=byte_sha(self.path(self.output + "/LOCAL_MIRROR_COMPLETE.json").read_bytes()),
                    scientific_gate=analysis["scientific_gate"], completed_unix=time.time(), final_full_verification_required=True)
        return True

    def run(self):
        lease = self.path(self.private + "/coordinator.lock")
        lease.parent.mkdir(parents=True, exist_ok=True)
        self.path(self.private + "/coordinator.lock")
        with self.helpers.PublisherLock(lease):
            for name, content in self.control_snapshots:
                self.publish(self.private + "/control_sources/" + name + "_" + byte_sha(content)[:20] + ".py", content, exclusive=True)
            heartbeat = threading.Thread(target=self.heartbeat, name="g2-owner-heartbeat", daemon=True)
            heartbeat.start()
            try:
                while not self.stop.is_set():
                    require(self.heartbeat_error is None, "Independent coordinator heartbeat failed")
                    if self.path("artifacts/monitoring/g2/STOP_COORDINATOR").exists():
                        self.status("STOPPED_BY_OWNER")
                        return 3
                    try:
                        if self.cycle():
                            return 0
                    except (IntegrityError, ValueError, FileNotFoundError, PermissionError) as exc:
                        self.status("ATTENTION_REQUIRED", error_type=type(exc).__name__, error=str(exc))
                        return 2
                    except Exception as exc:
                        if fatal_connection(exc):
                            self.status("FAILED", error_type=type(exc).__name__, reason="AUTHENTICATION_OR_HOST_KEY_FAILURE")
                            return 2
                        if retryable_connection(exc):
                            self.status("CONNECTION_RETRY", error_type=type(exc).__name__, progress_preserved=True)
                        else:
                            self.status("FAILED", error_type=type(exc).__name__, reason="LOCAL_OR_CONTROL_FAILURE")
                            return 2
                    self.stop.wait(self.args.interval_seconds)
                require(self.heartbeat_error is None, "Independent coordinator heartbeat failed")
                return 3
            except KeyboardInterrupt:
                self.status("STOPPED_BY_OWNER")
                return 3
            except Exception as exc:
                self.status("FAILED", error_type=type(exc).__name__, reason="COORDINATOR_HEARTBEAT_OR_CONTROL_FAILURE")
                return 2
            finally:
                self.stop.set()
                heartbeat.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--known-hosts", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.interval_seconds < 5 or not 1 <= args.port <= 65535:
        parser.error("A valid SSH port and an interval of at least five seconds are required")
    return Coordinator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
