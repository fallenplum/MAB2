"""Read-only completed-unit tar transport for the local G2 coordinator."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import stat
import tarfile
import threading
import time
import traceback


HELPER_PATH = "artifacts/private/monitoring/mirror_only.py"
HELPER_SHA256 = "273008dfafa95fe67f522fd116024c50f2176298072f6f4b95cf6267dc7e5ccb"
MAX_SMALL_BYTES = 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
CHANNEL_TIMEOUT = 30


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (canonical(value) + "\n").encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in a unit transport artifact")
        result[key] = value
    return result


def decode(content):
    value = json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Unit transport JSON must be an object")
    return value


def relative_parts(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Unsafe unit transport path")
    return name.split("/")


def unit_id(question_id, seed):
    return sha(canonical([question_id, seed]).encode("utf-8"))[:20]


def schedule_for(unit):
    result, positions = [], {}
    for index, request in enumerate(unit["mask_requests"]):
        mask, reference, aliases = request["mask"], request["reference_mask"], request["aliases"]
        for candidate in (mask,) if reference is None else (mask, reference):
            if (not isinstance(candidate, list) or len(candidate) != 12
                    or any(type(bit) is not int or bit not in (0, 1) for bit in candidate)):
                raise ValueError("Frozen unit has an invalid explicit graph mask")
        if reference is not None and tuple(reference) not in positions:
            raise ValueError("Frozen reference anchor is not earlier in its unit")
        if not isinstance(aliases, list) or any(not isinstance(alias, dict) for alias in aliases):
            raise ValueError("Frozen mask aliases must be explicit metadata objects")
        key = tuple(mask)
        if key not in positions:
            positions[key] = len(result)
            result.append({"mask": mask, "reference_mask": reference, "aliases": [], "request_indices": []})
        result[positions[key]]["aliases"].extend(aliases)
        result[positions[key]]["request_indices"].append(index)
    if not result or len(result) > 26:
        raise ValueError("A G2 unit must contain at most 26 unique masks")
    return result


def expected_unit_plan(unit, plan_hash, protocol_hash):
    return {"schema": "g2_collection_unit_v1", "question_id": unit["question_id"], "seed": unit["seed"],
            "plan_sha256": plan_hash, "protocol_hash": protocol_hash,
            "mask_schedule": schedule_for(unit), "labels_read": False}


def validate_done(content, unit, plan, plan_hash):
    done = decode(content)
    uid = unit_id(unit["question_id"], unit["seed"])
    worker = done.get("worker_id")
    if worker not in plan["qualification"]["worker_ids"]:
        raise ValueError("Completed unit belongs to an unqualified worker")
    expected = {"schema": "g2_collection_done_v1", "question_id": unit["question_id"], "seed": unit["seed"],
                "unit_id": uid, "worker_id": worker, "plan_sha256": plan_hash, "labels_read": False,
                "observations": len(schedule_for(unit)),
                "unit_frozen_path": f"segments/{worker}/units/{uid}/UNIT_OUTPUTS_FROZEN.json"}
    if any(done.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed unit identity or frozen path differs from the plan")
    if any(not re.fullmatch(r"[0-9a-f]{64}", done.get(key, "")) for key in ("claim_sha256", "unit_frozen_sha256")):
        raise ValueError("Completed unit has an invalid artifact digest")
    return done


def validate_marker(content, done, unit, plan_hash, protocol_hash):
    if sha(content) != done["unit_frozen_sha256"]:
        raise ValueError("UNIT_OUTPUTS_FROZEN bytes differ from its completion receipt")
    marker = decode(content)
    schedule = schedule_for(unit)
    expected = {"schema": "g2_collection_unit_outputs_v1", "question_id": unit["question_id"], "seed": unit["seed"],
                "plan_sha256": plan_hash, "protocol_hash": protocol_hash, "labels_read": False,
                "first_gold_opened_unix": None, "observations": len(schedule)}
    if any(marker.get(key) != value for key, value in expected.items()) or len(marker["rows"]) != len(schedule):
        raise ValueError("Frozen unit marker is incomplete or has a different identity")
    uid = unit_id(unit["question_id"], unit["seed"])
    members = {"UNIT_PLAN.json": sha(json_bytes(expected_unit_plan(unit, plan_hash, protocol_hash))),
               "UNIT_OUTPUTS_FROZEN.json": done["unit_frozen_sha256"]}
    for item, row in zip(schedule, marker["rows"]):
        mask_id = "".join(map(str, item["mask"]))
        if (set(row) != {"mask_sha256", "raw_path", "raw_sha256", "execution_path", "execution_sha256"}
                or row["mask_sha256"] != sha(canonical(item["mask"]).encode("utf-8"))):
            raise ValueError("Frozen unit row differs from its unique mask schedule")
        for folder, path_key, hash_key in (("raw", "raw_path", "raw_sha256"), ("executions", "execution_path", "execution_sha256")):
            name = f"{folder}/{mask_id}.json"
            if (row[path_key] != f"units/{uid}/{name}"
                    or not re.fullmatch(r"[0-9a-f]{64}", row[hash_key])):
                raise ValueError("Frozen unit row has a different relative path or digest")
            members[name] = row[hash_key]
    return marker, members


def validate_archive(compressed, expected_hashes):
    """Validate an entire bounded archive before publishing any member."""
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("Compressed unit exceeds the transport size cap")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as source:
        unpacked = source.read(MAX_ARCHIVE_BYTES + 1)
    if len(unpacked) > MAX_ARCHIVE_BYTES:
        raise ValueError("Uncompressed unit exceeds the transport size cap")
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(unpacked), mode="r:", ignore_zeros=True) as archive:
        for member in archive:
            relative_parts(member.name)
            if (member.name not in expected_hashes or member.name in contents
                    or not member.isfile() or member.sparse is not None
                    or member.size < 0 or member.size > MAX_MEMBER_BYTES):
                raise ValueError("Unexpected, duplicate, linked, or oversized unit archive member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Regular unit archive member has no readable payload")
            content = source.read(member.size + 1)
            if len(content) != member.size or sha(content) != expected_hashes[member.name]:
                raise ValueError("Unit archive member bytes differ from its frozen SHA")
            contents[member.name] = content
    if set(contents) != set(expected_hashes):
        raise ValueError("Unit archive omits a required committed member")
    return contents


def remote_file(sftp, name):
    info = sftp.lstat(name)
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > MAX_SMALL_BYTES:
        raise ValueError("Remote unit metadata is linked or exceeds its size cap")
    with sftp.open(name, "rb") as handle:
        opened = handle.stat()
        if opened.st_size != info.st_size:
            raise EOFError("Remote unit metadata changed before opening")
        content = handle.read(info.st_size + 1)
    if len(content) != info.st_size:
        raise EOFError("Remote unit metadata changed while reading")
    return content


def remote_tar(client, directory, members):
    command = "tar -C " + shlex.quote(directory) + " -czf - -- " + " ".join(shlex.quote(name) for name in sorted(members))
    stdin, stdout, stderr = client.exec_command(command, timeout=CHANNEL_TIMEOUT)
    stdin.close()
    channel = stdout.channel
    channel.settimeout(CHANNEL_TIMEOUT)
    started, compressed = time.monotonic(), bytearray()
    try:
        while True:
            if time.monotonic() - started > 120:
                raise TimeoutError("Unit archive transport exceeded its total duration bound")
            block = stdout.read(65536)
            if not block:
                break
            compressed.extend(block)
            if len(compressed) > MAX_COMPRESSED_BYTES:
                raise ValueError("Compressed unit exceeds the transport size cap")
        # GNU tar emits only bounded diagnostics for these <=54 fixed names.
        diagnostic = stderr.read(4097)
        deadline = time.monotonic() + CHANNEL_TIMEOUT
        while not channel.exit_status_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError("Remote read-only tar did not exit within its timeout")
            time.sleep(.05)
        if channel.recv_exit_status() != 0 or len(diagnostic) > 4096:
            raise RuntimeError("Remote read-only tar failed")
        return bytes(compressed)
    finally:
        for stream in (stdout, stderr):
            try:
                stream.close()
            except Exception:
                pass
        try:
            channel.close()
        except Exception:
            pass


def close_connection(client, sftp):
    for handle in (sftp, client):
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def fatal_authentication(exc):
    return (type(exc).__name__ in {"AuthenticationException", "BadHostKeyException", "PasswordRequiredException"}
            or type(exc).__name__ == "SSHException" and "not found in known_hosts" in str(exc))


def connect(client_factory):
    client = None
    try:
        client = client_factory()
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(CHANNEL_TIMEOUT)
        return client, sftp
    except BaseException:
        close_connection(client, None)
        raise


def load_helpers(root):
    path = root.joinpath(*HELPER_PATH.split("/"))
    if sha(path.read_bytes()) != HELPER_SHA256:
        raise ValueError("Local safe-publication helper differs from its pinned source")
    spec = importlib.util.spec_from_file_location("g2_pinned_mirror_publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mirror_units(root, remote_root, plan, client_factory, max_workers=4, *, progress_callback=None):
    """Mirror committed units using independent read-only SSH/SFTP connections.

    The caller supplies a strict-host-key client factory and owns experiment
    control. Cached unit receipts compare current source completion bytes.
    Final full-file verification remains a separate coordinator operation.
    """
    root = Path(root).resolve(strict=True)
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError("Completed-unit transport requires one to four independent connections")
    remote = PurePosixPath(remote_root)
    if not remote.is_absolute() or any(part == ".." for part in remote.parts) or "\\" in remote_root or "\x00" in remote_root:
        raise ValueError("Remote project root must be an absolute POSIX path")
    if (plan.get("schema") != "graph_credit_g2_plan_v1" or plan.get("status") != "PLAN_FROZEN"
            or plan.get("labels_read") is not False or not re.fullmatch(r"[A-Za-z0-9_-]+", plan["run_id"])
            or plan["output_dir"] != "results/g2/" + plan["run_id"]):
        raise ValueError("Only a frozen, label-free G2 plan may be mirrored")
    helpers = load_helpers(root)
    info = root.lstat()
    identity = (info.st_dev, info.st_ino)
    def local(relative):
        return helpers.local_path(root, relative, root_identity=identity)
    def publish(relative, content):
        helpers.atomic_bytes(local(relative), content, root=root, root_identity=identity)
    output_relative = plan["output_dir"]
    plan_content = local(output_relative + "/frozen_plan.json").read_bytes()
    if decode(plan_content) != plan:
        raise ValueError("Passed unit design differs from the local frozen plan bytes")
    plan_hash = sha(plan_content)
    freeze = decode(local(output_relative + "/PLAN_FROZEN.json").read_bytes())
    if freeze.get("state") != "PLAN_FROZEN" or freeze.get("plan_sha256") != plan_hash or freeze.get("labels_read") is not False:
        raise ValueError("Local frozen-plan marker does not bind the unit transport")
    units = {unit_id(unit["question_id"], unit["seed"]): unit for unit in plan["units"]}
    if len(units) != len(plan["units"]):
        raise ValueError("Frozen unit transport identifiers collide")
    private = "artifacts/private/g2/" + plan["run_id"]
    lease_path = local(private + "/unit_mirror.lock")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    local(private + "/unit_mirror.lock")
    with helpers.PublisherLock(lease_path):
        return _mirror_pool(root, remote, plan, plan_hash, units, client_factory, max_workers,
                            local, publish, private, progress_callback)


def _mirror_pool(root, remote, plan, plan_hash, units, client_factory, max_workers, local, publish, private, progress_callback):
    remote_output = str(remote / plan["output_dir"])
    first_error = None
    for attempt in range(3):
        client = sftp = None
        try:
            client, sftp = connect(client_factory)
            names = sftp.listdir(remote_output + "/claims")
            break
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if fatal_authentication(exc) or attempt == 2:
                raise RuntimeError("Completed-unit inventory connection failed") from first_error
            time.sleep(min(2 ** attempt, 4))
        finally:
            close_connection(client, sftp)
    done_names = [name for name in names if name.endswith(".done.json")]
    if any(not re.fullmatch(r"[0-9a-f]{20}\.done\.json", name) or name[:-10] not in units for name in done_names):
        raise ValueError("Remote completion inventory includes an unplanned unit")
    if len(set(done_names)) != len(done_names):
        raise ValueError("Remote completion inventory repeats a unit name")
    snapshot_units = tuple(name[:-10] for name in sorted(done_names))
    tasks = queue.Queue()
    for uid in snapshot_units:
        tasks.put(uid)
    summary = {"schema": "graph_credit_g2_unit_mirror_v1", "run_id": plan["run_id"], "plan_sha256": plan_hash,
               "remote_completed_units": len(done_names), "expected_units": len(units), "mirrored_new": 0,
               "skipped_cached": 0, "pending": len(units) - len(done_names), "failures": [],
               "callback_errors": [], "inventory_snapshot_sha256": sha(json_bytes(snapshot_units)),
               "tar_bytes_received": 0, "members_committed": 0, "labels_read": False,
               "final_full_verification_required": True}
    lock, abort, settled = threading.Lock(), threading.Event(), set()
    def worker():
        client = sftp = None
        try:
            while True:
                if abort.is_set():
                    return
                try:
                    uid = tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    first_failure = None
                    result = None
                    for attempt in range(3):
                        phase = "connect" if client is None else "unit"
                        try:
                            if client is None:
                                client, sftp = connect(client_factory)
                            phase = "unit"
                            result = _mirror_one(uid, units[uid], plan, plan_hash, remote_output,
                                                 client, sftp, local, publish, private)
                            break
                        except Exception as exc:
                            if first_failure is None:
                                first_failure = exc
                            close_connection(client, sftp)
                            client = sftp = None
                            if isinstance(exc, ValueError) or fatal_authentication(exc) or attempt == 2:
                                if fatal_authentication(exc) or phase == "connect":
                                    abort.set()
                                with lock:
                                    summary["failures"].append({"unit_id": uid, "error_type": type(first_failure).__name__,
                                                                "last_error_type": type(exc).__name__, "phase": phase, "attempts": attempt + 1})
                                break
                            time.sleep(min(2 ** attempt, 4))
                    if result is None:
                        continue
                    # A completed transfer is settled once. Progress publication
                    # belongs outside the transfer/reconnection retry boundary.
                    with lock:
                        if uid in settled:
                            raise RuntimeError("A mirrored unit was settled more than once")
                        settled.add(uid)
                        summary["skipped_cached" if result["cached"] else "mirrored_new"] += 1
                        summary["tar_bytes_received"] += result["tar_bytes"]
                        summary["members_committed"] += result["members_committed"]
                        if progress_callback is not None:
                            try:
                                progress_callback({"unit_id": uid, "mirrored_units": len(settled),
                                                   "remote_completed_units": len(snapshot_units),
                                                   "tar_bytes_received": summary["tar_bytes_received"]})
                            except Exception as exc:
                                abort.set()
                                summary["callback_errors"].append({
                                    "unit_id": uid, "phase": "progress_callback", "unit_settled": True,
                                    "error_type": type(exc).__name__, "errno": getattr(exc, "errno", None),
                                    "winerror": getattr(exc, "winerror", None),
                                    "traceback_frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                                         for frame in traceback.extract_tb(exc.__traceback__)]})
                finally:
                    tasks.task_done()
        finally:
            close_connection(client, sftp)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker) for _ in range(max_workers)]
        for future in futures:
            future.result()
    if summary["mirrored_new"] + summary["skipped_cached"] != len(settled):
        raise RuntimeError("Mirror outcome counters disagree with distinct settled units")
    summary["settled_units"] = len(settled)
    summary["unmirrored_completed_units"] = len(snapshot_units) - len(settled)
    summary["status"] = ("ATTENTION_REQUIRED" if summary["failures"] or summary["callback_errors"] or summary["unmirrored_completed_units"]
                         else "IN_PROGRESS" if summary["pending"] else "PASS")
    summary["completed_unix"] = time.time()
    return summary


def _mirror_one(uid, unit, plan, plan_hash, remote_output, client, sftp, local, publish, private):
    done_relative = plan["output_dir"] + "/claims/" + uid + ".done.json"
    done_content = remote_file(sftp, remote_output + "/claims/" + uid + ".done.json")
    done = validate_done(done_content, unit, plan, plan_hash)
    unit_relative = plan["output_dir"] + "/" + done["unit_frozen_path"].removesuffix("/UNIT_OUTPUTS_FROZEN.json")
    receipt_relative = private + "/mirrored_units/" + uid + ".json"
    try:
        receipt = decode(local(receipt_relative).read_bytes())
        local_marker = local(unit_relative + "/UNIT_OUTPUTS_FROZEN.json").read_bytes()
        _, expected = validate_marker(local_marker, done, unit, plan_hash, plan["protocol_hash"])
        if (receipt.get("schema") == "graph_credit_g2_mirrored_unit_v1" and receipt.get("plan_sha256") == plan_hash
                and receipt.get("done_sha256") == sha(done_content) and receipt.get("unit_id") == uid
                and receipt.get("unit_relative_path") == unit_relative and receipt.get("labels_read") is False
                and receipt.get("unit_frozen_sha256") == done["unit_frozen_sha256"]
                and {name: item["sha256"] for name, item in receipt["members"].items()} == expected):
            return {"cached": True, "tar_bytes": 0, "members_committed": 0}
    except FileNotFoundError:
        pass
    except (ValueError, KeyError, TypeError):
        # A damaged local receipt is a request to retransmit the whole unit.
        pass
    claim_relative = plan["output_dir"] + "/claims/" + uid + ".json"
    claim_content = remote_file(sftp, remote_output + "/claims/" + uid + ".json")
    claim = decode(claim_content)
    if (sha(claim_content) != done["claim_sha256"]
            or any(claim.get(name) != done[name] for name in ("question_id", "seed", "worker_id", "plan_sha256", "labels_read"))
            or claim.get("schema") != "g2_collection_claim_v1"):
        raise ValueError("Remote completion receipt does not bind its owner claim")
    marker_content = remote_file(sftp, remote_output + "/" + done["unit_frozen_path"])
    _, expected = validate_marker(marker_content, done, unit, plan_hash, plan["protocol_hash"])
    remote_directory = remote_output + "/" + done["unit_frozen_path"].removesuffix("/UNIT_OUTPUTS_FROZEN.json")
    compressed = remote_tar(client, remote_directory, expected)
    contents = validate_archive(compressed, expected)
    if contents["UNIT_OUTPUTS_FROZEN.json"] != marker_content:
        raise ValueError("Remote frozen unit marker changed between metadata and tar reads")
    for name, content in sorted(contents.items()):
        publish(unit_relative + "/" + name, content)
    publish(claim_relative, claim_content)
    publish(done_relative, done_content)
    receipt = {"schema": "graph_credit_g2_mirrored_unit_v1", "run_id": plan["run_id"], "unit_id": uid,
               "question_id": unit["question_id"], "seed": unit["seed"], "worker_id": done["worker_id"],
               "plan_sha256": plan_hash, "done_sha256": sha(done_content), "claim_sha256": sha(claim_content),
               "unit_frozen_sha256": done["unit_frozen_sha256"], "unit_relative_path": unit_relative,
               "members": {name: {"sha256": sha(content), "bytes": len(content)} for name, content in contents.items()},
               "tar_bytes_received": len(compressed), "mirrored_unix": time.time(), "labels_read": False,
               "final_full_verification_required": True}
    publish(receipt_relative, json_bytes(receipt))
    return {"cached": False, "tar_bytes": len(compressed), "members_committed": len(contents)}
