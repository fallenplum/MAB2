"""Independent CPU G2 acceptance after complete locally mirrored termination.

No frozen pipeline module is imported. The original scorer is imported only
after all metadata/byte gates and the durable audit label-access event. Ridge
uses independent Cholesky solves, while the production implementation uses SVD.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

import numpy as np
from scipy.linalg import cho_factor, cho_solve

SEEDS = (2701, 2702, 2703, 2704, 2705)
ALPHAS = tuple(10.0 ** value for value in range(-4, 4))
COUNTS = {"train": 200, "tune": 80, "test": 120}
EXPECTED_OUTPUTS = 51970
COEFFICIENT_ATOL, COEFFICIENT_RTOL = 1e-8, 1e-6
STAT_ATOL, STAT_RTOL = 1e-12, 1e-10
ORIGINAL_SCORER_SHA = "da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e"
ORIGINAL_SOURCE_SHA = "03856d909ce94b270bc7d31519acdb74a58913cfbec1886d1aa2bca4cb830b95"
PROTOCOL_SHA = "5c52efae884ee59481f22396c46afd4830e1bc9b5abb0f803b39a373b7600426"
CALIBRATION_SHA = "4517478a518d09c2c1afd5cf7b3e453e5947c62b128e9c98138512323328b6e1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
PIPELINE_NAMES = {"prepare.py", "collector.py", "predictor.py", "workflow.py", "supervisor.py"}
PIPELINE_PINS = {
    "prepare.py": "f60a75c8508e9b34324d67a6ded0f5c9bcf3b3da3b8539b4c7b98b73a4fd6de8",
    "collector.py": "fae5f62a08757ba09f5256564f4b6043e1fcf4b906a0e73de95a3f56dbdd1fba",
    "predictor.py": "d6d93dc938ce7cff682a5457c59b779bed5668999619e5afb614773f756e8e03",
    "workflow.py": "f46a92f78e81c9593213fff3410933048d531debeb28835f89a0f667f86d706f",
    "supervisor.py": "a4a3b3bbabcdcce4be41797b816a4fc331a24ff14fcae3e2d0fc2404352319cb",
}
MODEL_NAMES = ("B0", "C1", "C2")


class AuditError(RuntimeError):
    def __init__(self, message, relative_path=None):
        self.relative_path = relative_path
        super().__init__(message)


class WaitingError(FileNotFoundError):
    pass


def require(condition, message, relative_path=None):
    if not condition:
        raise AuditError(message, relative_path)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON member " + key)
        result[key] = value
    return result


def decode(content):
    return json.loads(content, object_pairs_hook=unique_object)


def safe_path(root, relative, *, exists=True):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Unsafe project-relative evidence path", str(relative))
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Unsafe evidence path components", relative)
    path = root
    for part in parts:
        path = path / part
        try:
            info = path.lstat()
        except FileNotFoundError:
            if exists:
                raise WaitingError("Mirror artifact not present " + relative)
            break
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "Evidence link/reparse point is forbidden", relative)
    result = root.joinpath(*parts)
    require(result.is_relative_to(root), "Evidence escapes root", relative)
    if exists:
        require(result.is_file(), "Evidence is not a regular file", relative)
    return result


def no_gold(relative):
    name = Path(relative).name.lower()
    require(not (name.endswith(".jsonl") and any(word in name for word in ("label", "gold"))),
            "Label files require the explicit post-terminal reader", relative)


def file_sha(root, relative):
    no_gold(relative)
    value = hashlib.sha256()
    with safe_path(root, relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_object(root, relative):
    no_gold(relative)
    value = decode(safe_path(root, relative).read_bytes())
    require(isinstance(value, dict), "Expected evidence object", relative)
    return value


def rows(root, relative):
    no_gold(relative)
    result = []
    with safe_path(root, relative).open("rb") as handle:
        for line in handle:
            require(line.endswith(b"\n") and line.strip(), "Incomplete/blank JSONL record", relative)
            value = decode(line)
            require(isinstance(value, dict), "Non-object JSONL record", relative)
            result.append(value)
    return result


def check_hash(root, relative, expected):
    require(isinstance(expected, str) and SHA.fullmatch(expected), "Invalid expected SHA", relative)
    actual = file_sha(root, relative)
    require(actual == expected, "Evidence bytes SHA mismatch", relative)
    return actual


def atomic_json(root, relative, value, *, exclusive=False):
    path = safe_path(root, relative, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:16] + ".audittmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((canonical(value) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        else:
            os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_sha(root, relative)


def near(actual, expected, *, coefficient=False):
    tolerance = (COEFFICIENT_ATOL, COEFFICIENT_RTOL) if coefficient else (STAT_ATOL, STAT_RTOL)
    return bool(np.allclose(actual, expected, atol=tolerance[0], rtol=tolerance[1], equal_nan=False))


def decimal_number(value):
    text = str(value).strip()
    if text.startswith("$"):
        text = text[1:].strip()
    if re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text) is None:
        return None
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def independent_answer(output):
    text, candidates = str(output).strip(), []
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "answer" in value:
            answer = value["answer"]
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                return None
            candidates.append(str(answer))
    for pattern in (r'"answer"\s*:\s*"([^"\n]*)"',
                    r'"answer"\s*:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?)\s*(?=[,}\n])',
                    r"(?im)^\s*(?:answer|final answer)\s*:\s*([^\n]+)$", r"####\s*([^\n]+)"):
        candidates.extend(re.findall(pattern, text))
    if not candidates:
        candidates.append(text)
    values = [decimal_number(value) for value in candidates]
    return values[0] if values and all(value is not None and value == values[0] for value in values) else None


def independent_score(output, label):
    require(label.get("task", "gsm8k").lower() == "gsm8k", "Non-GSM label task")
    target = decimal_number(str(label.get("answer", label.get("target"))).split("####")[-1].strip())
    require(target is not None, "Invalid GSM gold number")
    observed = independent_answer(output)
    return float(observed is not None and observed == target)


def graph_basis(mask, order):
    require(len(mask) == 12 and all(type(bit) is int and bit in (0, 1) for bit in mask), "Invalid graph mask")
    result = [1.0, *mask]
    if order == 2:
        result.extend(mask[left] * mask[right] for left in range(12) for right in range(left + 1, 12))
    else:
        require(order == 1, "Unsupported graph basis")
    return np.asarray(result, dtype=np.float64)


def psi(phi, mask, order):
    phi = np.asarray(phi, dtype=np.float64)
    require(phi.shape == (9,) and np.isfinite(phi).all() and phi[-1] == 1, "Invalid frozen phi")
    return np.multiply.outer(phi, graph_basis(mask, order)).reshape(-1)


def solve_ridge_path(matrix, target, weights, alphas=ALPHAS):
    matrix, target, weights = map(lambda value: np.asarray(value, dtype=np.float64), (matrix, target, weights))
    require(matrix.ndim == 2 and target.shape == weights.shape == (len(matrix),)
            and len(matrix) and np.isfinite(matrix).all() and np.isfinite(target).all()
            and np.isfinite(weights).all() and np.all(weights > 0), "Invalid independent ridge inputs")
    gram = matrix.T @ (weights[:, None] * matrix)
    right = matrix.T @ (weights * target)
    identity = np.eye(matrix.shape[1], dtype=np.float64)
    coefficients = {}
    for alpha in alphas:
        require(math.isfinite(alpha) and alpha > 0, "Invalid ridge alpha")
        factor = cho_factor(gram + alpha * identity, lower=True, check_finite=False)
        coefficients[alpha] = cho_solve(factor, right, check_finite=False)
    return coefficients


def select_alpha(scores):
    smallest = min(scores.values())
    return max(alpha for alpha in scores if scores[alpha] <= smallest + 1e-12)


def cross_seed_risk(prediction, credits):
    require(len(credits) == 5, "Five independent paired-seed credits required")
    errors = [prediction - value for value in credits]
    products = [errors[left] * errors[right] for left in range(5) for right in range(left + 1, 5)]
    return math.fsum(products) / 10


def bootstrap_question_difference(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "Invalid question-loss differences")
    generator = np.random.Generator(np.random.PCG64(39173))
    sampled_means = []
    for _ in range(10000):
        positions = generator.integers(0, len(values), size=len(values))
        sampled_means.append(math.fsum(float(values[index]) for index in positions) / len(values))
    return {"estimate": math.fsum(values.tolist()) / len(values),
            "interval": np.quantile(sampled_means, [.025, .975], method="linear").tolist(),
            "question_count": len(values), "bootstrap_samples": 10000, "bootstrap_seed": 39173,
            "confidence": .95, "resampling_unit": "whole_question"}


def bounded_hash_verification(root, tasks, *, validators=None, sizes=None, execution_summary=None, progress_callback=None):
    """Snapshot first, then keep at most 128 jobs and six file readers active."""
    ordered = sorted(tasks.items())
    verified = {}
    validators = validators or {}
    sizes = sizes or {}
    def work(item):
        relative, expected = item
        if relative in sizes:
            require(safe_path(root, relative).stat().st_size == sizes[relative], "Mirror member size mismatch", relative)
        if relative not in validators:
            return relative, check_hash(root, relative, expected), None
        no_gold(relative)
        path = safe_path(root, relative)
        require(path.stat().st_size <= 16 * 1024 * 1024, "Execution/raw JSON exceeds bounded reader size", relative)
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        require(actual == expected, "Execution/raw bytes SHA mismatch", relative)
        return relative, actual, validators[relative](decode(content))
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="g2-independent-sha") as pool:
        for start in range(0, len(ordered), 128):
            for relative, actual, extra in pool.map(work, ordered[start:start + 128]):
                verified[relative] = actual
                if extra is not None and execution_summary is not None:
                    fresh, referenced = set(extra["fresh_ids"]), set(extra["referenced_ids"])
                    require(not fresh & execution_summary["fresh_ids"], "A paid call is freshly counted in multiple executions", relative)
                    execution_summary["fresh_ids"].update(fresh)
                    execution_summary["referenced_ids"].update(referenced)
            if progress_callback:
                progress_callback({"phase": "VERIFYING_MIRROR_BYTES", "completed": len(verified), "total": len(ordered)})
    return verified


def unit_id(qid, seed):
    return digest([qid, seed])[:20]


def _add_task(tasks, relative, expected):
    require(isinstance(expected, str) and SHA.fullmatch(expected), "Invalid SHA in mirror/source snapshot", relative)
    require(relative not in tasks or tasks[relative] == expected, "Conflicting expected hashes for one evidence file", relative)
    tasks[relative] = expected


def _descriptor(root, descriptor, tasks):
    relative = descriptor["path"]
    _add_task(tasks, relative, descriptor["sha256"])
    require(safe_path(root, relative).stat().st_size == descriptor["bytes"], "Evidence descriptor size mismatch", relative)


def _expected_design(plan):
    require(Counter(plan["partitions"].values()) == Counter(COUNTS), "Expected train200/tune80/test120")
    require(plan.get("expected_units") == len(plan["units"]) == 2000, "Expected exactly 2000 question/seed units")
    units, expected = {}, {}
    for unit in plan["units"]:
        qid, seed = unit["question_id"], unit["seed"]
        require(type(seed) is int and seed in SEEDS and unit["partition"] == plan["partitions"][qid], "Unit partition/seed mismatch")
        uid = unit_id(qid, seed)
        require(uid not in units, "Repeated/colliding collection unit")
        units[uid] = unit
        for request in unit["mask_requests"]:
            mask = request["mask"]
            graph_basis(mask, 1)
            key = (qid, seed, tuple(mask))
            expected[key] = unit["partition"]
    require({(unit["question_id"], unit["seed"]) for unit in units.values()}
            == {(qid, seed) for qid in plan["partitions"] for seed in SEEDS}, "Missing frozen question/seed unit")
    require(len(expected) == plan.get("expected_outputs") == EXPECTED_OUTPUTS, "Expected the frozen 51970 unique graph outputs")
    return units, expected


def verify_design_generation(plan, questions):
    """Rebuild the predeclared split/anchor/flip hashes without pipeline code."""
    ordered = sorted((question["id"] for question in questions),
                     key=lambda qid: digest(["graph_credit/G2/split/v1", qid]))
    partitions = {qid: "train" if index < 200 else "tune" if index < 280 else "test"
                  for index, qid in enumerate(ordered)}
    require(plan["partitions"] == partitions, "Question partitions differ from the frozen split hash rule")
    all_masks = {}
    expected_units = []
    for qid in ordered:
        requests, anchors, contrasts, previous = [], [], [], None
        for family, threshold in (("bernoulli_025", 64), ("bernoulli_075", 192)):
            anchor = [int(int(digest(["graph_credit/G2/anchor/v1", qid, family, edge])[:2], 16) < threshold)
                      for edge in range(12)]
            mask_id = "".join(map(str, anchor))
            anchors.append({"anchor_id": family, "mask_id": mask_id, "mask": anchor})
            requests.append({"mask": anchor, "reference_mask": previous,
                             "aliases": [{"anchor_id": family, "kind": "anchor"}]})
            for edge in range(12):
                flipped = list(anchor)
                flipped[edge] = 1 - flipped[edge]
                requests.append({"mask": flipped, "reference_mask": anchor,
                                 "aliases": [{"anchor_id": family, "kind": "single_flip", "edge": edge}]})
                off, on = (anchor, flipped) if anchor[edge] == 0 else (flipped, anchor)
                contrasts.append({"question_id": qid, "anchor_id": family, "edge": edge,
                                  "mask0_id": "".join(map(str, off)), "mask1_id": "".join(map(str, on))})
            previous = anchor
        design = plan["designs"][qid]
        require(design.get("anchors") == anchors and design.get("mask_requests") == requests
                and design.get("contrasts") == contrasts, "Question graph design differs from the frozen anchor/flip hash rule")
        for request in requests:
            all_masks["".join(map(str, request["mask"]))] = request["mask"]
        expected_units.extend({"question_id": qid, "seed": seed, "partition": partitions[qid], "mask_requests": requests} for seed in SEEDS)
    require(plan["masks"] == all_masks and plan["units"] == expected_units, "Global mask map/unit schedule differs from prespecified design")


def _contrasts(plan, partition):
    result = []
    for qid in sorted(plan["partitions"]):
        if plan["partitions"][qid] != partition:
            continue
        entries = plan["designs"][qid]["contrasts"]
        require(len(entries) == 24, "Question lacks 24 anchor/edge targets")
        seen = set()
        for entry in entries:
            require(entry["question_id"] == qid and type(entry["edge"]) is int and entry["edge"] in range(12), "Invalid contrast identity")
            key = (entry["anchor_id"], entry["edge"])
            require(key not in seen, "Duplicate contrast coordinate")
            seen.add(key)
            off, on = plan["masks"][entry["mask0_id"]], plan["masks"][entry["mask1_id"]]
            graph_basis(off, 1)
            graph_basis(on, 1)
            edge = entry["edge"]
            require(off[edge] == 0 and on[edge] == 1 and all(off[i] == on[i] for i in range(12) if i != edge), "Incorrect signed edge contrast")
            result.append({"question_id": qid, "anchor_id": entry["anchor_id"], "edge": edge,
                           "mask0": tuple(off), "mask1": tuple(on)})
        require({anchor for anchor, edge in seen} == {"bernoulli_025", "bernoulli_075"}
                and len(seen) == 24, "Unexpected anchor/edge support")
    return sorted(result, key=lambda entry: (entry["question_id"], entry["anchor_id"], entry["edge"]))


def _verify_access_chain(root, plan, base, plan_hash, global_hash, prediction_hash, prediction, analysis, tasks):
    train_relative, heldout_relative = base + "/TRAIN_TUNE_LABEL_RELEASE.json", base + "/HELDOUT_LABEL_RELEASE.json"
    train, heldout = read_object(root, train_relative), read_object(root, heldout_relative)
    require(train.get("state") == "TRAIN_TUNE_LABELS_RELEASED" and train.get("plan_sha256") == plan_hash
            and train.get("global_outputs_frozen_sha256") == global_hash, "Train/tune release is not frozen-generation-bound")
    require(heldout.get("state") == "HELDOUT_LABELS_RELEASED" and heldout.get("plan_sha256") == plan_hash
            and heldout.get("predictions_frozen_sha256") == prediction_hash, "Heldout release is not prediction-bound")
    label_hashes = {"train": train.get("train_labels_sha256"), "tune": train.get("tune_labels_sha256"),
                    "test": heldout.get("heldout_labels_sha256")}
    require(all(isinstance(value, str) and SHA.fullmatch(value) for value in label_hashes.values()), "Release label SHA is missing")
    train_hash, heldout_hash = file_sha(root, train_relative), file_sha(root, heldout_relative)
    require(train_hash == prediction.get("train_tune_label_release_sha256")
            and heldout_hash == analysis.get("heldout_label_release_sha256")
            and analysis.get("heldout_labels_sha256") == label_hashes["test"]
            and prediction.get("training_labels_sha256") == {key: label_hashes[key] for key in ("train", "tune")},
            "Production label releases differ from scored/frozen lineage")
    _add_task(tasks, train_relative, train_hash)
    _add_task(tasks, heldout_relative, heldout_hash)
    events = {}
    for name, partitions, binding, release_hash, expected_event_hash in (
        ("FIRST_TRAIN_TUNE_LABELS_OPENED.json", ("train", "tune"), {"global_outputs_frozen_sha256": global_hash},
         train_hash, prediction["first_train_tune_labels_opened_sha256"]),
        ("FIRST_HELDOUT_LABELS_OPENED.json", ("test",), {"global_outputs_frozen_sha256": global_hash,
         "predictions_frozen_sha256": prediction_hash}, heldout_hash, analysis["first_heldout_labels_opened_sha256"]),
    ):
        relative = base + "/" + name
        event = read_object(root, relative)
        _add_task(tasks, relative, expected_event_hash)
        require(event.get("schema") == "graph_credit_g2_first_label_access_v1"
                and event.get("event") == "HANDLES_OPENED_BEFORE_FIRST_READ"
                and event.get("plan_sha256") == plan_hash and event.get("run_id") == plan["run_id"]
                and event.get("source_manifest_sha256") == plan["source_manifest_sha256"]
                and event.get("pipeline_source_manifest_sha256") == digest(plan["pipeline_source_manifest"])
                and event.get("label_release_sha256") == release_hash
                and all(event.get(key) == value for key, value in binding.items())
                and event.get("expected_labels_sha256") == {part: label_hashes[part] for part in partitions}
                and set(event.get("handles", {})) == set(partitions)
                and all(event["handles"][part]["path"] == plan["label_release"][part] for part in partitions),
                "Durable first-label event does not match its earlier freeze/release", relative)
        require(type(event.get("first_gold_opened_unix")) in (int, float)
                and type(event.get("created_unix")) in (int, float)
                and math.isfinite(event["first_gold_opened_unix"]) and math.isfinite(event["created_unix"])
                and event["first_gold_opened_unix"] <= event["created_unix"], "First-label event open/write order is invalid", relative)
        events[name] = event
    require(events["FIRST_TRAIN_TUNE_LABELS_OPENED.json"]["created_unix"] <= prediction["created_unix"]
            <= events["FIRST_HELDOUT_LABELS_OPENED.json"]["first_gold_opened_unix"] <= analysis["completed_unix"],
            "Same-server production event chronology is inconsistent")
    owner_names = ("LOCAL_GLOBAL_BARRIER_VERIFIED.json", "LOCAL_DATA_OWNER_FIRST_LABEL_READ.json",
                   "LOCAL_LABEL_SPLIT_FROZEN.json", "LOCAL_PREDICTION_REPLAY.json")
    owner, owner_hashes = {}, {}
    for name in owner_names:
        relative = base + "/" + name
        value, value_hash, _ = _snapshot_object(root, relative)
        owner[name], owner_hashes[name] = value, value_hash
        _add_task(tasks, relative, value_hash)
    barrier, first, split, replay = (owner[name] for name in owner_names)
    require(barrier.get("schema") == "graph_credit_g2_owner_global_barrier_v1" and barrier.get("status") == "PASS"
            and barrier.get("plan_sha256") == plan_hash and barrier.get("global_outputs_frozen_sha256") == global_hash
            and barrier.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and barrier.get("pipeline_source_manifest_sha256") == digest(plan["pipeline_source_manifest"])
            and barrier.get("observations") == EXPECTED_OUTPUTS and barrier.get("units") == 2000 and barrier.get("labels_read") is False,
            "Local data owner did not verify the global barrier before release")
    require(first.get("schema") == "graph_credit_g2_data_owner_first_label_read_v1"
            and first.get("event") == "HANDLE_OPENED_BEFORE_FIRST_READ" and first.get("plan_sha256") == plan_hash
            and first.get("global_outputs_frozen_sha256") == global_hash
            and split.get("schema") == "graph_credit_g2_owner_label_split_v1" and split.get("plan_sha256") == plan_hash
            and split.get("global_outputs_frozen_sha256") == global_hash
            and split.get("first_owner_label_read_sha256") == owner_hashes[owner_names[1]]
            and split.get("partitions_sha256") == digest(plan["partitions"]), "Data-owner first-read/split chain differs")
    for partition in COUNTS:
        require(split["labels"][partition]["sha256"] == label_hashes[partition]
                and split["labels"][partition]["path"] == plan["label_release"][partition]
                and split["labels"][partition]["questions"] == COUNTS[partition], "Owner split differs from released partition bytes")
    require(all(release.get("owner_first_label_read_sha256") == owner_hashes[owner_names[1]]
                and release.get("local_label_split_sha256") == owner_hashes[owner_names[2]] for release in (train, heldout)),
            "Label release did not bind the original owner split")
    require(replay.get("schema") == "graph_credit_g2_owner_prediction_replay_v1" and replay.get("status") == "PASS"
            and replay.get("plan_sha256") == plan_hash and replay.get("predictions_frozen_sha256") == prediction_hash
            and replay.get("parameters_sha256") == prediction["parameters_sha256"]
            and replay.get("predictions_sha256") == prediction["predictions_sha256"]
            and replay.get("coordinates") == 2880 and replay.get("questions") == 120,
            "Owner did not independently replay the frozen heldout predictions before release")
    require(barrier["verified_unix"] <= first["first_gold_opened_unix"] <= split["created_unix"]
            <= train["released_unix"] <= replay["verified_unix"] <= heldout["released_unix"],
            "Same-owner barrier/read/split/prediction/release chronology differs")
    events.update(owner)
    return label_hashes, events


def verify_server_event_order(global_marker, prediction, analysis, events):
    train, heldout = events["FIRST_TRAIN_TUNE_LABELS_OPENED.json"], events["FIRST_HELDOUT_LABELS_OPENED.json"]
    timeline = (global_marker.get("frozen_unix"), train.get("first_gold_opened_unix"), train.get("created_unix"),
                prediction.get("created_unix"), heldout.get("first_gold_opened_unix"), heldout.get("created_unix"),
                analysis.get("completed_unix"))
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in timeline),
            "A same-server freeze/first-open/completion timestamp is missing or invalid")
    require(all(left <= right for left, right in zip(timeline, timeline[1:])),
            "Same-server global-freeze/train-open/prediction/heldout-open/completion order differs")


def _snapshot_object(root, relative):
    no_gold(relative)
    content = safe_path(root, relative).read_bytes()
    value = decode(content)
    require(isinstance(value, dict), "Expected snapshot object", relative)
    return value, hashlib.sha256(content).hexdigest(), len(content)


def terminal_metadata(root, plan_relative):
    root = Path(root).resolve(strict=True)
    plan, plan_hash, _ = _snapshot_object(root, plan_relative)
    require(plan.get("schema") == "graph_credit_g2_plan_v1" and plan.get("status") == "PLAN_FROZEN"
            and plan.get("labels_read") is False, "Unfrozen G2 plan")
    base = plan["output_dir"]
    require(base == "results/g2/" + plan["run_id"] and plan_relative == base + "/frozen_plan.json", "Noncanonical G2 plan path")
    analysis_relative = base + "/G2_ANALYSIS.json"
    analysis, analysis_hash, _ = _snapshot_object(root, analysis_relative)
    if analysis.get("scientific_gate") not in {"G2_PASS", "INSUFFICIENT_PREDICTIVE_GAIN"}:
        raise WaitingError("G2 analysis is not at a complete scientific gate")
    require(analysis.get("schema") == "graph_credit_g2_heldout_evaluation_v1" and analysis.get("run_id") == plan["run_id"]
            and analysis.get("scientific_completion") is False
            and analysis.get("analysis_semantic_sha256") == digest({key: value for key, value in analysis.items() if key != "analysis_semantic_sha256"})
            and analysis.get("plan_sha256") == plan_hash and analysis.get("primary", {}).get("question_count") == 120
            and analysis.get("coverage", {}).get("complete") is True, "Incomplete/changed terminal G2 analysis")
    tasks, sizes = {plan_relative: plan_hash, analysis_relative: analysis_hash}, {}
    plan_marker_relative = base + "/PLAN_FROZEN.json"
    plan_marker = read_object(root, plan_marker_relative)
    _add_task(tasks, plan_marker_relative, file_sha(root, plan_marker_relative))
    require(plan_marker.get("state") == "PLAN_FROZEN" and plan_marker.get("plan_sha256") == plan_hash
            and plan_marker.get("labels_read") is False, "Initial plan freeze does not bind the actual plan")
    source, pipeline = plan["source_manifest"], plan["pipeline_source_manifest"]
    require(len(source) == 18 and digest(source) == plan["source_manifest_sha256"] == ORIGINAL_SOURCE_SHA,
            "Original 18-file source manifest differs")
    require(plan["protocol_hash"] == PROTOCOL_SHA and plan["graph"].get("rounds") == 2
            and plan["graph"].get("n_workers") == 4 and plan["graph"].get("message_tokens") == 128,
            "G2 original two-round/four-role protocol differs")
    require(len(pipeline) == 5 and {Path(path).name for path in pipeline} == PIPELINE_NAMES, "Expected the five frozen G2 core sources")
    for manifest in (source, pipeline):
        for relative, identity in manifest.items():
            _add_task(tasks, relative, identity["sha256"])
            sizes[relative] = identity["bytes"]
    for relative, identity in pipeline.items():
        require(identity["sha256"] == PIPELINE_PINS[Path(relative).name], "G2 core differs from independent source pin", relative)
    require(source["src/graph_credit/scoring.py"]["sha256"] == ORIGINAL_SCORER_SHA, "Original numeric scorer pin differs")
    question_relative = plan["questions"]["path"]
    require(plan["questions"]["sha256"] == CALIBRATION_SHA, "G2 unlabelled calibration source pin differs")
    _add_task(tasks, question_relative, CALIBRATION_SHA)
    questions = rows(root, question_relative)
    require(len(questions) == 400 and {row["id"] for row in questions} == set(plan["partitions"]), "Calibration question coverage differs")
    for question in questions:
        require(not {"answer", "label", "gold", "target", "solution", "quality"} & set(question), "A question contains quality feedback")
        payload = {"id": question["id"], "question": question["question"], "context": question.get("context", "")}
        require(digest(payload) == plan["questions"]["payload_sha256"][question["id"]], "Question payload hash differs")
    verify_design_generation(plan, questions)
    pipeline_hash = digest(pipeline)
    prediction_relative = base + "/PREDICTIONS_FROZEN.json"
    prediction, prediction_hash, _ = _snapshot_object(root, prediction_relative)
    global_relative = base + "/GLOBAL_OUTPUTS_FROZEN.json"
    global_marker, global_hash, _ = _snapshot_object(root, global_relative)
    local_relative = base + "/LOCAL_MIRROR_COMPLETE.json"
    local, local_hash, _ = _snapshot_object(root, local_relative)
    require(local.get("schema") == "graph_credit_g2_local_mirror_complete_v1" and local.get("state") == "LOCAL_MIRROR_COMPLETE"
            and local.get("run_id") == plan["run_id"] and local.get("plan_sha256") == plan_hash
            and local.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and local.get("pipeline_source_manifest_sha256") == pipeline_hash and local.get("mirrored_units") == 2000
            and local.get("global_outputs_frozen_sha256") == global_hash
            and local.get("predictions_frozen_sha256") == prediction_hash and local.get("analysis_sha256") == analysis_hash,
            "Local mirror completion is not bound to the terminal source/results")
    private_relative = base + "/PRIVATE_METADATA_MIRROR_RECEIPT.json"
    private = read_object(root, private_relative)
    _add_task(tasks, private_relative, local["private_metadata_receipt_sha256"])
    require(private.get("schema") == "graph_credit_g2_private_metadata_mirror_v1" and private.get("run_id") == plan["run_id"]
            and private.get("plan_sha256") == plan_hash and private.get("labels_read") is False,
            "Private metadata mirror receipt identity differs")
    for descriptor in private["descriptors"]:
        _descriptor(root, descriptor, tasks)
    for relative, expected in ((prediction_relative, prediction_hash), (global_relative, global_hash), (local_relative, local_hash)):
        _add_task(tasks, relative, expected)
    require(global_marker.get("schema") == "graph_credit_g2_outputs_frozen_v1"
            and global_marker.get("run_id") == plan["run_id"]
            and global_marker.get("state") == "GLOBAL_OUTPUTS_FROZEN" and global_marker.get("labels_read") is False
            and global_marker.get("first_gold_opened_unix") is None and global_marker.get("plan_sha256") == plan_hash
            and global_marker.get("completed_units") == global_marker.get("expected_units") == 2000
            and global_marker.get("observations") == global_marker.get("expected_outputs") == EXPECTED_OUTPUTS
            and global_marker.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and global_marker.get("pipeline_source_manifest_sha256") == pipeline_hash
            and global_marker.get("protocol_hash") == plan["protocol_hash"], "Global freeze is not the full label-free G2 set")
    require(prediction.get("schema") == "graph_credit_g2_predictions_frozen_v1" and prediction.get("state") == "PREDICTIONS_FROZEN"
            and prediction.get("run_id") == plan["run_id"] and prediction.get("scientific_completion") is False
            and prediction.get("plan_sha256") == plan_hash and prediction.get("global_outputs_frozen_sha256") == global_hash
            and prediction.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and prediction.get("pipeline_source_manifest_sha256") == pipeline_hash
            and prediction.get("heldout_prediction_rows") == 2880 and prediction.get("heldout_labels_read") is False,
            "Heldout prediction freeze identity differs")
    require(analysis.get("predictions_frozen_sha256") == prediction_hash
            and analysis.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and analysis.get("pipeline_source_manifest_sha256") == pipeline_hash, "Analysis has another predictor/source")
    feature = plan["features"]
    _add_task(tasks, feature["path"], feature["sha256"])
    _add_task(tasks, feature["receipt"], feature["receipt_sha256"])
    require(global_marker.get("features_sha256") == prediction.get("feature_sha256") == feature["sha256"]
            and global_marker.get("features_receipt_sha256") == prediction.get("features_receipt_sha256") == feature["receipt_sha256"],
            "Feature/PCA bytes are not bound before prediction and labels")
    feature_receipt = read_object(root, feature["receipt"])
    require(feature_receipt.get("status") == "PASS_LABEL_FREE" and feature_receipt.get("labels_read") is False
            and feature_receipt.get("pca_fit_questions") == 200 and feature_receipt.get("transformed_questions") == 400
            and feature_receipt.get("unlabeled_design_rank") == 711 and feature_receipt.get("dimensions") == 9
            and feature_receipt.get("feature_sha256") == feature["sha256"], "Train-fitted feature/PCA receipt differs")
    with np.load(safe_path(root, feature["path"]), allow_pickle=False) as data:
        qids, phi_array = data["question_ids"].tolist(), np.asarray(data["phi"], dtype=np.float64)
    require(len(qids) == len(set(qids)) == 400 and set(qids) == set(plan["partitions"])
            and phi_array.shape == (400, 9) and np.isfinite(phi_array).all() and np.all(phi_array[:, -1] == 1), "Feature question coverage differs")
    require(feature_receipt["train_ids_sha256"] == digest([qid for qid in qids if plan["partitions"][qid] == "train"]), "PCA training IDs differ")
    parameter_relative, predictions_relative = base + "/parameters.json", base + "/heldout_predictions.json"
    _add_task(tasks, parameter_relative, prediction["parameters_sha256"])
    _add_task(tasks, predictions_relative, prediction["predictions_sha256"])
    parameters = read_object(root, parameter_relative)
    predictions = decode(safe_path(root, predictions_relative).read_bytes())
    require(parameters.get("schema") == "graph_credit_g2_predictor_v1"
            and parameters.get("parameter_sha256") == prediction["parameters_semantic_sha256"]
            == digest({key: value for key, value in parameters.items() if key != "parameter_sha256"})
            and isinstance(predictions, list) and len(predictions) == 2880
            and digest(predictions) == prediction["predictions_semantic_sha256"] == analysis["predictions_sha256"], "Prediction/parameter semantic hash differs")
    for name in ("observations", "physical_cost_audit"):
        _descriptor(root, global_marker["artifacts"][name], tasks)
        require(global_marker["artifact_hashes"][name] == global_marker["artifacts"][name]["sha256"], "Global artifact alias differs")
    require(global_marker["artifact_hashes"]["features"] == feature["sha256"]
            and global_marker["artifact_hashes"]["features_receipt"] == feature["receipt_sha256"], "Global feature artifact aliases differ")
    _add_task(tasks, base + "/G2_COST_SUMMARY.json", analysis["cost_summary_sha256"])
    label_hashes, access_events = _verify_access_chain(root, plan, base, plan_hash, global_hash, prediction_hash, prediction, analysis, tasks)
    verify_server_event_order(global_marker, prediction, analysis, access_events)
    units, expected = _expected_design(plan)
    observation_rows = rows(root, global_marker["artifacts"]["observations"]["path"])
    indexed, by_unit = {}, defaultdict(list)
    for row in observation_rows:
        mask = row["graph_mask"]
        graph_basis(mask, 1)
        key = (row["question_id"], row["seed"], tuple(mask))
        require(key in expected and key not in indexed and row.get("partition") == expected[key]
                and row.get("quality") is None and row.get("labels_read") is False
                and row.get("plan_sha256") == plan_hash and row.get("protocol_hash") == plan["protocol_hash"]
                and row.get("source_manifest_sha256") == plan["source_manifest_sha256"]
                and row.get("mask_id") == "".join(map(str, mask)) and row.get("mask_sha256") == digest(mask),
                "Unknown/duplicate/scored graph row or invalid source identity")
        indexed[key] = row
        by_unit[unit_id(key[0], key[1])].append(row)
    require(set(indexed) == set(expected) and len(observation_rows) == EXPECTED_OUTPUTS, "Incomplete unique G2 graph coverage")
    cost_audit = read_object(root, global_marker["artifacts"]["physical_cost_audit"]["path"])
    require(cost_audit.get("schema") == "graph_credit_g2_physical_cost_v1"
            and cost_audit.get("plan_sha256") == plan_hash and cost_audit.get("labels_read") is False,
            "Physical cost audit belongs to a different generation")
    require(cost_audit.get("inference_retries") == 0 and cost_audit.get("unrecorded_inflight_call_upper_bound") == 0,
            "Complete G2 generation cannot silently retry inference or erase unresolved inflight work")
    qualification_relative = base + "/GLOBAL_QUALIFICATION.json"
    qualification = read_object(root, qualification_relative)
    _add_task(tasks, qualification_relative, file_sha(root, qualification_relative))
    require(qualification.get("schema") == "graph_credit_g2_gpu_qualification_v1" and qualification.get("status") == "PASS"
            and qualification.get("plan_sha256") == plan_hash
            and qualification.get("source_manifest_sha256") == plan["source_manifest_sha256"]
            and qualification.get("pipeline_source_manifest_sha256") == pipeline_hash
            and qualification.get("labels_read") is False, "Global physical-GPU qualification is not bound PASS")
    signatures = set()
    for worker in plan["qualification"]["worker_ids"]:
        report_relative = base + "/qualification/" + worker + "/QUALIFICATION.json"
        report = read_object(root, report_relative)
        bound = qualification["gpu_reports"][worker.removeprefix("gpu")]
        _add_task(tasks, report_relative, bound["report_sha256"])
        require(report.get("status") == "PASS" and report.get("exact_replay") is True
                and report.get("worker_id") == worker and report.get("plan_sha256") == plan_hash
                and report.get("protocol_hash") == plan["protocol_hash"]
                and report.get("source_manifest_sha256") == plan["source_manifest_sha256"]
                and report.get("pipeline_source_manifest_sha256") == pipeline_hash
                and report.get("model_identity") == plan["model"]["identity"]
                and report.get("signature_sha256") == bound["signature_sha256"] and report.get("labels_read") is False,
                "A physical GPU execution signature differs from its frozen qualification")
        signatures.add(bound["signature_sha256"])
    require(len(plan["qualification"]["worker_ids"]) == 8 and len(signatures) == 1,
            "All eight physical GPUs must share one qualified execution signature")
    for category in ("physical_ledgers", "qualification_ledgers", "unit_receipts"):
        for descriptor in cost_audit[category]:
            _descriptor(root, descriptor, tasks)
    return {"root": root, "plan": plan, "plan_hash": plan_hash, "base": base, "analysis": analysis,
            "analysis_hash": analysis_hash, "prediction": prediction, "prediction_hash": prediction_hash,
            "parameters": parameters, "predictions": predictions, "global": global_marker,
            "global_hash": global_hash, "local": local, "local_hash": local_hash,
            "phi": dict(zip(qids, phi_array)), "label_hashes": label_hashes, "access_events": access_events,
            "tasks": tasks, "sizes": sizes, "units": units, "indexed": indexed, "by_unit": by_unit,
            "rows": observation_rows, "cost_audit": cost_audit}


def _raw_validator(row, segment):
    def validate(original):
        require(isinstance(original, dict) and row["execution_path"] == segment + "/" + original["execution_path"],
                "Global execution remapping differs from original raw")
        require(all(row.get(key) == value for key, value in original.items() if key != "execution_path"),
                "Scored/global index changed an original raw field")
        return None
    return validate


def _execution_validator(row, plan):
    def validate(execution):
        require(isinstance(execution, dict) and execution.get("question_id") == row["question_id"]
                and execution.get("seed") == row["seed"] and execution.get("mask") == row["graph_mask"]
                and execution.get("execution_id") == row["execution_id"] and execution.get("protocol_hash") == plan["protocol_hash"]
                and execution.get("question_fingerprint") == plan["questions"]["payload_sha256"][row["question_id"]],
                "Execution content differs from raw/frozen plan")
        nodes = execution.get("nodes", {})
        require(set(nodes) == {"aggregate", *(f"r{round_index}:w{worker}" for round_index in (1, 2) for worker in range(4))},
                "Execution does not contain exactly nine expanded nodes")
        require(nodes["aggregate"]["generation"]["text"] == row["service_output"], "Raw service text differs from actual aggregate generation")
        fresh, referenced = [], []
        for name, node in nodes.items():
            require(type(node.get("reused")) is bool, "Execution cache flag is missing")
            expected_seed = int(digest([row["question_id"], row["seed"], name])[:16], 16) % (2**31 - 1)
            require(node.get("seed") == expected_seed, "Node random stream differs from frozen seed derivation")
            call_id = node["generation"]["call_id"]
            require(isinstance(call_id, str) and call_id, "Physical call identity is missing")
            referenced.append(call_id)
            if not node["reused"]:
                fresh.append(call_id)
        require(len(set(fresh)) == len(fresh), "Duplicate fresh node call in one execution")
        return {"fresh_ids": fresh, "referenced_ids": referenced}
    return validate


def _unit_snapshot(context, uid, unit):
    root, plan, base = context["root"], context["plan"], context["base"]
    private = "artifacts/private/g2/" + plan["run_id"] + "/mirrored_units/" + uid + ".json"
    receipt, receipt_hash, receipt_bytes = _snapshot_object(root, private)
    require(receipt.get("schema") == "graph_credit_g2_mirrored_unit_v1" and receipt.get("run_id") == plan["run_id"]
            and receipt.get("unit_id") == uid and receipt.get("question_id") == unit["question_id"]
            and receipt.get("seed") == unit["seed"] and receipt.get("plan_sha256") == context["plan_hash"]
            and receipt.get("labels_read") is False and receipt.get("final_full_verification_required") is True,
            "Local unit mirror receipt identity differs", private)
    worker = receipt["worker_id"]
    require(worker in plan["qualification"]["worker_ids"], "Unit mirror owner was not a qualified GPU")
    segment = base + "/segments/" + worker
    unit_relative = segment + "/units/" + uid
    require(receipt.get("unit_relative_path") == unit_relative, "Unit mirror points to another owner/path")
    claim_relative, done_relative = base + "/claims/" + uid + ".json", base + "/claims/" + uid + ".done.json"
    claim = read_object(root, claim_relative)
    done = read_object(root, done_relative)
    common = {"question_id": unit["question_id"], "seed": unit["seed"], "worker_id": worker,
              "plan_sha256": context["plan_hash"], "labels_read": False}
    require(claim.get("schema") == "g2_collection_claim_v1" and done.get("schema") == "g2_collection_done_v1"
            and all(claim.get(key) == value and done.get(key) == value for key, value in common.items())
            and done.get("unit_id") == uid and done.get("claim_sha256") == receipt["claim_sha256"]
            and done.get("unit_frozen_sha256") == receipt["unit_frozen_sha256"]
            and done.get("unit_frozen_path") == "segments/" + worker + "/units/" + uid + "/UNIT_OUTPUTS_FROZEN.json",
            "Actual claim/done differs from local source-bound receipt")
    marker = read_object(root, unit_relative + "/UNIT_OUTPUTS_FROZEN.json")
    local_rows = context["by_unit"][uid]
    mask_ids = {"".join(map(str, request["mask"])) for request in unit["mask_requests"]}
    require(marker.get("schema") == "g2_collection_unit_outputs_v1" and marker.get("question_id") == unit["question_id"]
            and marker.get("seed") == unit["seed"] and marker.get("plan_sha256") == context["plan_hash"]
            and marker.get("protocol_hash") == plan["protocol_hash"] and marker.get("labels_read") is False
            and marker.get("first_gold_opened_unix") is None
            and marker.get("observations") == done.get("observations") == len(local_rows) == len(mask_ids),
            "Actual unit output freeze is incomplete")
    members = receipt["members"]
    expected_members = {"UNIT_PLAN.json", "UNIT_OUTPUTS_FROZEN.json"}
    expected_members.update(kind + "/" + mask + ".json" for kind in ("raw", "executions") for mask in mask_ids)
    require(set(members) == expected_members, "Unit mirror has missing/extra/runtime members")
    require(members["UNIT_OUTPUTS_FROZEN.json"]["sha256"] == receipt["unit_frozen_sha256"], "Unit marker source hash differs")
    reference = {item["mask_sha256"]: item for item in marker["rows"]}
    require(len(reference) == len(marker["rows"]) == len(local_rows), "Duplicate unit freeze reference")
    tasks, sizes, validators = {}, {}, {}
    _add_task(tasks, private, receipt_hash)
    _add_task(tasks, claim_relative, receipt["claim_sha256"])
    _add_task(tasks, done_relative, receipt["done_sha256"])
    for name, descriptor in members.items():
        relative = unit_relative + "/" + name
        _add_task(tasks, relative, descriptor["sha256"])
        sizes[relative] = descriptor["bytes"]
    for row in local_rows:
        mid = row["mask_id"]
        item = reference[row["mask_sha256"]]
        require(row.get("worker_id") == worker and row.get("unit_id") == uid
                and row["raw_path"] == unit_relative + "/raw/" + mid + ".json"
                and row["execution_path"] == unit_relative + "/executions/" + mid + ".json"
                and row["raw_sha256"] == members["raw/" + mid + ".json"]["sha256"] == item["raw_sha256"]
                and row["execution_sha256"] == members["executions/" + mid + ".json"]["sha256"] == item["execution_sha256"]
                and item["raw_path"] == "units/" + uid + "/raw/" + mid + ".json"
                and item["execution_path"] == "units/" + uid + "/executions/" + mid + ".json",
                "Global/claim/unit/member execution reference mismatch")
        validators[row["raw_path"]] = _raw_validator(row, segment)
        validators[row["execution_path"]] = _execution_validator(row, plan)
    return {"descriptor": {"path": private, "sha256": receipt_hash, "bytes": receipt_bytes},
            "tasks": tasks, "sizes": sizes, "validators": validators}


def verify_mirror_snapshot(context, progress_callback=None):
    root, tasks, sizes = context["root"], context["tasks"], context["sizes"]
    receipt_directory = root / "artifacts/private/g2" / context["plan"]["run_id"] / "mirrored_units"
    require({path.stem for path in receipt_directory.glob("*.json")} == set(context["units"]), "Local mirror receipt IDs are incomplete or extra")
    descriptors, validators = [], {}
    pending = sorted(context["units"].items())
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="g2-unit-snapshot") as pool:
        for start in range(0, len(pending), 64):
            batch = pending[start:start + 64]
            for value in pool.map(lambda item: _unit_snapshot(context, *item), batch):
                descriptors.append(value["descriptor"])
                for relative, expected in value["tasks"].items():
                    _add_task(tasks, relative, expected)
                sizes.update(value["sizes"])
                validators.update(value["validators"])
            if progress_callback:
                progress_callback({"phase": "SNAPSHOTTING_UNIT_RECEIPTS", "completed": len(descriptors), "total": 2000})
    require(digest(sorted(descriptors, key=lambda item: item["path"])) == context["local"]["unit_receipts_sha256"],
            "Local completion does not bind the actual 2000 mirror receipt bytes")
    summary = {"fresh_ids": set(), "referenced_ids": set()}
    hashes = bounded_hash_verification(root, tasks, validators=validators, sizes=sizes,
                                       execution_summary=summary, progress_callback=progress_callback)
    require(len(validators) == EXPECTED_OUTPUTS * 2, "Raw/execution paths are not unique across all graph outputs")
    context.update(execution_summary=summary, verified_hashes=hashes)
    return {"status": "PASS", "unique_graph_outputs": EXPECTED_OUTPUTS, "mirrored_units": 2000,
            "verified_files": len(hashes), "raw_execution_file_reads": len(validators), "io_workers": 6,
            "input_hash_manifest_sha256": digest(hashes), "unit_receipts_sha256": context["local"]["unit_receipts_sha256"]}


def open_audit_labels(context):
    """The caller must finish every terminal/byte gate before invoking this."""
    require("verified_hashes" in context, "Independent label access requires completed mirror verification")
    root, plan, base = context["root"], context["plan"], context["base"]
    relative = base + "/AUDIT_FIRST_G2_LABELS_OPENED.json"
    binding = {"plan_sha256": context["plan_hash"], "global_outputs_frozen_sha256": context["global_hash"],
               "predictions_frozen_sha256": context["prediction_hash"], "analysis_sha256": context["analysis_hash"],
               "local_mirror_complete_sha256": context["local_hash"], "expected_label_sha256": context["label_hashes"]}
    prior = read_object(root, relative) if safe_path(root, relative, exists=False).exists() else None
    if prior is not None:
        require(prior.get("schema") == "graph_credit_g2_independent_first_label_access_v1"
                and all(prior.get(key) == value for key, value in binding.items()), "Independent first-read event has different frozen inputs")
    with ExitStack() as stack:
        handles, opened = {}, {}
        for partition in COUNTS:
            label_relative = plan["label_release"][partition]
            handles[partition] = stack.enter_context(safe_path(root, label_relative).open("rb"))
            opened[partition] = {"path": label_relative, "handle_opened_unix": time.time()}
        event = {"schema": "graph_credit_g2_independent_first_label_access_v1",
                 "event": "HANDLES_OPENED_BEFORE_FIRST_READ", "created_unix": time.time(), "handles": opened,
                 "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **binding}
        event_hash = file_sha(root, relative) if prior is not None else atomic_json(root, relative, event, exclusive=True)
        attempt = base + "/audit_label_attempts/" + uuid.uuid4().hex[:20] + ".json"
        atomic_json(root, attempt, {**event, "first_event_sha256": event_hash}, exclusive=True)
        labels = {}
        for partition, handle in handles.items():
            context["audit_labels_read"] = True
            content = handle.read()
            require(hashlib.sha256(content).hexdigest() == context["label_hashes"][partition], "Independent label bytes differ from released gold")
            require(content.endswith(b"\n"), "Incomplete released label JSONL")
            values = {}
            for line in content.splitlines():
                require(line.strip(), "Blank released label record")
                value = decode(line)
                require(isinstance(value, dict) and value.get("id") not in values, "Duplicate/malformed split label")
                values[value["id"]] = value
            require(set(values) == {qid for qid, part in plan["partitions"].items() if part == partition}
                    and len(values) == COUNTS[partition], "Released split label coverage differs")
            labels.update(values)
    return labels, {"first_access_event_path": relative, "first_access_event_sha256": event_hash,
                    "current_attempt_path": attempt, "first_access_event_reused": prior is not None}


def score_replay(context, labels, progress_callback=None):
    root = context["root"]
    source = "src/graph_credit/scoring.py"
    check_hash(root, source, ORIGINAL_SCORER_SHA)
    specification = importlib.util.spec_from_file_location("g2_independent_pinned_original_scorer", safe_path(root, source))
    scorer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(scorer)
    scored, mismatches = {}, []
    for index, row in enumerate(context["rows"], 1):
        key = (row["question_id"], row["seed"], tuple(row["graph_mask"]))
        value = independent_score(row["service_output"], labels[row["question_id"]])
        official = float(scorer.score_item(row["service_output"], labels[row["question_id"]], task="gsm8k"))
        require(math.isfinite(official) and official in (0, 1), "Original scorer returned invalid quality")
        if value != official:
            mismatches.append(key)
        scored[key] = value
        if progress_callback and index % 1000 == 0:
            progress_callback({"phase": "REPLAYING_TWO_SCORERS", "completed": index, "total": len(context["rows"])})
    if mismatches:
        error = AuditError("Independent/original raw-output scoring differs on " + str(len(mismatches)) + " graphs")
        error.score_mismatch_count = len(mismatches)
        raise error
    return scored, {"status": "PASS", "raw_outputs_replayed": len(scored), "mismatch_count": 0,
                    "comparison": "Independent Decimal scorer versus pinned original scorer on real raw service_output",
                    "production_per_row_scored_log": "Not produced by the frozen pipeline; no row-log comparison claimed",
                    "original_scorer_sha256": ORIGINAL_SCORER_SHA}


def quality_table(scored):
    grouped = defaultdict(dict)
    for (qid, seed, mask), value in scored.items():
        require(seed not in grouped[(qid, mask)], "Repeated same-graph seed quality")
        grouped[(qid, mask)][seed] = value
    means = {}
    for key, by_seed in grouped.items():
        require(set(by_seed) == set(SEEDS), "Graph quality mean lacks five seeds")
        means[key] = math.fsum(by_seed.values()) / 5
    return means


def _contrast_matrix(contrasts, phi, order):
    return np.asarray([psi(phi[row["question_id"]], row["mask1"], order)
                       - psi(phi[row["question_id"]], row["mask0"], order) for row in contrasts])


def _mean_credits(means, contrasts):
    return np.asarray([means[(row["question_id"], row["mask1"])] - means[(row["question_id"], row["mask0"])] for row in contrasts])


def _macro_mse(values, targets, contrasts):
    by_question = defaultdict(list)
    for row, value, target in zip(contrasts, values, targets):
        by_question[row["question_id"]].append((float(value) - float(target)) ** 2)
    require(all(len(losses) == 24 for losses in by_question.values()), "Tuning requires all 24 targets per question")
    return math.fsum(math.fsum(losses) / 24 for losses in by_question.values()) / len(by_question)


def ridge_and_prediction_replay(context, scored, progress_callback=None):
    plan, parameters, phi = context["plan"], context["parameters"], context["phi"]
    require(parameters.get("phi_order") == "PC1..PC8,1" and parameters.get("tensor_order") == "phi_major_kron"
            and parameters.get("all_coefficients_penalized") is True
            and parameters.get("alpha_grid") == list(ALPHAS) and parameters.get("alpha_tie_atol") == 1e-12,
            "Frozen predictor representation/regularization metadata differs")
    require(set(parameters["train_question_ids"]) == {qid for qid, part in plan["partitions"].items() if part == "train"}
            and len(parameters["train_question_ids"]) == 200
            and set(parameters["tune_question_ids"]) == {qid for qid, part in plan["partitions"].items() if part == "tune"}
            and len(parameters["tune_question_ids"]) == 80, "Predictor training/tuning populations differ")
    means = quality_table(scored)
    train_keys = sorted(key for key in means if plan["partitions"][key[0]] == "train")
    counts = Counter(qid for qid, mask in train_keys)
    targets = np.asarray([means[key] for key in train_keys])
    weights = np.asarray([1.0 / counts[qid] for qid, mask in train_keys])
    require(near(weights.sum(), 200), "Independent training weights do not sum to 200 questions")
    require(parameters.get("unique_training_masks_by_question") == dict(counts)
            and parameters.get("train_unique_graph_rows") == len(train_keys), "Training aliases became extra regression rows")
    train_contrasts, tune_contrasts, test_contrasts = (_contrasts(plan, partition) for partition in ("train", "tune", "test"))
    baseline = []
    for edge in range(12):
        terms = []
        for row in train_contrasts:
            if row["edge"] == edge:
                for seed in SEEDS:
                    terms.append(scored[(row["question_id"], seed, row["mask1"])] - scored[(row["question_id"], seed, row["mask0"])])
        baseline.append(math.fsum(terms) / len(terms))
    require(near(baseline, parameters["B0"], coefficient=True), "Train question/anchor/seed baseline differs")
    independent_models, manual_predictions = {}, {"B0": np.asarray([baseline[row["edge"]] for row in test_contrasts])}
    tune_targets = _mean_credits(means, tune_contrasts)
    for name, order, dimension in (("C1", 1, 117), ("C2", 2, 711)):
        matrix = np.asarray([psi(phi[qid], mask, order) for qid, mask in train_keys])
        require(matrix.shape == (len(train_keys), dimension), "Independent design dimension differs")
        path = solve_ridge_path(matrix, targets, weights)
        tune_matrix = _contrast_matrix(tune_contrasts, phi, order)
        scores = {alpha: _macro_mse(tune_matrix @ coefficient, tune_targets, tune_contrasts) for alpha, coefficient in path.items()}
        chosen = select_alpha(scores)
        saved = parameters["models"][name]
        require(saved.get("alpha") == chosen and saved.get("order") == order and saved.get("dimensions") == dimension,
                "Independent tuning/tie-break selected another alpha for " + name)
        require(len(saved["alpha_scores"]) == 8 and {item["alpha"] for item in saved["alpha_scores"]} == set(ALPHAS), "Saved tuning grid differs")
        for item in saved["alpha_scores"]:
            require(near(item["tune_credit_mse"], scores[item["alpha"]]), "Independent tune risk differs for " + name)
        require(near(saved["coefficients"], path[chosen], coefficient=True), "Independent Cholesky coefficients differ for " + name)
        require(near(saved["tune_credit_mse"], scores[chosen]), "Selected tuning risk differs")
        # Arithmetic verification uses the actual frozen coefficients. Their
        # independent Cholesky fit was checked above within declared tolerance.
        manual_predictions[name] = _contrast_matrix(test_contrasts, phi, order) @ np.asarray(saved["coefficients"])
        independent_models[name] = {"alpha": chosen, "dimensions": dimension, "tune_credit_mse": scores[chosen],
                                    "maximum_absolute_coefficient_difference": float(np.max(np.abs(path[chosen] - saved["coefficients"])))}
        if progress_callback:
            progress_callback({"phase": "INDEPENDENT_CHOLESKY_REPLAY", "model": name, "alpha": chosen})
    predicted = {}
    for row in context["predictions"]:
        key = (row["question_id"], row["anchor_id"], row["edge"])
        require(key not in predicted and row.get("parameter_sha256") == parameters["parameter_sha256"], "Duplicate/unbound prediction")
        predicted[key] = row
    for index, contrast in enumerate(test_contrasts):
        key = (contrast["question_id"], contrast["anchor_id"], contrast["edge"])
        require(key in predicted, "Heldout prediction coordinate missing")
        row = predicted[key]
        require(row["mask0"] == list(contrast["mask0"]) and row["mask1"] == list(contrast["mask1"])
                and set(row["predictions"]) == set(MODEL_NAMES), "Frozen prediction uses another signed graph contrast")
        for model in MODEL_NAMES:
            require(near(row["predictions"][model], manual_predictions[model][index], coefficient=True), "Manual kron/dot prediction differs")
    require(len(predicted) == len(test_contrasts) == 2880, "Unexpected heldout prediction support")
    return predicted, {"status": "PASS", "implementation": "independent weighted Gram Cholesky/cho_solve",
                       "training_questions": 200, "tuning_questions": 80, "unique_training_graph_rows": len(train_keys),
                       "sum_training_row_weights": float(weights.sum()), "all_coefficients_penalized": True,
                       "refit_with_tune": False, "models": independent_models,
                       "coefficient_tolerance": {"absolute": COEFFICIENT_ATOL, "relative": COEFFICIENT_RTOL}},


def statistics_replay(context, scored, predicted):
    question_risks = defaultdict(lambda: defaultdict(list))
    for row in _contrasts(context["plan"], "test"):
        qid = row["question_id"]
        key = (qid, row["anchor_id"], row["edge"])
        credits = [scored[(qid, seed, row["mask1"])] - scored[(qid, seed, row["mask0"])] for seed in SEEDS]
        values = predicted[key]["predictions"]
        risks = {name: cross_seed_risk(values[name], credits) for name in MODEL_NAMES}
        noise_cancelled = (values["B0"] - np.mean(credits)) ** 2 - (values["C2"] - np.mean(credits)) ** 2
        require(near(risks["B0"] - risks["C2"], noise_cancelled), "Independent paired-noise cancellation identity failed")
        for name, value in risks.items():
            question_risks[qid][name].append(value)
    per_question = {qid: {name: math.fsum(by_model[name]) / 24 for name in MODEL_NAMES}
                    for qid, by_model in question_risks.items()}
    require(len(per_question) == 120 and all(len(values) == 24 for by_model in question_risks.values() for values in by_model.values()),
            "Incomplete independent heldout question/anchor/edge support")
    primary = bootstrap_question_difference([per_question[qid]["B0"] - per_question[qid]["C2"] for qid in sorted(per_question)])
    diagnostic = bootstrap_question_difference([per_question[qid]["C1"] - per_question[qid]["C2"] for qid in sorted(per_question)])
    analysis = context["analysis"]
    for name, independent in (("primary", primary), ("first_order_diagnostic", diagnostic)):
        saved = analysis[name]
        require(saved.get("question_count") == 120 and saved.get("bootstrap_samples") == 10000
                and saved.get("bootstrap_seed") == 39173 and saved.get("confidence") == .95
                and saved.get("resampling_unit") == "whole_question"
                and near(saved["estimate"], independent["estimate"]) and near(saved["interval"], independent["interval"]),
                "Independent paired question bootstrap differs for " + name)
    require(set(analysis["question_risks"]) == set(per_question), "Saved question-risk support differs")
    for qid, values in per_question.items():
        for name, value in values.items():
            require(near(value, analysis["question_risks"][qid][name]), "Independent per-question risk differs")
    means = {name: math.fsum(values[name] for values in per_question.values()) / 120 for name in MODEL_NAMES}
    require(all(near(means[name], analysis["mean_cross_seed_risk"][name]) for name in MODEL_NAMES), "Mean cross-seed risk differs")
    gate = "G2_PASS" if primary["interval"][0] > 0 else "INSUFFICIENT_PREDICTIVE_GAIN"
    require(gate == analysis["scientific_gate"], "Independent scientific gate differs")
    return {"status": "PASS", "primary": primary, "first_order_diagnostic": diagnostic,
            "mean_cross_seed_risk": means, "negative_question_risks": sum(value < 0 for values in per_question.values() for value in values.values()),
            "negative_values_retained": True, "scientific_gate": gate,
            "comparison_tolerance": {"absolute": STAT_ATOL, "relative": STAT_RTOL}}


def cost_replay(context):
    root, audit = context["root"], context["cost_audit"]
    metrics = ("input_tokens", "output_tokens", "gpu_seconds", "elapsed_seconds")
    all_ids, formal_ids, successful_ids, failed_ids = set(), set(), set(), set()
    summaries = {}
    for name, descriptor_key in (("formal", "physical_ledgers"), ("qualification", "qualification_ledgers")):
        totals = {metric: [] for metric in metrics}
        missing, statuses = Counter(), Counter()
        count = 0
        for descriptor in audit[descriptor_key]:
            # Ledger bytes were checked in the parallel pre-label snapshot.
            for row in rows(root, descriptor["path"]):
                call_id = row.get("call_id")
                require(isinstance(call_id, str) and call_id and call_id not in all_ids, "Physical ledger ID is missing or duplicated")
                require(row.get("status") in {"SUCCESS", "FAILED"}, "Physical ledger status is invalid")
                all_ids.add(call_id)
                statuses[row["status"]] += 1
                count += 1
                if name == "formal":
                    formal_ids.add(call_id)
                    if row["status"] == "SUCCESS":
                        successful_ids.add(call_id)
                    else:
                        failed_ids.add(call_id)
                for metric in metrics:
                    value = row.get(metric)
                    if value is None:
                        missing[metric] += 1
                    else:
                        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Malformed ledger cost metric")
                        totals[metric].append(value)
        summaries[name] = {"calls": count, "successful_calls": statuses["SUCCESS"], "failed_calls": statuses["FAILED"],
                           "metrics": {metric: {"known_sum": math.fsum(totals[metric]), "unknown_calls": missing[metric],
                                       "complete_total": math.fsum(totals[metric]) if not missing[metric] else None} for metric in metrics}}
    fresh = context["execution_summary"]["fresh_ids"]
    referenced = context["execution_summary"]["referenced_ids"]
    require(fresh <= successful_ids and referenced <= successful_ids, "An execution references an absent/failed physical call")
    require(summaries["formal"]["calls"] == audit["formal_calls"] <= context["plan"]["formal_call_upper_bound"]
            and audit["referenced_calls"] == len(fresh)
            and set(audit["orphan_call_ids"]) == formal_ids - fresh
            and set(audit["failed_call_ids"]) == failed_ids, "Formal fresh/orphan/failure accounting differs")
    if "qualification_calls" in audit:
        require(audit["qualification_calls"] == summaries["qualification"]["calls"], "Qualification call count differs from its paid ledgers")
    saved = read_object(root, context["base"] + "/G2_COST_SUMMARY.json")
    for name in ("formal", "qualification"):
        claimed = saved["generation"][name]
        actual = summaries[name]
        require(all(claimed[field] == actual[field] for field in ("calls", "successful_calls", "failed_calls")), "CPU workflow cost call counts differ")
        for metric in metrics:
            for field in ("known_sum", "unknown_calls", "complete_total"):
                left, right = claimed["metrics"][metric][field], actual["metrics"][metric][field]
                require(left is None and right is None or left is not None and right is not None
                        and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-6), "CPU workflow cost total/unknown count differs")
    attempts, known_cpu, unknown_cpu = [], [], 0
    for path in sorted((root / context["base"] / "cpu_attempts").glob("*.json")):
        relative = path.relative_to(root).as_posix()
        event = read_object(root, relative)
        require(event.get("plan_sha256") == context["plan_hash"] and event.get("mode") in {"predict", "evaluate"}, "CPU attempt belongs to another plan")
        value = event.get("cpu_seconds")
        known = event.get("state") in {"COMPLETED", "FAILED"} and type(value) in (int, float) and math.isfinite(value) and value >= 0
        if known:
            known_cpu.append(value)
        else:
            unknown_cpu += 1
        attempts.append({"path": relative, "sha256": file_sha(root, relative), "state": event.get("state"),
                         "cpu_seconds": value if known else None})
    feature_receipt_path = context["plan"]["features"]["receipt"]
    feature_receipt = read_object(root, feature_receipt_path)
    preprocessing = {"cpu_seconds": None, "cpu_status": "UNKNOWN_NOT_RECORDED",
                     "feature_receipt_sha256": file_sha(root, feature_receipt_path),
                     "feature_receipt_created_unix": feature_receipt.get("created_unix"),
                     "observed_launch_to_feature_receipt_wall_interval_seconds": None,
                     "scope": "Initial CPU sentence encoding and train-only PCA preparation"}
    launch_relative = "artifacts/audit/G2_FEATURE_PREPARATION_LAUNCH.json"
    if safe_path(root, launch_relative, exists=False).exists():
        launch = read_object(root, launch_relative)
        launch_record = launch.get("launch", launch.get("serverlaunch", {}))
        launched, completed = launch_record.get("launched_unix"), feature_receipt.get("created_unix")
        preprocessing.update(launch_receipt_sha256=file_sha(root, launch_relative), server_launched_unix=launched)
        if all(type(value) in (int, float) and math.isfinite(value) for value in (launched, completed)) and completed >= launched:
            preprocessing["observed_launch_to_feature_receipt_wall_interval_seconds"] = completed - launched
        preprocessing["wall_window_scope"] = "Observed interval between launch and feature-completion receipts includes startup/waiting; CPU duration remains unknown"
    return {"status": "PASS", "formal": summaries["formal"], "qualification": summaries["qualification"],
            "orphan_formal_calls": len(formal_ids - fresh), "fresh_formal_calls": len(fresh),
            "cpu_attempts": attempts, "known_cpu_attempt_seconds": math.fsum(known_cpu), "unknown_cpu_attempts": unknown_cpu,
            "complete_cpu_attempt_seconds": math.fsum(known_cpu) if not unknown_cpu else None,
            "preprocessing_cost_cpu_unknown": True, "preprocessing": preprocessing,
            "inference_retries": audit.get("inference_retries"), "unrecorded_inflight_call_upper_bound": audit.get("unrecorded_inflight_call_upper_bound"),
            "scope": "Each G2 formal/qualification physical ledger once; aliases/backup copies excluded; CPU attempts separate; earlier G1 phases retain their own ledgers",
            "unknown_costs_imputed": False}


def audit(root, plan_path, progress_callback=None):
    root = Path(root).resolve(strict=True)
    plan_path = Path(plan_path)
    plan_relative = plan_path.relative_to(root).as_posix() if plan_path.is_absolute() else plan_path.as_posix()
    started, cpu_started = time.time(), time.process_time()
    report = {"schema": "graph_credit_g2_independent_acceptance_v1", "status": "WAITING_NOT_TERMINAL",
              "labels_read": False, "scientific_completion": False, "g2_stage_complete": False,
              "project_complete": False, "scientific_completion_scope": "G2 independent scientific acceptance only", "started_unix": started,
              "scope": "G2 heldout credit prediction, frozen-byte integrity and independent arithmetic replay"}
    context = None
    try:
        context = terminal_metadata(root, plan_relative)
        report["run_id"] = context["plan"]["run_id"]
        report["plan_sha256"] = context["plan_hash"]
        report["integrity"] = verify_mirror_snapshot(context, progress_callback)
        labels, access = open_audit_labels(context)
        report.update(labels_read=True, audit_label_access=access)
        scored, scoring = score_replay(context, labels, progress_callback)
        report["score_replay"] = scoring
        predicted, ridge = ridge_and_prediction_replay(context, scored, progress_callback)
        report["ridge_replay"] = ridge
        report["prediction_replay"] = {"status": "PASS", "rows": len(predicted), "implementation": "independent binary graph basis and phi-major outer product/dot"}
        statistics = statistics_replay(context, scored, predicted)
        report["statistics"] = statistics
        report["scientific_gate"] = statistics["scientific_gate"]
        report["costs"] = cost_replay(context)
        # Recheck small terminal anchors after numerical work. Raw/execution
        # members were read once in the parallel immutable snapshot above.
        for name, expected in (("GLOBAL_OUTPUTS_FROZEN.json", context["global_hash"]),
                               ("PREDICTIONS_FROZEN.json", context["prediction_hash"]),
                               ("G2_ANALYSIS.json", context["analysis_hash"]),
                               ("LOCAL_MIRROR_COMPLETE.json", context["local_hash"])):
            check_hash(root, context["base"] + "/" + name, expected)
        report.update(status="ACCEPTED_WITH_DOCUMENTED_LIMITATIONS", scientific_completion=True,
                      g2_stage_complete=True, project_complete=False,
                      G3_contract_state="G2_PRIMARY_GATE_MET_OTHER_CONTRACT_CHECKS_REMAIN" if report["scientific_gate"] == "G2_PASS" else "STOP_G3_UNDER_CURRENT_CONTRACT",
                      label_timing={"status": "DURABLE_FREEZE_RELEASE_FIRST_OPEN_HASH_CHAIN_VERIFIED",
                          "evidence": "Global freeze bytes bind train/tune release and persistent first-open event; prediction freeze bytes bind heldout release and persistent first-open event",
                          "same_server_event_order_verified": True,
                          "cross_host_release_clock_order": "Hash dependencies establish order; local/server wall clocks are not assumed synchronized",
                          "production_events": context["access_events"]},
                      reference_source_pins={"original_source_manifest": ORIGINAL_SOURCE_SHA, "original_scorer": ORIGINAL_SCORER_SHA,
                                             "pipeline": PIPELINE_PINS, "protocol": PROTOCOL_SHA, "calibration_questions": CALIBRATION_SHA},
                      artifact_sha256={"global_outputs_frozen": context["global_hash"], "predictions_frozen": context["prediction_hash"],
                                       "analysis": context["analysis_hash"], "local_mirror_complete": context["local_hash"]},
                      feature_verification="Frozen feature/PCA artifact bytes, training-ID receipt and source contract")
    except (WaitingError, FileNotFoundError) as error:
        report.update(status="WAITING_NOT_TERMINAL", reason=str(error))
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error), relative_path=getattr(error, "relative_path", None))
        if hasattr(error, "score_mismatch_count"):
            report["score_replay"] = {"status": "FAIL", "mismatch_count": error.score_mismatch_count,
                                      "comparison": "Independent versus pinned original scorer on raw service outputs"}
    finally:
        report.update(labels_read=bool(context and context.get("audit_labels_read")),
                      completed_unix=time.time(), audit_cpu_seconds=time.process_time() - cpu_started,
                      auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    result = audit(root, args.plan)
    output = args.output or root / "artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json"
    relative = output.relative_to(root).as_posix() if output.is_absolute() else output.as_posix()
    atomic_json(root, relative, result)
    print(canonical({key: result.get(key) for key in ("status", "run_id", "scientific_gate", "labels_read", "scientific_completion", "g2_stage_complete", "project_complete")}), flush=True)
    return 0 if result["status"] == "ACCEPTED_WITH_DOCUMENTED_LIMITATIONS" else 3 if result["status"] == "WAITING_NOT_TERMINAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
