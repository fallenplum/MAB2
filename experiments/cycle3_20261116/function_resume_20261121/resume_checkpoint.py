"""Verify a partial Formal10 archive and validate the complementary work list."""
import hashlib
import json
from pathlib import Path

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()

def load(path):
    return json.loads(path.read_bytes())

def require(value, reason):
    if not value:
        raise ValueError(reason)

def verified_units(archive, archive_sha, original_plan, questions, selection):
    manifest_path = archive / "ARCHIVE_MANIFEST.json"
    require(digest(manifest_path) == archive_sha, "Parent manifest changed")
    manifest = load(manifest_path)
    for rel, expected in manifest.items():
        path = archive / rel
        require(path.resolve().is_relative_to(archive.resolve()), "Parent archive path escaped")
        require(digest(path) == expected, "Parent archive file changed")
    require("CLOSURE.json" in manifest, "Closure missing from manifest")
    closure = load(archive / "CLOSURE.json")
    require(closure["status"] == "PARTIAL_OR_FAILED" and closure["all_workers_exited"] and closure["plan_sha256"] == digest(original_plan), "Unexpected parent closure")
    plan = load(original_plan)
    require(digest(questions) == plan["questions_sha256"] and digest(selection) == plan["selection_sha256"], "Input pins differ")
    rows = [json.loads(line) for line in questions.read_bytes().splitlines()]
    selected = load(selection)["questions"]
    require(len(rows) == len(selected) == 16, "Original pilot coverage differs")
    verified = {}
    seeds = plan["seeds"]
    conditions = {"G0", "G1", "G0_MINUS_EU", "G0_MINUS_EV", "G1_MINUS_EU", "G1_MINUS_EV", "SINGLE_Q", "SINGLE_C", "BOK3", "REPAIR3"}
    for marker in archive.glob("worker_*/unit_*/COMPLETE.json"):
        unit = marker.parent
        for name in ("INPUT.json", "MATRIX.json", "COMPLETE.json"):
            require((unit / name).relative_to(archive).as_posix() in manifest, "Unpinned unit")
        complete = load(marker)
        index = complete["index"]
        require(type(index) is int and 0 <= index < 16 and index not in verified, "Invalid or duplicate completed unit")
        require(unit.name == f"unit_{index:03d}", "Unit index/path mismatch")
        worker = unit.parent.name.removeprefix("worker_")
        require(index in plan["assignments"][worker], "Unexpected worker assignment")
        proof = load(unit / "INPUT.json")
        require(proof == {"index":index,"payload":rows[index],"shock":selected[index]["removed_upstream"]}, "Input identity differs")
        encoded = json.dumps(rows[index], ensure_ascii=False, sort_keys=True, separators=(",",":")).encode()
        require(hashlib.sha256(encoded + b"\n").hexdigest() == selected[index]["input_sha256"], "Selection hash mismatch")
        matrix_file = unit / "MATRIX.json"
        matrix = load(matrix_file)
        require(digest(matrix_file) == complete["matrix_sha256"], "Matrix pin differs")
        require(matrix["question_fingerprint"] == hashlib.sha256(encoded).hexdigest() and matrix["protocol_sha256"] == plan["qualified_operator_protocol_sha256"], "Operator or task hash differs")
        require(matrix["hidden_test_access"] is False and matrix["execution_seeds"] == seeds, "Unexpected scoring or seeds")
        keys = [(v["execution_seed"], v["condition"]) for v in matrix["conditions"] if v["status"] == "COMPLETE"]
        require(complete["terminal_conditions"] == len(keys) == len(matrix["conditions"]) == 30 and set(keys) == {(s,c) for s in seeds for c in conditions}, "Incomplete task matrix")
        verified[index] = complete["matrix_sha256"]
    return verified

def verify_assignments(remaining, assignments, count):
    require(sorted(remaining) == list(range(6, 16)), "Unexpected restart boundary")
    require(count in (1, 2, 4) and set(assignments) == {str(i) for i in range(count)}, "Unexpected workers")
    require(all(len(indices) >= 2 for indices in assignments.values()), "Insufficient prefix")
    require(sorted(i for indices in assignments.values() for i in indices) == sorted(remaining), "Duplicate or missing work")

def verify_restart(plan):
    require(plan["mode"]=="formal" and 0 < plan["aggregate_reservation_GPU_hours"] <= 6.0, "Restart budget or mode mismatch")
    parent = plan["resume_parent"]
    archive = Path(parent["archive"])
    original = Path(parent["plan"])
    require(digest(original) == parent["plan_sha256"], "Parent plan changed")
    old = load(original)
    for key in ("seeds","models","import_paths","questions","questions_sha256","selection","selection_sha256","qualified_operator_protocol_sha256","minimum_gpu_free_MiB","minimum_disk_bytes","prefix_per_worker"):
        require(plan[key] == old[key], "Frozen setting changed: " + key)
    require(all(plan["source_pins"].get(k) == v for k,v in old["source_pins"].items()), "Qualified source bundle changed")
    require(plan["output"] != old["output"], "Refusing parent overwrite")
    done = verified_units(archive, parent["archive_sha256"], original, Path(plan["questions"]), Path(plan["selection"]))
    verify_assignments(set(range(16)) - done.keys(), plan["assignments"], len(plan["gpus"]))
