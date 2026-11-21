"""Reconcile private physical call journals with complete per-task matrices."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value, reason):
    if not value:
        raise ValueError(reason)


def summarize(archive, manifest_sha):
    manifest_path = archive / "ARCHIVE_MANIFEST.json"
    check(digest(manifest_path) == manifest_sha, "Manifest pin differs")
    manifest = read(manifest_path)
    files = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    check(files == set(manifest) | {manifest_path.name}, "Archive coverage differs")
    for name, expected in manifest.items():
        check(digest(archive / name) == expected, "File pin differs")
    closure = read(archive / "CLOSURE.json")
    check(closure["status"] in {"PARTIAL_OR_FAILED", "COMPLETE_UNSCORED"}
          and closure["all_workers_exited"] and not closure["quality_seen"], "Unexpected closure")
    check(math.isclose(closure["actual_GPU_hours"] * 3600,
                       sum(closure["residence_seconds_by_worker"]), abs_tol=1e-8), "Residence mismatch")

    tasks = set()
    traces = {}
    conditions = 0
    for marker in archive.glob("worker_*/unit_*/COMPLETE.json"):
        unit = marker.parent
        matrix_path = unit / "MATRIX.json"
        check(digest(matrix_path) == read(marker)["matrix_sha256"], "Completion marker mismatch")
        matrix = read(matrix_path)
        check(matrix["hidden_test_access"] is False, "Unexpected scoring")
        fingerprint = matrix["question_fingerprint"]
        check(fingerprint not in tasks, "Completed task repeated")
        tasks.add(fingerprint)
        conditions += len(matrix["conditions"])
        for row in matrix["request_trace"]:
            key = (unit.parent.name, row["model"], row["request_id"])
            check(key not in traces, "Repeated matrix trace")
            traces[key] = (fingerprint, row["generation"], row["request_fingerprint"])

    totals = defaultdict(Counter)
    matched, native_ids = set(), set()
    for journal in sorted(archive.glob("worker_*/*_requests.jsonl")):
        worker = journal.parent.name
        model = journal.name.removesuffix("_requests.jsonl")
        events = [json.loads(line) for line in journal.read_bytes().splitlines()]
        check(events[0]["event"] == "HEADER" and events[0]["model"] == model, "Invalid journal header")
        starts, terminals = {}, set()
        for row in events[1:]:
            context = row["context"]
            key = row["request_id"]
            group = "completed_tasks" if context["question_fingerprint"] in tasks else "interrupted_tasks"
            counts = totals[group]
            check(row["model"] == context["model"] == model, "Model differs")
            if row["event"] == "START":
                check(key not in starts, "Duplicate start")
                starts[key] = row
                counts["started"] += 1
                continue
            check(key in starts and key not in terminals, "Orphan or duplicate terminal event")
            for field, value in starts[key].items():
                if field != "event":
                    check(row[field] == value, "Request identity changed")
            terminals.add(key)
            if row["event"] == "ERROR":
                counts["errors"] += 1
                continue
            check(row["event"] == "RETURNED", "Unknown terminal event")
            gen = row["generation"]
            native = row["native_events"]
            check(len(native) == 1 and native[0]["status"] == "SUCCESS", "Native event differs")
            for name in ("call_id", "input_tokens", "output_tokens", "gpu_seconds"):
                check(native[0][name] == gen[name], "Native generation differs")
            check(gen["call_id"] not in native_ids, "Duplicate physical call")
            native_ids.add(gen["call_id"])
            for name in ("input_tokens", "output_tokens"):
                check(type(gen[name]) is int and gen[name] >= 0, "Invalid token count")
                counts[name] += gen[name]
            counts["returned"] += 1
            if group == "completed_tasks":
                trace = (worker, model, context["request_id"])
                check(trace not in matched and traces.get(trace) == (
                    context["question_fingerprint"], gen, context["request_fingerprint"]
                ), "Complete matrix differs from physical log")
                matched.add(trace)
        for key in starts.keys() - terminals:
            group = "completed_tasks" if starts[key]["context"]["question_fingerprint"] in tasks else "interrupted_tasks"
            totals[group]["unresolved_starts"] += 1
    check(matched == traces.keys(), "Incomplete native traceback coverage")
    for group, counts in totals.items():
        check(counts["started"] == counts["returned"] + counts["errors"] + counts["unresolved_starts"], "Counts differ")
        if group == "completed_tasks":
            check(counts["errors"] == counts["unresolved_starts"] == 0, "Complete task has unresolved cost")
    if closure["status"] == "COMPLETE_UNSCORED":
        check("interrupted_tasks" not in totals and tasks, "Complete segment has interrupted or missing tasks")
    return {
        "archive_sha256": manifest_sha, "status": closure["status"], "completed_tasks": len(tasks),
        "complete_conditions": conditions, "matrix_requests_matched": len(matched),
        "actual_GPU_hours_including_interruption": closure["actual_GPU_hours"],
        "physical_journal_counts": {k: dict(v) for k, v in totals.items()},
        "token_coverage": "Returned generations only. Token counts for unresolved calls are unknown.",
        "hidden_scores_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive, args.manifest_sha), indent=2))
