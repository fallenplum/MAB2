"""Assemble a scorer view from two immutable archives; retain actual exit codes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from resume_checkpoint import digest, load, require, verified_units
from segment_cost_audit import summarize


def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check_join(closure):
    groups = closure["source_segments"]
    require(closure["status"] == "COMPLETE_UNSCORED" and closure["all_workers_exited"], "Collection incomplete")
    require([g["status"] for g in groups] == ["PARTIAL_OR_FAILED", "COMPLETE_UNSCORED"], "Unexpected segment state")
    require([g["indices"] for g in groups] == [list(range(6)), list(range(6, 16))], "Wrong joint coverage")
    require(groups[0]["exit_codes"] == [-15, -15] and groups[1]["exit_codes"] == [0] * 4, "Unexpected source exit codes")
    require(closure["exit_codes"] == groups[0]["exit_codes"] + groups[1]["exit_codes"], "Source exits changed")


def assemble(old, new, old_plan_file, new_plan_file, inputs, selection, old_sha, new_sha, output):
    old_plan, plan = load(old_plan_file), load(new_plan_file)
    done = verified_units(old, old_sha, old_plan_file, inputs, selection)
    require(set(done) == set(range(6)), "Parent coverage differs")
    require(plan["resume_parent"]["archive_sha256"] == old_sha
            and plan["resume_parent"]["plan_sha256"] == digest(old_plan_file), "Parent provenance differs")
    for key in ("seeds", "models", "questions_sha256", "selection_sha256", "qualified_operator_protocol_sha256"):
        require(plan[key] == old_plan[key], "Frozen plan changed")
    rows = [json.loads(line) for line in inputs.read_bytes().splitlines()]
    selected = load(selection)["questions"]
    work = output.with_name(output.name + ".building")
    work.mkdir()
    groups, residences, exits, task_ids = [], [], [], set()
    conditions = {"G0","G1","G0_MINUS_EU","G0_MINUS_EV","G1_MINUS_EU","G1_MINUS_EV","SINGLE_Q","SINGLE_C","BOK3","REPAIR3"}
    for segment, (src, pin, plan_file, p, offset) in enumerate([
        (old, old_sha, old_plan_file, old_plan, 0), (new, new_sha, new_plan_file, plan, len(old_plan["gpus"])),
    ]):
        cost = summarize(src, pin)
        c = load(src / "CLOSURE.json")
        require(c["plan_sha256"] == digest(plan_file), "Segment plan differs")
        if segment == 1:
            require(c["status"] == "COMPLETE_UNSCORED" and len(c["exit_codes"]) == 4
                    and all(v == 0 for v in c["exit_codes"]), "Resume did not finish normally")
        indices = []
        for marker in sorted(src.glob("worker_*/unit_*/COMPLETE.json")):
            unit = marker.parent
            complete = load(marker)
            i = complete["index"]
            worker = unit.parent.name.removeprefix("worker_")
            require(type(i) is int and 0 <= i < 16 and i in p["assignments"][worker], "Unexpected unit")
            require(i not in task_ids and unit.name == f"unit_{i:03d}", "Duplicated or misplaced unit")
            require(load(unit / "INPUT.json") == {"index":i, "payload":rows[i], "shock":selected[i]["removed_upstream"]}, "Input drift")
            payload = canonical(rows[i])
            require(hashlib.sha256(payload + b"\n").hexdigest() == selected[i]["input_sha256"], "Selection drift")
            m = load(unit / "MATRIX.json")
            require(m["question_fingerprint"] == hashlib.sha256(payload).hexdigest()
                    and m["protocol_sha256"] == p["qualified_operator_protocol_sha256"]
                    and m["execution_seeds"] == p["seeds"] and not m["hidden_test_access"], "Matrix identity drift")
            require(len(m["conditions"]) == complete["terminal_conditions"] == 30 and
                    {(r["execution_seed"], r["condition"]) for r in m["conditions"] if r["status"] == "COMPLETE"}
                    == {(s, k) for s in p["seeds"] for k in conditions}, "Conditions incomplete")
            dest = work / f"worker_{offset+int(worker)}" / unit.name
            dest.mkdir(parents=True)
            for name in ("MATRIX.json","INPUT.json","COMPLETE.json"):
                shutil.copyfile(unit / name, dest / name)
                require(digest(dest / name) == digest(unit / name), "Scorer input copy differs")
            indices.append(i)
            task_ids.add(i)
        groups.append({"status":c["status"], "indices":sorted(indices), "exit_codes":c["exit_codes"],
                       "manifest_sha256":pin, "physical_cost":cost})
        residences.extend(c["residence_seconds_by_worker"])
        exits.extend(c["exit_codes"])
    closure = {"status":"COMPLETE_UNSCORED","all_workers_exited":True,"exit_codes":exits,
               "source_segments":groups,"residence_seconds_by_worker":residences,
               "actual_GPU_hours":sum(residences)/3600,"quality_seen":False,
               "kind":"verified_two_segment_scorer_view"}
    check_join(closure)
    dump(work / "CLOSURE.json", closure)
    dump(work / "ARCHIVE_MANIFEST.json", {f.relative_to(work).as_posix():digest(f) for f in work.rglob("*") if f.is_file()})
    require(not output.exists(), "Scorer view already exists")
    os.rename(work, output)
