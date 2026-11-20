"""Check restart admission with a local mirror of the private parent archive.

Run with the original private run directory as the sole argument. The directory
contains PLAN.json, DEPLOY_REQUEST.json, the transferred inputs and archive/.
No inference, grading, plan publication or file write takes place.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import resume_checkpoint
import resume_plan
import runtime


def changed_plan(plan, case):
    changed = copy.deepcopy(plan)
    worker = changed["assignments"]["0"]
    if case == "overlap":
        worker[0] = 5
    elif case == "duplicate":
        worker[1] = worker[0]
    elif case == "missing":
        worker.pop()
    elif case == "budget":
        changed["aggregate_reservation_GPU_hours"] = 6.1
    elif case == "parent":
        changed["resume_parent"]["archive_sha256"] = "f" * 64
    elif case == "seed":
        changed["seeds"][0] += 1
    elif case == "model":
        changed["models"]["qwen"]["identity"]["dtype"] = "float32"
    else:
        raise ValueError(case)
    return changed


def run(parent, manifest_sha):
    old = resume_checkpoint.load(parent / "PLAN.json")
    transfers = resume_checkpoint.load(parent / "DEPLOY_REQUEST.json")["uploads"]
    mapping = {item["remote"]: Path(item["local"]) for item in transfers}
    mapping[old["output"]] = parent / "archive"
    local_code = Path(__file__).resolve().parent
    remote_code = old["import_paths"][0].rsplit("/", 1)[0] + "/cycle3_function_resume_20261121"
    for filename in ("resume_checkpoint.py", "runtime.py"):
        mapping[remote_code + "/" + filename] = local_code / filename

    def local(path):
        return mapping.get(str(path), Path(path))

    def local_sha(path):
        return resume_checkpoint.digest(local(path))

    original_sha = resume_checkpoint.digest(parent / "PLAN.json")
    cases = ("overlap", "duplicate", "missing", "budget", "parent", "seed", "model")
    passed, rejected = 0, 0
    with patch.object(runtime, "sha", local_sha), patch.object(resume_checkpoint, "Path", local):
        for count in (1, 2, 4):
            plan = resume_plan.build_plan(
                old, range(6), manifest_sha, original_sha, remote_code,
                old["output"] + "_offline_example", local_code, list(range(count)), 6.0,
            )
            runtime.verify(plan)
            passed += 1
            for case in cases:
                try:
                    runtime.verify(changed_plan(plan, case))
                except (ValueError, AssertionError):
                    rejected += 1
                else:
                    raise RuntimeError(f"Incorrectly admitted case {case} for {count} workers")
    return {"valid_plans": passed, "invalid_plans_rejected": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("--manifest-sha", required=True, help="Trusted digest from the recorded pause")
    args = parser.parse_args()
    print(run(args.parent, args.manifest_sha))
