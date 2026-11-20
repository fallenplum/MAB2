"""Build a Formal10 continuation plan from a verified partial collection."""
from copy import deepcopy
from pathlib import Path
import math
from resume_checkpoint import verify_assignments, require, digest

def build_plan(old, completed, parent_sha, original_plan_sha, restart_dir, restart_output, sources, gpus, reserved_hours):
    count = len(gpus)
    remaining = sorted(set(range(16)) - set(completed))
    require(count in (1, 2, 4) and len(set(gpus)) == count and all(type(i) is int and i in range(8) for i in gpus), "Invalid GPUs")
    require(0 < reserved_hours <= 6.0, "Resume exceeds six GPU hours")
    assignments = {str(i): remaining[i::count] for i in range(count)}
    verify_assignments(remaining, assignments, count)
    plan = deepcopy(old)
    plan.update(output=restart_output, gpus=gpus, worker_cap_seconds=math.floor(reserved_hours*3600/count)-60,
                aggregate_reservation_GPU_hours=reserved_hours, assignments=assignments,
                resume_parent={"archive":old["output"],"archive_sha256":parent_sha,
                               "plan":old["import_paths"][0]+"/PLAN.json","plan_sha256":original_plan_sha})
    for name in ("runtime.py","resume_checkpoint.py"):
        plan["source_pins"][restart_dir+"/"+name] = digest(Path(sources)/name)
    return plan
