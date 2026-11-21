"""Aggregate both segments and engineering runs without imputing missing returns."""
from collections import defaultdict, Counter
import json
import math


def load(path):
    return json.loads(path.read_bytes())


def summarize_costs(archives):
    models = defaultdict(Counter)
    trace_seconds, load_seconds, placement_seconds = 0.0, 0.0, 0.0
    total_residence = 0.0
    segments = []
    for name, archive in archives:
        c = load(archive/"CLOSURE.json")
        assert c["all_workers_exited"]
        assert math.isclose(c["actual_GPU_hours"],sum(c["residence_seconds_by_worker"])/3600,abs_tol=1e-9)
        total_residence += c["actual_GPU_hours"]
        segments.append({"segment":name,"status":c["status"],"actual_GPU_hours":c["actual_GPU_hours"]})
        for journal in archive.glob("worker_*/*_requests.jsonl"):
            counts=models[journal.name.removesuffix("_requests.jsonl")]
            for line in journal.read_bytes().splitlines():
                row=json.loads(line);event=row["event"]
                counts[event]+=1
                if event!="RETURNED":continue
                gen=row["generation"]
                for field in ("input_tokens","output_tokens","gpu_seconds","elapsed_seconds"):
                    assert math.isfinite(gen[field]) and gen[field]>=0
                    counts[field]+=gen[field]
        load_seconds+=sum(load(p)["seconds"] for p in archive.glob("worker_*/LOAD_*.json"))
        for journal in archive.glob("worker_*/PLACEMENT.jsonl"):
            placement_seconds+=sum(json.loads(line)["seconds"] for line in journal.read_bytes().splitlines())
        for p in archive.glob("worker_*/unit_*/MATRIX.json"):
            trace_seconds+=load(p)["physical_visible_execution_seconds"]
    for value in models.values():
        value["unresolved_calls"]=value["START"]-value["RETURNED"]-value["ERROR"]
        assert value["unresolved_calls"]>=0
    return dict(segments=segments,actual_GPU_hours=total_residence,model_physical_journals=dict(models),
                loading_worker_seconds=load_seconds,placement_worker_seconds=placement_seconds,
                complete_matrix_visible_execution_seconds=trace_seconds,
                incomplete_matrix_visible_seconds_and_unreturned_tokens_are_unknown=True)
