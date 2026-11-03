"""Real label-free deserialize/cache-closure equivalence test for elastic G1."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-label", required=True, type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    private = root / "artifacts/private/performance"
    for name in ("elastic_worker", "cross_gpu_audit"):
        spec = importlib.util.spec_from_file_location(name, private / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    worker, auditor = sys.modules["elastic_worker"], sys.modules["cross_gpu_audit"]
    module, backend_class, pilot = worker.load_frozen(root)
    plan = read(root / "results/pilot/motivation_v3_20261103/shard_000/frozen_plan.json")
    folder = root / "results/pilot" / f"elastic_resume_gpu{args.gpu_label}_20261103"
    folder.mkdir(exist_ok=False)
    reference = root / "results/pilot/s0_v3_20261103/qualification_executions"
    first = read(reference / "r2_full_first.json")
    expected = read(reference / "r2_switch_closure.json")
    question = json.loads((root / "data/derived/qualification_questions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if pilot.source_manifest(root) != plan["source_manifest"]:
        raise ValueError("Frozen source changed")
    backend = backend_class(root / "models/Qwen2.5-3B-Instruct", "cuda:0", folder / "physical_calls.jsonl")
    if backend.identity != plan["model_identity"]:
        raise ValueError("Model/runtime identity changed")
    executor = module.Executor(backend, worker.native_config(plan, module))
    base = worker.deserialize_execution(first, module)
    worker.validate_base(base, executor, question, first["mask"], first["seed"], module)
    result = executor.execute(question, expected["mask"], expected["seed"], base=base)
    actual = result.to_dict()
    (folder / "resumed_execution.json").write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
    # The immutable experimental interface is persisted JSON.  Comparing the
    # in-memory dataclass tuple mask to a JSON list creates a false mismatch.
    persisted = read(folder / "resumed_execution.json")
    exact = auditor.signature(persisted) == auditor.signature(expected)
    calls = len(backend.physical_calls)
    receipt = {"schema": "elastic_real_resume_qualification_v1", "gpu": args.gpu_label,
               "status": "PASS" if exact and calls == 2 else "FAIL", "exact_node_signature_equal": exact,
               "comparison_domain": "persisted_JSON_node_inputs_messages_outputs_and_seed",
               "physical_calls": calls, "expected_closure_calls": 2, "labels_read": False,
               "worker_source_sha256": hashlib.sha256((private / "elastic_worker.py").read_bytes()).hexdigest(),
               "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "original_frozen_plan_sha256": hashlib.sha256((root / "results/pilot/motivation_v3_20261103/shard_000/frozen_plan.json").read_bytes()).hexdigest(),
               "protocol_hash": executor.protocol_hash, "created_unix": time.time(),
               "physical_cost": result.cost(), "scientific_completion": False}
    (folder / "QUALIFICATION.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
