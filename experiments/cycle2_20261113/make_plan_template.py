"""Schema builder only. Caller supplies authorized label-free rows and frozen allocation.

No files are read on import. This does not select questions, inspect labels, or launch jobs.
Example: plan = make_plan(source_root, data_root, run_id, question_path,
    exclusion_path, model_path, model_identity, episodes, units)
Then independently review plan, set status='FROZEN', write exact bytes and pin SHA256.
"""
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import new_batch as nb


def make_plan(source_root, data_root, run_id, question_path, exclusion_path,
              model_path, model_identity, episodes, units, *, physical_gpus=(2, 4, 7),
              call_attempt_caps=(6000, 6000, 6000), graph=None):
    """episodes: [{episode,seed,permutation,complement}], units: see SCHEMA.md.

    question_path/exclusion_path/model_path are relative to data_root. data_root
    also owns results/cycle2/run_id. No labels are accepted into this API.
    """
    source_root, data_root = Path(source_root), Path(data_root)
    source_manifest = {p: nb.file_sha(nb.safe_path(source_root, p)) for p in nb.REQUIRED_SOURCES}
    module, _ = nb.load_operator(source_root, source_manifest)
    config = module.GraphConfig(**(graph or {}))
    eps = [{**e, "menu": nb.episode_menu(e["permutation"], e["complement"])} for e in episodes]
    questions = nb.read_jsonl(nb.safe_path(data_root, question_path))
    payloads = {q["id"]: module.digest(module.Executor.question_payload(q)) for q in questions}
    nb.require(len(payloads) == len(questions), "Duplicate question row")
    qmenu = nb.default_menu()
    qmenu += [{"name": "B_toggle_9_full_replay", "mask": qmenu[-1]["mask"],
               "base": None, "background": "B", "action": "replay", "replay_of": qmenu[-1]["name"]}]
    nb.require(len(physical_gpus) == len(call_attempt_caps) == 3, "Three declared workers required")
    return {
        "schema": nb.SCHEMA, "status": "DRAFT_REVIEW_REQUIRED", "run_id": run_id,
        "labels_read": False, "runner_sha256": nb.file_sha(nb.__file__),
        "source_manifest": source_manifest, "graph": dataclasses.asdict(config),
        "model": {"path": model_path, "identity": model_identity},
        "executor_protocol_hash": module.Executor(SimpleNamespace(identity=model_identity), config).protocol_hash,
        "questions": {"path": question_path, "sha256": nb.file_sha(nb.safe_path(data_root, question_path)),
                      "payload_sha256": payloads},
        "exclusions": {"path": exclusion_path, "sha256": nb.file_sha(nb.safe_path(data_root, exclusion_path))},
        "episodes": eps, "qualification_menu": qmenu, "units": units,
        "workers": [{"index": i, "physical_gpu": gpu, "resident_wall_cap_seconds": 8*3600,
                     "max_call_attempts": call_attempt_caps[i]} for i, gpu in enumerate(physical_gpus)],
        "budget": {"resident_wall_gpu_seconds": 24*3600, "max_call_attempts": sum(call_attempt_caps)},
        "max_unit_attempts": 2, "output_dir": "results/cycle2/"+run_id,
        "policy_information_firewall": "Raw menus unscored; root must lock policy before any S-label release",
    }
