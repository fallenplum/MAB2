"""Post-hoc decision diagnostics on the already evaluated G2 development data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

from decision_diagnostics import decision_diagnostics
from decision_uncertainty import uncertainty_diagnostics


BASE = "results/decision_diagnostics/d1_v2_20261105"
OLD = "results/g2/g2_v1_20261104"
PLAN = OLD + "/frozen_plan.json"
ACCEPTANCE = "artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json"
SOURCES = {
    "execution_tools/g2/descriptive_appendix.py": "3c6eb81fd21cd320b218c814bc4e369cbaf1fcd854abf328c4e84c51260c14d1",
    "experiments/g2/predictor.py": "d6d93dc938ce7cff682a5457c59b779bed5668999619e5afb614773f756e8e03",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recipe(root, context, helper):
    pins = {name: sha((root / name).read_bytes()) for name in SOURCES}
    assert pins == SOURCES
    for name in ["run_existing_g2_decisions.py", "decision_diagnostics.py", "decision_uncertainty.py"]:
        path = Path(__file__).parent / name
        pins[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    feature = context["plan"]["features"]
    assert sha((root / feature["path"]).read_bytes()) == feature["sha256"]
    context["anchors"][feature["path"]] = feature["sha256"]
    return {
        "schema": "graph_credit_decision_diagnostic_plan_v1",
        "analysis_id": "d1_v2_20261105",
        "engineering_revision": "Feature bytes bound before loading and checked again before publication",
        "prior_exploratory_attempt_preserved": "results/decision_diagnostics/d1_20261105",
        "status": "EXPLORATORY_POST_G2_RESULTS_KNOWN",
        "source_pins": pins,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "input_anchors": context["anchors"],
        "prior_G2_gate": context["acceptance"]["scientific_gate"],
        "old_G2_gate_changed": False,
        "old_120_question_role": "Previously scored heldout, now used for post-hoc exploration",
        "independent_confirmation": False,
        "question_count": 120,
        "models": ["B0", "C1", "C2"],
        "model_selection_or_retraining": False,
        "threshold": 0.0,
        "quality_cost_lambda": 0.0,
        "action_set": "stay plus twelve individual toggles at each of two frozen anchors",
        "action_tie_break": "lowest edge index among strictly positive predicted maxima, otherwise stay",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 49173,
        "resampling_unit": "whole_question_after_equal_mean_over_two_anchors",
        "multiplicity_adjustment": "none",
        "sign_and_rank_target": "five-generation-seed empirical existence-credit mean, a noisy proxy",
        "uncertainty_proxy": "sqrt(d^T (X^T W X + alpha I)^-1 d); no residual-noise scaling or coverage guarantee",
        "uncertainty_error_target": "observed nonzero-mean sign mismatch; descriptive and noise-contaminated",
        "original_scorer_sha256": context["scorer_sha256"],
        "full_budget_or_online_benefit_claimed": False,
        "multi_edit_or_recovery_trajectory_evaluated": False,
        "G3_permission": "Pending a separately frozen independent confirmation and mechanism protocol",
    }


def prepare_arrays(context, coordinates):
    plan = context["plan"]
    qids = sorted(qid for qid, part in plan["partitions"].items() if part == "test")
    families = sorted(context["families"])
    positions = {(qid, family, edge): (q, a, edge) for q, qid in enumerate(qids)
                 for a, family in enumerate(families) for edge in range(12)}
    predictions = np.full((120, 2, 12, 3), np.nan)
    targets = np.full((120, 2, 12, 5), np.nan)
    bits = np.empty((120, 2, 12), dtype=np.int8)
    for row in context["predictions"]:
        index = positions[(row["question_id"], row["anchor_id"], row["edge"])]
        predictions[index] = [row["predictions"][name] for name in ["B0", "C1", "C2"]]
    for row in coordinates:
        targets[positions[(row["qid"], row["family"], row["edge"])]] = row["targets"]
    for q, qid in enumerate(qids):
        anchors = {row["anchor_id"]: row["mask"] for row in plan["designs"][qid]["anchors"]}
        for a, family in enumerate(families):
            bits[q, a] = anchors[family]
    assert np.isfinite(predictions).all() and np.isfinite(targets).all()
    return qids, families, predictions, targets, bits


def ridge_direction_leverage(matrix, weights, directions, alpha):
    matrix, weights, directions = (np.asarray(value, dtype=np.float64) for value in [matrix, weights, directions])
    assert matrix.ndim == directions.ndim == 2 and directions.shape[1] == matrix.shape[1]
    assert weights.shape == (len(matrix),) and np.all(weights > 0) and alpha > 0
    assert np.isfinite(matrix).all() and np.isfinite(weights).all() and np.isfinite(directions).all()
    weighted = matrix * np.sqrt(weights)[:, None]
    gram = weighted.T @ weighted + alpha * np.eye(matrix.shape[1])
    solved = cho_solve(cho_factor(gram, lower=True), directions.T).T
    variances = np.sum(directions * solved, axis=1)
    assert np.isfinite(variances).all() and np.all(variances >= 0)
    return np.sqrt(variances)


def decode_frozen_features(raw, expected_sha256):
    assert sha(raw) == expected_sha256, "PCA feature bytes differ from the accepted input"
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {str(qid): value for qid, value in zip(data["question_ids"].tolist(), data["phi"])}


def leverage_arrays(root, context, qids, families, predictor):
    plan = context["plan"]
    parameters = json.loads((root / (OLD + "/parameters.json")).read_text(encoding="utf-8"))
    feature = plan["features"]
    phi = decode_frozen_features((root / feature["path"]).read_bytes(), feature["sha256"])
    keys, weights = [], []
    for qid in sorted(parameters["train_question_ids"]):
        masks = sorted({tuple(row["mask"]) for row in plan["designs"][qid]["mask_requests"]})
        assert len(masks) == parameters["unique_training_masks_by_question"][qid]
        keys.extend((qid, mask) for mask in masks)
        weights.extend([1 / len(masks)] * len(masks))
    assert len(keys) == parameters["train_unique_graph_rows"] == 5194
    contrasts = {(row["question_id"], row["anchor_id"], row["edge"]): row
                 for row in context["predictions"]}
    result = np.empty((120, 2, 12, 2))
    for column, name in enumerate(["C1", "C2"]):
        model = parameters["models"][name]
        matrix = np.stack([predictor.feature_vector(phi[qid], mask, model["order"]) for qid, mask in keys])
        assert matrix.shape[1] == model["dimensions"]
        directions = []
        for qid in qids:
            for family in families:
                for edge in range(12):
                    row = contrasts[(qid, family, edge)]
                    directions.append(predictor.feature_vector(phi[qid], row["mask1"], model["order"])
                                      - predictor.feature_vector(phi[qid], row["mask0"], model["order"]))
        directions = np.stack(directions)
        result[..., column] = ridge_direction_leverage(matrix, weights, directions, model["alpha"]).reshape(120, 2, 12)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["freeze", "run"], required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    for name, expected in SOURCES.items():
        assert sha((root / name).read_bytes()) == expected
    helper = load_module(root / "execution_tools/g2/descriptive_appendix.py", "decision_existing_g2_helpers")
    workspace = helper.Workspace(root)
    with workspace.lease(BASE + "/diagnostic.lock"):
        context = helper.source_context(workspace, PLAN)
        helper.prelabel_gate(workspace, context, ACCEPTANCE)
        expected_recipe = recipe(root, context, helper)
        plan_path = BASE + "/DIAGNOSTIC_PLAN.json"
        if args.mode == "freeze":
            workspace.write(plan_path, expected_recipe, exclusive=True)
            print(json.dumps({"status": "EXPLORATORY_RECIPE_FIXED_BEFORE_NEW_METRICS", "sha256": sha(workspace.read(plan_path))}))
            return 0
        assert workspace.json(plan_path) == expected_recipe
        assert not workspace.path(BASE + "/DECISION_DIAGNOSTICS.json").exists()
        started, cpu = time.time(), time.process_time()
        attempt_name = BASE + "/attempts/" + uuid.uuid4().hex[:20] + ".json"
        attempt = {"schema": "graph_credit_decision_diagnostic_attempt_v1", "status": "STARTED", "started_unix": started,
                   "recipe_sha256": sha(workspace.read(plan_path)), "labels_read": False}
        workspace.write(attempt_name, attempt, exclusive=True)
        try:
            label_path = context["plan"]["label_release"]["test"]
            with workspace.path(label_path).open("rb") as handle:
                event = {"schema": "graph_credit_posthoc_label_access_v1", "event": "OPENED_BEFORE_READ",
                         "source_sha256": sha(Path(__file__).read_bytes()), "recipe_sha256": attempt["recipe_sha256"],
                         "label_path": label_path, "expected_sha256": context["expected_labels_sha256"],
                         "opened_unix": time.time(), "historically_already_scored": True}
                workspace.write(BASE + "/DIAGNOSTIC_LABEL_ACCESS.json", event, exclusive=True)
                attempt["labels_read"] = True
                raw = handle.read()
            assert sha(raw) == context["expected_labels_sha256"]
            labels = {row["id"]: row for row in [helper.decode(line) for line in raw.splitlines()]}
            assert len(labels) == 120
            coordinates = helper.scored_coordinates(context, labels, helper.original_scorer(workspace, context))
            qids, families, predictions, targets, bits = prepare_arrays(context, coordinates)
            predictor = load_module(root / "experiments/g2/predictor.py", "decision_existing_g2_predictor")
            leverage = leverage_arrays(root, context, qids, families, predictor)
            result = decision_diagnostics(predictions, targets, bits)
            uncertainty = uncertainty_diagnostics(predictions, targets, leverage)
            private_path = workspace.path(BASE + "/private/diagnostic_inputs.npz")
            private_path.parent.mkdir(parents=True, exist_ok=True)
            with private_path.open("xb") as handle:
                np.savez_compressed(handle, predictions=predictions, observed_differences=targets, anchor_bits=bits, leverage=leverage)
            for path, expected in context["anchors"].items():
                assert sha(workspace.read(path)) == expected
            report = {"schema": "graph_credit_posthoc_decision_diagnostics_v1", "status": "EXPLORATORY_COMPLETE",
                      "recipe_sha256": attempt["recipe_sha256"], "source_pins": expected_recipe["source_pins"],
                      "prior_acceptance_sha256": context["acceptance_sha256"], "prior_G2_gate": context["acceptance"]["scientific_gate"],
                      "old_G2_gate_changed": False, "independent_confirmation": False, "model_selected": False,
                      "G3_started": False, "contains_individual_identifiers_or_label_values": False,
                      "private_input_sha256": sha(private_path.read_bytes()), "families": families,
                      "decision": result, "uncertainty": uncertainty, "attempt_path": attempt_name,
                      "cpu_seconds_before_publication": time.process_time() - cpu, "completed_unix": time.time()}
            workspace.write(BASE + "/DECISION_DIAGNOSTICS.json", report, exclusive=True)
            attempt.update(status="COMPLETE", output_sha256=sha(workspace.read(BASE + "/DECISION_DIAGNOSTICS.json")))
            print(json.dumps({"status": report["status"], "output_sha256": attempt["output_sha256"], "old_G2_gate_changed": False,
                              "independent_confirmation": False, "G3_started": False}), flush=True)
        except BaseException as error:
            attempt.update(status="FAILED", error_type=type(error).__name__)
            raise
        finally:
            attempt.update(completed_unix=time.time(), cpu_seconds=time.process_time() - cpu, wall_seconds=time.time() - started)
            workspace.write(attempt_name, attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
