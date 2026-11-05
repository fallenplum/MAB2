"""Independently replay D1 real input mapping, scores and decision arithmetic."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)


def correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        return float(spearmanr(left, right).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-id", default="d1_v2_20261105")
    args = parser.parse_args()
    assert re.fullmatch(r"d1(?:_v2)?_20261105", args.analysis_id)
    root = args.root.resolve(strict=True)
    base = root / "results/decision_diagnostics" / args.analysis_id
    old = root / "results/g2/g2_v1_20261104"
    audit = root / "artifacts/audit"
    output = audit / "D1_REAL_INPUT_AND_ARITHMETIC_REPLAY.json"
    assert not output.exists()
    attempt = uuid.uuid4().hex[:20]
    started, cpu = time.time(), time.process_time()
    report = {"schema": "graph_credit_d1_real_replay_v1", "status": "RUNNING",
              "analysis_id": args.analysis_id, "source_sha256": sha(Path(__file__).read_bytes()),
              "started_unix": started, "labels_read": False, "independent_confirmation": False,
              "scope": "Independent arithmetic and mapping on previously used development data"}
    write(audit / "d1_replay_attempts" / (attempt + ".intent.json"), report)
    try:
        plan = read(old / "frozen_plan.json")
        result = read(base / "DECISION_DIAGNOSTICS.json")
        recipe = read(base / "DIAGNOSTIC_PLAN.json")
        assert sha((base / "DIAGNOSTIC_PLAN.json").read_bytes()) == result["recipe_sha256"]
        for name, expected in recipe["input_anchors"].items():
            assert sha((root / name).read_bytes()) == expected
        for name, expected in result["source_pins"].items():
            assert sha((root / name).read_bytes()) == expected
        feature = plan["features"]
        assert sha((root / feature["path"]).read_bytes()) == feature["sha256"]
        raw_inputs = (base / "private/diagnostic_inputs.npz").read_bytes()
        assert sha(raw_inputs) == result["private_input_sha256"]
        import io
        with np.load(io.BytesIO(raw_inputs), allow_pickle=False) as data:
            fitted, observed, bits, leverage = [data[name].copy() for name in ["predictions", "observed_differences", "anchor_bits", "leverage"]]
        assert fitted.shape == (120, 2, 12, 3) and observed.shape == (120, 2, 12, 5)
        assert leverage.shape == (120, 2, 12, 2) and np.isfinite(leverage).all() and (leverage >= 0).all()
        auditor_path = root / "execution_tools/g2/independent_audit.py"
        auditor_sha = "10961c1761b06bc2035b56079b609c60d3239b01aac7189fddb4cf6e797f635d"
        assert sha(auditor_path.read_bytes()) == auditor_sha
        spec = importlib.util.spec_from_file_location("d1_independent_g2_scorer", auditor_path)
        auditor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = auditor
        spec.loader.exec_module(auditor)
        label_path = root / plan["label_release"]["test"]
        with label_path.open("rb") as handle:
            write(audit / "d1_replay_attempts" / (attempt + ".labels.json"),
                  {"event": "OPENED_BEFORE_READ", "opened_unix": time.time(), "historically_already_scored": True,
                   "source_sha256": report["source_sha256"], "result_sha256": sha((base / "DECISION_DIAGNOSTICS.json").read_bytes())})
            report["labels_read"] = True
            label_raw = handle.read()
        assert sha(label_raw) == read(old / "G2_ANALYSIS.json")["heldout_labels_sha256"]
        labels = {row["id"]: row for row in map(json.loads, label_raw.splitlines())}
        qids = sorted(qid for qid, part in plan["partitions"].items() if part == "test")
        assert set(qids) == set(labels) and len(qids) == 120
        families = sorted(result["families"])
        marker = read(old / "GLOBAL_OUTPUTS_FROZEN.json")
        descriptor = marker["artifacts"]["observations"]
        raw_observations = (root / descriptor["path"]).read_bytes()
        assert sha(raw_observations) == descriptor["sha256"]
        quality = {}
        for row in map(json.loads, raw_observations.splitlines()):
            if row["question_id"] in labels:
                key = (row["question_id"], row["seed"], tuple(row["graph_mask"]))
                assert key not in quality
                quality[key] = auditor.independent_score(row["service_output"], labels[row["question_id"]])
        prediction_rows = read(old / "heldout_predictions.json")
        predictions = {(row["question_id"], row["anchor_id"], row["edge"]): row["predictions"] for row in prediction_rows}
        for q, qid in enumerate(qids):
            contrasts = {(row["anchor_id"], row["edge"]): row for row in plan["designs"][qid]["contrasts"]}
            anchors = {row["anchor_id"]: row["mask"] for row in plan["designs"][qid]["anchors"]}
            for a, family in enumerate(families):
                np.testing.assert_array_equal(bits[q, a], anchors[family])
                for edge in range(12):
                    row = contrasts[(family, edge)]
                    left, right = tuple(plan["masks"][row["mask0_id"]]), tuple(plan["masks"][row["mask1_id"]])
                    targets = [quality[(qid, seed, right)] - quality[(qid, seed, left)] for seed in range(2701, 2706)]
                    np.testing.assert_array_equal(observed[q, a, edge], targets)
                    np.testing.assert_array_equal(fitted[q, a, edge], [predictions[(qid, family, edge)][name] for name in ["B0", "C1", "C2"]])
        names = ["B0", "C1", "C2", "ZERO_STAY", "UNIFORM_13_ACTIONS"]
        blocks = {name: [] for name in names}
        action_counts = {name: [0] * 13 for name in names[:3]}
        for q in range(120):
            anchor_gains = {name: [] for name in names}
            for a in range(2):
                outcomes = [(1 - 2 * int(bits[q, a, edge])) * math.fsum(observed[q, a, edge]) / 5 for edge in range(12)] + [0.]
                for column, name in enumerate(names[:3]):
                    candidates = [((1 - 2 * int(bits[q, a, edge])) * fitted[q, a, edge, column], edge) for edge in range(12)]
                    positive = sorted((item for item in candidates if item[0] > 0), key=lambda item: (-item[0], item[1]))
                    action = positive[0][1] if positive else 12
                    action_counts[name][action] += 1
                    anchor_gains[name].append(outcomes[action])
                anchor_gains["ZERO_STAY"].append(0.)
                anchor_gains["UNIFORM_13_ACTIONS"].append(math.fsum(outcomes) / 13)
            for name in names:
                blocks[name].append(math.fsum(anchor_gains[name]) / 2)
        decision = result["decision"]
        maximum_error = 0.
        def check(actual, expected):
            nonlocal maximum_error
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            maximum_error = max(maximum_error, error)
            assert error <= 1e-12
        for name in names:
            check(math.fsum(blocks[name]) / 120, decision["policies"][name]["mean_observed_gain"])
            if name in action_counts:
                assert action_counts[name] == decision["policies"][name]["selection_support"]["action_counts_edge0_through_edge11_then_stay"]
        indices = np.random.default_rng(49173).integers(0, 120, size=(10000, 120))
        for name, value in decision["paired_comparisons"].items():
            left, right = name.split("_minus_", 1)
            differences = np.asarray(blocks[left]) - np.asarray(blocks[right])
            check(math.fsum(differences) / 120, value["estimate"])
            check(np.quantile(differences[indices].mean(axis=1), [.025, .975], method="linear"), value["interval"])
        means = observed.mean(axis=-1)
        for column, name in enumerate(names[:3]):
            ranks = [correlation(fitted[q, a, :, column], means[q, a]) for q in range(120) for a in range(2)]
            valid = [value for value in ranks if value is not None]
            expected = decision["spearman"]["models"][name]["overall"]
            assert len(valid) == expected["defined_pairs"] and len(ranks) - len(valid) == expected["undefined_pairs"]
            check(math.fsum(valid) / len(valid), expected["mean_defined_spearman"])
        subset = means != 0
        for column, name in enumerate(["C1", "C2"]):
            model = result["uncertainty"]["models"][name]
            wrong = np.sign(fitted[..., column + 1][subset]) != np.sign(means[subset])
            proxies = [leverage[..., column][subset], -np.abs(fitted[..., column + 1][subset]) / np.maximum(leverage[..., column][subset], 1e-12)]
            for scores, key in zip(proxies, ["leverage_roc_auc", "negative_normalized_margin_roc_auc"]):
                wins = scores[wrong, None] - scores[~wrong][None, :]
                check(float((wins > 0).mean() + .5 * (wins == 0).mean()), model["observed_nonzero_mean_sign_diagnostic"][key])
        report.update(status="PASS", original_scores_replayed=len(quality), coordinates_exactly_mapped=2880,
                      predictions_exactly_mapped=8640, observed_seed_differences_exactly_mapped=14400,
                      policy_means_replayed=5, paired_bootstrap_comparisons_replayed=8,
                      spearman_models_replayed=3, uncertainty_auc_replayed=4,
                      maximum_arithmetic_absolute_difference=maximum_error, leverage_geometry_recomputed=False,
                      feature_bytes_sha256=feature["sha256"], original_independent_scorer_sha256=auditor_sha,
                      diagnostic_output_sha256=sha((base / "DECISION_DIAGNOSTICS.json").read_bytes()))
    except BaseException as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(completed_unix=time.time(), cpu_seconds=time.process_time() - cpu,
                      wall_seconds=time.time() - started, timing_scope="This replay process after imports")
        write(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
