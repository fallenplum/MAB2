"""Pre-frozen, post-acceptance descriptive G2 appendix without changing its gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np


SEEDS = (2701, 2702, 2703, 2704, 2705)
MODELS = ("ZERO", "B0", "C1", "C2")
REFERENCES = ("ZERO", "B0", "C1")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 39173
HELPER_PATH = "artifacts/private/monitoring/mirror_only.py"
HELPER_SHA = "273008dfafa95fe67f522fd116024c50f2176298072f6f4b95cf6267dc7e5ccb"
GOLD_EVENTS = ("LOCAL_DATA_OWNER_FIRST_LABEL_READ.json", "LOCAL_LABEL_SPLIT_FROZEN.json",
               "TRAIN_TUNE_LABEL_RELEASE.json", "FIRST_TRAIN_TUNE_LABELS_OPENED.json",
               "HELDOUT_LABEL_RELEASE.json", "FIRST_HELDOUT_LABELS_OPENED.json")


class IntegrityError(RuntimeError):
    pass


class WaitingError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def decode(raw):
    def unique(pairs):
        value = {}
        for name, item in pairs:
            require(name not in value, "Duplicate JSON field")
            value[name] = item
        return value
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        helper = self.root.joinpath(*HELPER_PATH.split("/"))
        require(bytes_sha(helper.read_bytes()) == HELPER_SHA, "Safe publication helper pin changed")
        spec = importlib.util.spec_from_file_location("g2_appendix_safe_publication", helper)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        info = self.root.lstat()
        self.identity = (info.st_dev, info.st_ino)

    def path(self, name):
        return self.helper.local_path(self.root, name, root_identity=self.identity)

    def read(self, name):
        return self.path(name).read_bytes()

    def json(self, name):
        return decode(self.read(name))

    def write(self, name, value, *, exclusive=False):
        raw = (canonical(value) + "\n").encode("utf-8")
        target = self.path(name)
        if exclusive and target.exists():
            require(target.read_bytes() == raw, "Immutable descriptive artifact already differs")
            return bytes_sha(raw)
        self.helper.atomic_bytes(target, raw, root=self.root, root_identity=self.identity)
        return bytes_sha(raw)

    @contextmanager
    def lease(self, name):
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path(name)
        with self.helper.PublisherLock(path):
            yield


def source_context(workspace, plan_path):
    raw = workspace.read(plan_path)
    plan = decode(raw)
    require(plan.get("schema") == "graph_credit_g2_plan_v1" and plan.get("status") == "PLAN_FROZEN"
            and plan.get("labels_read") is False and plan["expected_units"] == 2000 and plan["expected_outputs"] == 51970,
            "The canonical frozen G2 plan is required")
    require(plan_path == plan["output_dir"] + "/frozen_plan.json", "Unexpected G2 plan path")
    for manifest in (plan["source_manifest"], plan["pipeline_source_manifest"]):
        for name, item in manifest.items():
            require(bytes_sha(workspace.read(name)) == item["sha256"], "A frozen original scientific source changed")
    scorers = [name for name in plan["source_manifest"] if Path(name).name == "scoring.py"]
    require(len(scorers) == 1, "Exactly one original frozen scorer is required")
    families = sorted({row["anchor_id"] for design in plan["designs"].values() for row in design["contrasts"]})
    require(len(families) == 2 and Counter(plan["partitions"].values()) == {"train": 200, "tune": 80, "test": 120},
            "Unexpected frozen family/partition support")
    return {"plan": plan, "plan_sha256": bytes_sha(raw), "plan_path": plan_path,
            "base": plan["output_dir"] + "/descriptive_appendix", "source_manifest_sha256": digest(plan["source_manifest"]),
            "pipeline_source_manifest_sha256": digest(plan["pipeline_source_manifest"]), "scorer_path": scorers[0],
            "scorer_sha256": plan["source_manifest"][scorers[0]]["sha256"], "families": families}


def recipe_value(context):
    numpy_core = getattr(np, "_core", np.core)
    dependency_hashes = {"numpy_init": bytes_sha(Path(np.__file__).read_bytes()),
                         "numpy_multiarray_extension": bytes_sha(Path(numpy_core._multiarray_umath.__file__).read_bytes())}
    return {"schema": "graph_credit_g2_descriptive_recipe_v1", "state": "DESCRIPTIVE_CODE_FROZEN",
            "run_id": context["plan"]["run_id"], "plan_sha256": context["plan_sha256"],
            "source_manifest_sha256": context["source_manifest_sha256"],
            "pipeline_source_manifest_sha256": context["pipeline_source_manifest_sha256"],
            "protocol_hash": context["plan"]["protocol_hash"], "source_sha256": bytes_sha(Path(__file__).read_bytes()),
            "scorer_path": context["scorer_path"], "scorer_sha256": context["scorer_sha256"], "helper_sha256": HELPER_SHA,
            "python": platform.python_version(), "numpy": np.__version__, "seeds": list(SEEDS), "models": list(MODELS),
            "numeric_dependency_sha256": dependency_hashes,
            "families": context["families"], "edges": list(range(12)), "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED, "confidence": .95, "resampling_unit": "whole_question",
            "comparisons": [name + "_risk_minus_C2_risk" for name in REFERENCES],
            "observation_strata": ["all_five_observed_differences_zero", "any_observed_difference_nonzero"],
            "stratum_estimator": "coordinate-weighted ratio after whole-question resampling",
            "conditional_true_risk_unbiasedness_claimed": False, "multiplicity_adjustment": "none",
            "inference_scope": "descriptive exploratory appendix", "primary_gate_modified": False,
            "aggregate_only": True, "labels_read": False}


def freeze_recipe(workspace, context):
    name = context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json"
    expected = recipe_value(context)
    if workspace.path(name).exists():
        prior = workspace.json(name)
        require({key: value for key, value in prior.items() if key != "created_unix"} == expected
                and finite(prior.get("created_unix")), "Existing descriptive source freeze differs")
        return prior
    require(not any(workspace.path(context["plan"]["output_dir"] + "/" + event).exists() for event in GOLD_EVENTS),
            "Descriptive source must be frozen before any recorded G2 label release or access")
    require(not workspace.path(context["base"] + "/DESCRIPTIVE_FIRST_LABEL_ACCESS.json").exists(),
            "Cannot replace a missing descriptive recipe after descriptive gold access")
    value = {**expected, "created_unix": time.time()}
    workspace.write(name, value, exclusive=True)
    return value


def check_hash(workspace, name, expected):
    raw = workspace.read(name)
    require(bytes_sha(raw) == expected, "A frozen descriptive input byte SHA changed")
    return raw


def prelabel_gate(workspace, context, acceptance_path):
    recipe_path = context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json"
    recipe_raw = workspace.read(recipe_path)
    recipe = decode(recipe_raw)
    require({key: value for key, value in recipe.items() if key != "created_unix"} == recipe_value(context)
            and finite(recipe.get("created_unix")), "Descriptive script/dependency freeze no longer matches")
    if not workspace.path(acceptance_path).exists():
        raise WaitingError("Independent G2 acceptance has not been published")
    acceptance_raw = workspace.read(acceptance_path)
    acceptance = decode(acceptance_raw)
    if acceptance.get("status") == "WAITING_NOT_TERMINAL":
        raise WaitingError("Independent G2 acceptance is not terminal")
    require(acceptance.get("schema") == "graph_credit_g2_independent_acceptance_v1"
            and acceptance.get("status") == "ACCEPTED_WITH_DOCUMENTED_LIMITATIONS"
            and acceptance.get("g2_stage_complete") is True and acceptance.get("scientific_completion") is True
            and acceptance.get("project_complete") is False and acceptance.get("run_id") == context["plan"]["run_id"]
            and acceptance.get("plan_sha256") == context["plan_sha256"], "Independent G2 acceptance is incomplete or foreign")
    require(all(acceptance.get(name, {}).get("status") == "PASS" for name in
                ("integrity", "score_replay", "ridge_replay", "prediction_replay", "statistics", "costs")),
            "An independent acceptance component has not passed")
    base, plan = context["plan"]["output_dir"], context["plan"]
    anchors = {context["plan_path"]: context["plan_sha256"], acceptance_path: bytes_sha(acceptance_raw), recipe_path: bytes_sha(recipe_raw)}
    values = {}
    for filename, key in (("GLOBAL_OUTPUTS_FROZEN.json", "global_outputs_frozen"),
                          ("PREDICTIONS_FROZEN.json", "predictions_frozen"), ("G2_ANALYSIS.json", "analysis"),
                          ("LOCAL_MIRROR_COMPLETE.json", "local_mirror_complete")):
        name, expected = base + "/" + filename, acceptance["artifact_sha256"][key]
        values[key] = decode(check_hash(workspace, name, expected))
        anchors[name] = expected
    global_marker, prediction, analysis = values["global_outputs_frozen"], values["predictions_frozen"], values["analysis"]
    require(global_marker.get("schema") == "graph_credit_g2_outputs_frozen_v1"
            and prediction.get("schema") == "graph_credit_g2_predictions_frozen_v1"
            and analysis.get("schema") == "graph_credit_g2_heldout_evaluation_v1"
            and global_marker["plan_sha256"] == prediction["plan_sha256"] == analysis["plan_sha256"] == context["plan_sha256"]
            and global_marker["state"] == "GLOBAL_OUTPUTS_FROZEN" and prediction["state"] == "PREDICTIONS_FROZEN"
            and global_marker.get("observations") == 51970 and global_marker.get("completed_units") == 2000
            and global_marker.get("labels_read") is False and global_marker.get("first_gold_opened_unix") is None
            and prediction.get("heldout_prediction_rows") == 2880 and prediction.get("heldout_labels_read") is False
            and prediction["global_outputs_frozen_sha256"] == acceptance["artifact_sha256"]["global_outputs_frozen"]
            and analysis["predictions_frozen_sha256"] == acceptance["artifact_sha256"]["predictions_frozen"]
            and analysis["scientific_gate"] == acceptance["scientific_gate"]
            and analysis["analysis_semantic_sha256"] == digest({key: value for key, value in analysis.items() if key != "analysis_semantic_sha256"}),
            "Accepted generation/prediction/analysis hash chain differs")
    for value in (global_marker, prediction, analysis):
        require(value["source_manifest_sha256"] == context["source_manifest_sha256"]
                and value["pipeline_source_manifest_sha256"] == context["pipeline_source_manifest_sha256"], "Accepted scientific source identity changed")
    require(prediction["feature_sha256"] == global_marker["features_sha256"] == plan["features"]["sha256"]
            and prediction["features_receipt_sha256"] == global_marker["features_receipt_sha256"] == plan["features"]["receipt_sha256"],
            "Accepted feature provenance differs")
    descriptor = global_marker["artifacts"]["observations"]
    observation_raw = check_hash(workspace, descriptor["path"], descriptor["sha256"])
    require(len(observation_raw) == descriptor["bytes"], "Canonical observation size changed")
    anchors[descriptor["path"]] = descriptor["sha256"]
    prediction_path = base + "/heldout_predictions.json"
    predictions = decode(check_hash(workspace, prediction_path, prediction["predictions_sha256"]))
    require(digest(predictions) == prediction["predictions_semantic_sha256"], "Frozen prediction semantic SHA differs")
    anchors[prediction_path] = prediction["predictions_sha256"]
    parameter_path = base + "/parameters.json"
    check_hash(workspace, parameter_path, prediction["parameters_sha256"])
    anchors[parameter_path] = prediction["parameters_sha256"]
    release_path = base + "/HELDOUT_LABEL_RELEASE.json"
    release = decode(check_hash(workspace, release_path, analysis["heldout_label_release_sha256"]))
    require(release["state"] == "HELDOUT_LABELS_RELEASED" and release["plan_sha256"] == context["plan_sha256"]
            and release["predictions_frozen_sha256"] == acceptance["artifact_sha256"]["predictions_frozen"]
            and release["heldout_labels_sha256"] == analysis["heldout_labels_sha256"], "Accepted heldout release differs")
    anchors[release_path] = analysis["heldout_label_release_sha256"]
    require(acceptance["reference_source_pins"]["original_scorer"] == context["scorer_sha256"], "Accepted original scorer pin differs")
    context.update(anchors=anchors, acceptance=acceptance, analysis=analysis, predictions=predictions,
                   observation_raw=observation_raw, expected_labels_sha256=release["heldout_labels_sha256"],
                   recipe_sha256=bytes_sha(recipe_raw), acceptance_sha256=bytes_sha(acceptance_raw))
    return context


def open_labels(workspace, context):
    name = context["base"] + "/DESCRIPTIVE_FIRST_LABEL_ACCESS.json"
    binding = {"run_id": context["plan"]["run_id"], "plan_sha256": context["plan_sha256"],
               "recipe_sha256": context["recipe_sha256"], "acceptance_sha256": context["acceptance_sha256"],
               "artifact_sha256": context["acceptance"]["artifact_sha256"], "expected_labels_sha256": context["expected_labels_sha256"],
               "source_sha256": bytes_sha(Path(__file__).read_bytes()), "scorer_sha256": context["scorer_sha256"],
               "label_path": context["plan"]["label_release"]["test"]}
    prior = workspace.json(name) if workspace.path(name).exists() else None
    if prior is not None:
        require(prior.get("schema") == "graph_credit_g2_descriptive_first_label_access_v1"
                and prior.get("event") == "HANDLE_OPENED_BEFORE_FIRST_READ"
                and all(prior.get(key) == value for key, value in binding.items()), "Prior descriptive first-label event binds different inputs")
    with workspace.path(binding["label_path"]).open("rb") as handle:
        info = os.fstat(handle.fileno())
        event = {"schema": "graph_credit_g2_descriptive_first_label_access_v1", "event": "HANDLE_OPENED_BEFORE_FIRST_READ", **binding,
                 "first_gold_opened_unix": time.time(), "handle": {"bytes_at_open": info.st_size, "device": info.st_dev, "inode": info.st_ino}}
        if prior is None:
            workspace.write(name, event, exclusive=True)
        context["first_label_event_sha256"] = bytes_sha(workspace.read(name))
        workspace.write(context["base"] + "/label_access_attempts/" + uuid.uuid4().hex[:20] + ".json",
                        {**event, "first_event_sha256": context["first_label_event_sha256"], "is_first_access": prior is None}, exclusive=True)
        context["labels_read"] = True
        raw = handle.read()
    require(bytes_sha(raw) == context["expected_labels_sha256"] and raw.endswith(b"\n"), "Heldout label bytes differ from accepted inputs")
    labels = {}
    for line in raw.splitlines():
        row = decode(line)
        require(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in labels, "Malformed or duplicate heldout label")
        labels[row["id"]] = row
    expected = {qid for qid, part in context["plan"]["partitions"].items() if part == "test"}
    require(set(labels) == expected and len(labels) == 120, "Heldout label support changed")
    return labels


def original_scorer(workspace, context):
    path = workspace.path(context["scorer_path"])
    require(bytes_sha(path.read_bytes()) == context["scorer_sha256"], "Original scorer changed")
    spec = importlib.util.spec_from_file_location("g2_descriptive_original_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_item


def cross_seed_risk(targets, prediction):
    errors = [prediction - value for value in targets]
    return math.fsum(left * right for left, right in itertools.combinations(errors, 2)) / 10


def scored_coordinates(context, labels, scorer):
    plan = context["plan"]
    test = {qid for qid, part in plan["partitions"].items() if part == "test"}
    quality, seen = {}, set()
    for line in context["observation_raw"].splitlines():
        row = decode(line)
        require(row["quality"] is None and row["labels_read"] is False and row["plan_sha256"] == context["plan_sha256"],
                "Canonical raw generation provenance changed")
        key = (row["question_id"], row["seed"], tuple(row["graph_mask"]))
        require(key not in seen, "Duplicate unique generation output")
        seen.add(key)
        if key[0] not in test:
            continue
        require(row["partition"] == "test" and type(key[1]) is int and key[1] in SEEDS, "Unexpected heldout generation seed/partition")
        value = scorer(row["service_output"], labels[key[0]], task="gsm8k")
        require(finite(value) and value in (0, 1), "Original scorer returned an invalid binary quality")
        quality[key] = value
    expected = {(unit["question_id"], unit["seed"], tuple(request["mask"]))
                for unit in plan["units"] if unit["partition"] == "test" for request in unit["mask_requests"]}
    require(set(quality) == expected and len(seen) == plan["expected_outputs"], "Generation output support changed")
    coordinates, predicted = [], {}
    for row in context["predictions"]:
        key = (row["question_id"], row["anchor_id"], row["edge"])
        require(key not in predicted and set(row["predictions"]) == {"B0", "C1", "C2"}, "Frozen prediction coordinate/model duplicates")
        predicted[key] = row
    for qid in sorted(test):
        for contrast in sorted(plan["designs"][qid]["contrasts"], key=lambda row: (row["anchor_id"], row["edge"])):
            key = (qid, contrast["anchor_id"], contrast["edge"])
            row = predicted.pop(key)
            left, right = plan["masks"][contrast["mask0_id"]], plan["masks"][contrast["mask1_id"]]
            require(row["mask0"] == left and row["mask1"] == right, "Frozen prediction mask differs from plan")
            values = [quality[(qid, seed, tuple(right))] - quality[(qid, seed, tuple(left))] for seed in SEEDS]
            estimates = {"ZERO": 0.0, **row["predictions"]}
            require(all(finite(value) for value in estimates.values()), "Nonfinite frozen prediction")
            mean = math.fsum(values) / 5
            coordinates.append({"qid": qid, "family": key[1], "edge": key[2], "targets": values,
                                "risk": {name: cross_seed_risk(values, estimates[name]) for name in MODELS},
                                "observed_squared_error": {name: (estimates[name] - mean) ** 2 for name in MODELS}})
    require(not predicted and len(coordinates) == 2880, "Frozen prediction support changed")
    return coordinates


def bootstrap_counts(question_count):
    require(question_count >= 2, "Descriptive bootstrap requires at least two question blocks")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, question_count, size=(BOOTSTRAP_SAMPLES, question_count))
    counts = np.zeros((BOOTSTRAP_SAMPLES, question_count), dtype=np.int64)
    np.add.at(counts, (np.arange(BOOTSTRAP_SAMPLES)[:, None], indices), 1)
    return counts


def summarize_panel(coordinates, question_ids, counts):
    positions = {qid: index for index, qid in enumerate(question_ids)}
    denominator = np.zeros(len(question_ids), dtype=np.int64)
    risks, errors = np.zeros((len(question_ids), 4)), np.zeros((len(question_ids), 4))
    for row in coordinates:
        index = positions[row["qid"]]
        denominator[index] += 1
        for column, name in enumerate(MODELS):
            risks[index, column] += row["risk"][name]
            errors[index, column] += row["observed_squared_error"][name]
    total = int(denominator.sum())
    bootstrap_denominator = counts @ denominator
    valid = bootstrap_denominator > 0
    bootstrap_means = (counts[valid] @ risks) / bootstrap_denominator[valid, None]
    model_stats = {}
    for column, name in enumerate(MODELS):
        model_stats[name] = {"mean_cross_seed_product_risk": float(risks[:, column].sum() / total) if total else None,
                             "mean_observed_mean_credit_squared_error": float(errors[:, column].sum() / total) if total else None,
                             "negative_coordinate_risk_count": sum(row["risk"][name] < 0 for row in coordinates)}
    comparisons = {}
    for column, name in enumerate(REFERENCES):
        values = bootstrap_means[:, column] - bootstrap_means[:, 3]
        comparisons[name + "_risk_minus_C2_risk"] = {
            "estimate": float((risks[:, column].sum() - risks[:, 3].sum()) / total) if total else None,
            "interval": np.quantile(values, [.025, .975]).tolist() if len(values) else None,
            "confidence": .95, "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "valid_bootstrap_samples": int(valid.sum()), "undefined_empty_stratum_samples": int((~valid).sum()),
            "resampling_unit": "whole_question", "exploratory": True, "multiplicity_adjustment": "none"}
    return {"support": {"question_population": len(question_ids), "questions_with_coordinates": int((denominator > 0).sum()),
                         "coordinates": total, "seed_contrasts": 5 * total, "cross_seed_pairs": 10 * total,
                         "per_question_coordinate_count_histogram": [{"coordinates": int(size), "questions": int(number)}
                             for size, number in sorted(Counter(denominator.tolist()).items())]},
            "models": model_stats, "comparisons": comparisons}


def analyze_coordinates(coordinates, analysis, families):
    by_question = defaultdict(list)
    expected_cells = {(family, edge) for family in families for edge in range(12)}
    for row in coordinates:
        by_question[row["qid"]].append(row)
    question_ids = sorted(by_question)
    require(set(question_ids) == set(analysis["question_risks"]), "Original question risk support differs")
    question_risks = {}
    for qid in question_ids:
        rows = by_question[qid]
        require(len(rows) == 24 and {(row["family"], row["edge"]) for row in rows} == expected_cells, "A question lacks a fixed family-edge coordinate")
        question_risks[qid] = {name: math.fsum(row["risk"][name] for row in rows) / 24 for name in MODELS}
        require(all(finite(analysis["question_risks"][qid][name]) and math.isclose(question_risks[qid][name], analysis["question_risks"][qid][name],
                          rel_tol=1e-10, abs_tol=1e-12) for name in ("B0", "C1", "C2")),
                "Descriptive question risks differ from the original G2 analysis")
    counts = bootstrap_counts(len(question_ids))
    overall = summarize_panel(coordinates, question_ids, counts)
    for name in ("B0", "C1", "C2"):
        if "mean_cross_seed_risk" in analysis:
            require(math.isclose(overall["models"][name]["mean_cross_seed_product_risk"], analysis["mean_cross_seed_risk"][name],
                                 rel_tol=1e-10, abs_tol=1e-12), "Original overall risk differs")
    for original, comparison in (("primary", "B0_risk_minus_C2_risk"), ("first_order_diagnostic", "C1_risk_minus_C2_risk")):
        if original in analysis:
            repeated, saved = overall["comparisons"][comparison], analysis[original]
            require(math.isclose(repeated["estimate"], saved["estimate"], rel_tol=1e-10, abs_tol=1e-12)
                    and np.allclose(repeated["interval"], saved["interval"], rtol=1e-10, atol=1e-12),
                    "Original primary/diagnostic interval differs from the whole-question replay")
    family_results = [{"anchor_family": family, "edge": edge,
                       **summarize_panel([row for row in coordinates if row["family"] == family and row["edge"] == edge], question_ids, counts)}
                      for family, edge in sorted(expected_cells)]
    strata = {}
    for name, zero in (("all_five_observed_differences_zero", True), ("any_observed_difference_nonzero", False)):
        selected = [row for row in coordinates if all(value == 0 for value in row["targets"]) == zero]
        strata[name] = {**summarize_panel(selected, question_ids, counts),
                        "selection": "Membership uses the same five noisy observed credit differences",
                        "estimand": "descriptive coordinate-weighted conditional cross-seed product",
                        "conditional_true_risk_unbiasedness_claimed": False,
                        "interpretation": "Conditioning on observed targets removes the unconditional true-risk unbiasedness guarantee"}
    contributions = {}
    for name in REFERENCES:
        values = [question_risks[qid][name] - question_risks[qid]["C2"] for qid in question_ids]
        groups = {}
        signed = {"positive": [value for value in values if value > 0],
                  "negative": [value for value in values if value < 0], "zero": [value for value in values if value == 0]}
        for sign, chosen in signed.items():
            groups[sign] = {"questions": len(chosen), "sum": math.fsum(chosen),
                            "mean": math.fsum(chosen) / len(chosen) if chosen else None,
                            "absolute_sum": math.fsum(abs(value) for value in chosen),
                            "contribution_to_overall_question_mean": math.fsum(chosen) / len(question_ids)}
        contributions[name + "_risk_minus_C2_risk"] = {"question_population": len(question_ids), "sign_rule": "strict sign without threshold", "groups": groups}
    return {"overall": overall, "family_edge": family_results, "observed_credit_strata": strata,
            "signed_question_contributions": contributions, "original_question_risks_verified": True,
            "negative_risks_preserved": True, "exploratory_intervals_unadjusted": True}


def run_appendix(workspace, context, acceptance_path):
    prelabel_gate(workspace, context, acceptance_path)
    scorer = original_scorer(workspace, context)
    labels = open_labels(workspace, context)
    coordinates = scored_coordinates(context, labels, scorer)
    result = analyze_coordinates(coordinates, context["analysis"], context["families"])
    for name, expected in context["anchors"].items():
        check_hash(workspace, name, expected)
    for manifest in (context["plan"]["source_manifest"], context["plan"]["pipeline_source_manifest"]):
        for name, item in manifest.items():
            check_hash(workspace, name, item["sha256"])
    require({key: value for key, value in workspace.json(context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json").items() if key != "created_unix"}
            == recipe_value(context), "Descriptive source/dependencies changed during this attempt")
    return {"schema": "graph_credit_g2_descriptive_appendix_v1", "status": "DESCRIPTIVE_APPENDIX_COMPLETE",
            "run_id": context["plan"]["run_id"], "plan_sha256": context["plan_sha256"],
            "recipe_sha256": context["recipe_sha256"], "acceptance_sha256": context["acceptance_sha256"],
            "source_sha256": bytes_sha(Path(__file__).read_bytes()), "scorer_sha256": context["scorer_sha256"],
            "artifact_sha256": context["acceptance"]["artifact_sha256"], "first_label_access_sha256": context["first_label_event_sha256"],
            "primary_scientific_gate": context["analysis"]["scientific_gate"], "primary_gate_modified": False,
            "scientific_completion": False, "scope": "Post-acceptance descriptive exploratory appendix",
            "zero_credit_reference": 0.0, "cross_seed_product_formula": "sum_{s<t}(prediction-y_s)(prediction-y_t)/10",
            "confidence": .95, "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "multiplicity_adjustment": "none", "labels_read": True, "contains_individual_identifiers_or_label_values": False,
            **result}


def execute(workspace, plan_path, acceptance_path, mode):
    started, wall, cpu = time.time(), time.monotonic(), time.process_time()
    context = source_context(workspace, plan_path)
    with workspace.lease(context["base"] + "/appendix.lock"):
        if mode == "freeze":
            return freeze_recipe(workspace, context)
        attempt_path = context["base"] + "/cpu_attempts/" + uuid.uuid4().hex[:20] + ".json"
        attempt = {"schema": "graph_credit_g2_descriptive_cpu_attempt_v1", "status": "RUNNING",
                   "plan_sha256": context["plan_sha256"], "source_sha256": bytes_sha(Path(__file__).read_bytes()),
                   "started_unix": started, "labels_read": False, "cost_in_original_g2_summary": False}
        workspace.write(attempt_path, attempt, exclusive=True)
        try:
            report = run_appendix(workspace, context, acceptance_path)
            report.update(cpu_attempt_path=attempt_path, cpu_seconds_before_publication=time.process_time() - cpu,
                          wall_seconds_before_publication=time.monotonic() - wall, completed_unix=time.time())
            report["appendix_semantic_sha256"] = digest(report)
            report_path = context["base"] + "/DESCRIPTIVE_APPENDIX.json"
            prior = workspace.json(report_path) if workspace.path(report_path).exists() else None
            if prior is not None:
                volatile = {"cpu_attempt_path", "cpu_seconds_before_publication", "wall_seconds_before_publication", "completed_unix", "appendix_semantic_sha256"}
                require({key: value for key, value in prior.items() if key not in volatile} == {key: value for key, value in report.items() if key not in volatile}
                        and prior["appendix_semantic_sha256"] == digest({key: value for key, value in prior.items() if key != "appendix_semantic_sha256"}),
                        "Existing aggregate appendix differs from deterministic replay")
                report = prior
            else:
                workspace.write(report_path, report, exclusive=True)
            attempt.update(status="COMPLETE", output_sha256=bytes_sha(workspace.read(report_path)))
            return report
        except WaitingError:
            attempt["status"] = "WAITING_FOR_INDEPENDENT_ACCEPTANCE"
            raise
        except BaseException as exc:
            attempt.update(status="FAILED", error_type=type(exc).__name__)
            raise
        finally:
            attempt.update(labels_read=bool(context.get("labels_read")), completed_unix=time.time(),
                           cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
                           timing_scope="This appendix attempt including aggregate publication")
            workspace.write(attempt_path, attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--acceptance", default="artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json")
    parser.add_argument("--mode", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    workspace = Workspace(args.root)
    plan_path = Path(args.plan).relative_to(workspace.root).as_posix() if Path(args.plan).is_absolute() else Path(args.plan).as_posix()
    acceptance_path = Path(args.acceptance).relative_to(workspace.root).as_posix() if Path(args.acceptance).is_absolute() else Path(args.acceptance).as_posix()
    try:
        value = execute(workspace, plan_path, acceptance_path, args.mode)
    except WaitingError:
        print(canonical({"status": "WAITING_FOR_INDEPENDENT_ACCEPTANCE", "labels_read": False}), flush=True)
        return 3
    print(canonical({key: value.get(key) for key in ("status", "state", "run_id", "labels_read", "primary_gate_modified")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
