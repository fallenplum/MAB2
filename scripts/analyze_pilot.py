#!/usr/bin/env python3
"""Analyse preregistered paired 2x2 graph interventions, never train a policy.

The estimand is the question-weighted mean squared *expected* interaction.
Its U-statistic uses distinct generation seeds and can legitimately be negative.
All observations from a question stay in one bootstrap block, including repeated
graphs, contexts, edge pairs, and generation seeds. A PASS is a local G1 evidence
gate, not a claim that the subsequent active acquisition algorithm works.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


CELLS = ("00", "01", "10", "11")
TASK_ALIASES = {"gsm8k": "gsm8k", "gsm8k_main": "gsm8k", "gsm": "gsm8k"}
FAMILY_ALIASES = {
    "full": "full", "complete": "full", "dense": "full",
    "chain": "chain", "star": "star", "fixed_random": "fixed_random",
    "random": "fixed_random", "random_sparse": "fixed_random",
    "sequential_deletion": "sequential_deletion", "sequential": "sequential_deletion",
    "delete_sequence": "sequential_deletion",
}


@dataclass(frozen=True)
class GateConfig:
    min_questions: int = 200
    min_seeds: int = 5
    required_families: tuple[str, ...] = (
        "full", "chain", "star", "fixed_random", "sequential_deletion"
    )
    require_all_families_per_question: bool = True
    expected_task: str = "gsm8k"
    allowed_splits: tuple[str, ...] = ("dev", "motivation")
    confidence: float = 0.95
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 20261103
    # A useful-effect threshold must be fixed before the run. Zero only tests
    # distinguishability from generation noise, not economic usefulness.
    min_effect_squared: float = 0.0
    expected_protocol_hash: str | None = None
    require_plan_verification: bool = True

    def validate(self) -> None:
        if self.min_questions < 1 or self.min_seeds < 2:
            raise ValueError("min_questions >= 1 and min_seeds >= 2 are required")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between 0 and 1")
        if self.bootstrap_samples < 100:
            raise ValueError("at least 100 bootstrap replicates are required")
        if not math.isfinite(self.min_effect_squared) or self.min_effect_squared < 0:
            raise ValueError("min_effect_squared must be finite and nonnegative")
        if len(set(self.required_families)) != len(self.required_families):
            raise ValueError("required_families must be unique")
        if not self.required_families or not self.allowed_splits:
            raise ValueError("required_families and allowed_splits must be nonempty")


def interaction_statistics(interactions: Sequence[float]) -> dict[str, float | int]:
    """Independent seeds are replicates; no absolute-value/significance shortcut."""
    values = np.asarray(interactions, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite one-dimensional interactions required")
    count = len(values)
    signal = (float(values.sum()) ** 2 - float(values @ values)) / (count * (count - 1))
    variance = float(values.var(ddof=1))
    return {
        "n_seeds": count,
        "mean_interaction": float(values.mean()),
        "observed_mean_interaction_squared": float(values.mean()) ** 2,
        "observed_single_seed_interaction_squared": float(np.mean(values ** 2)),
        "signal_squared_unbiased": signal,
        "single_seed_noise_variance": variance,
        "noise_variance_of_mean": variance / count,
    }


def stratified_question_bootstrap(
    question_values: Mapping[tuple[str, str], float],
    strata: Mapping[tuple[str, str], str],
    *, samples: int, seed: int, confidence: float,
) -> dict[str, Any]:
    """Resample whole questions within task/split strata, keeping fixed sizes.

    A context is never a resampling unit because one question may have many
    contexts; resampling those as independent would produce pseudoreplication.
    """
    if not question_values:
        return {"estimate": None, "interval": None, "n_questions": 0, "strata": {}}
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(question_values):
        grouped[strata[key]].append(float(question_values[key]))
    rng = np.random.default_rng(seed)
    total = len(question_values)
    boot = np.zeros(samples, dtype=np.float64)
    stratum_counts = {}
    for name in sorted(grouped):
        values = np.asarray(grouped[name], dtype=np.float64)
        stratum_counts[name] = len(values)
        # Keep memory bounded for large user-supplied problem streams.
        for start in range(0, samples, 256):
            stop = min(start + 256, samples)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            boot[start:stop] += values[indices].sum(axis=1) / total
    alpha = (1 - confidence) / 2
    interval = np.quantile(boot, [alpha, 1 - alpha])
    return {
        "estimate": float(np.mean(list(question_values.values()))),
        "interval": [float(interval[0]), float(interval[1])],
        "confidence": confidence,
        "n_questions": total,
        "strata": stratum_counts,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "resampling_unit": "whole_question",
        "weighting": "equal_questions; equal_complete_context_edge_pair_groups_within_question",
    }


def _family(row: Mapping[str, Any]) -> str:
    family = str(row.get("context_family", "")).strip().lower()
    if not family:
        name = str(row.get("context_id", "")).lower()
        for prefix in sorted(FAMILY_ALIASES, key=len, reverse=True):
            if name == prefix or name.startswith((prefix + ":", prefix + "_", prefix + "-")):
                family = prefix
                break
        if not family:
            family = name
    return FAMILY_ALIASES.get(family, family)


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _plan_metadata(plan_path: str | Path | None) -> dict[str, Any]:
    if plan_path is None:
        return {"provided": False, "sha256": None, "path": None}
    path = Path(plan_path)
    data = path.read_bytes()
    # Parsing also rejects a non-JSON file accidentally supplied as a plan.
    plan = json.loads(data)
    if not isinstance(plan, dict):
        raise ValueError("frozen plan must be a JSON object")
    return {
        "provided": True, "sha256": hashlib.sha256(data).hexdigest(),
        "path": str(path.resolve()), "plan": plan,
    }


def _verify_plan_coverage(plan: Mapping[str, Any], groups: Mapping, family_by_group: Mapping,
                          protocols: set[str]) -> dict[str, Any]:
    """Verify the full Cartesian design, not merely a self-reported hash.

    This is execution-plan integrity, not proof of real-model execution, S0,
    data lineage, correct scoring, or generation-before-label disclosure.
    Those remain separate audits and are explicitly outside this statistic.
    """
    errors: list[str] = []
    missing_groups: list[list[Any]] = []
    result = {"verified": False, "errors": errors, "missing_group_count": 0,
              "missing_groups_first_25": [], "expected_groups": None,
              "expected_scored_rows": None}
    required = {"schema", "question_ids", "question_payload_sha256", "data_sha256",
                "seeds", "edge_pairs", "base_masks", "protocol_hash", "edge_order",
                "config", "model_identity", "model_manifest", "feedback_timing", "sampling_pairing",
                "source_manifest", "source_manifest_sha256", "scorer_identity"}
    absent = sorted(required - set(plan))
    if absent:
        errors.append(f"frozen plan missing required design fields: {absent}")
        return result
    try:
        if plan["schema"] != "graph_credit_pilot_plan_v1":
            raise ValueError("unsupported frozen plan schema")
        qids, seeds, pairs, contexts = (plan[k] for k in ("question_ids", "seeds", "edge_pairs", "base_masks"))
        if not isinstance(qids, list) or not qids or any(not isinstance(q, str) or not q for q in qids) or len(set(qids)) != len(qids):
            raise ValueError("plan question_ids must be nonempty unique strings")
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("plan seeds must be unique explicit integers")
        if not isinstance(pairs, list) or not pairs or any(not isinstance(p, list) or len(p) != 2 or any(type(e) is not int or e not in range(12) for e in p) or p[0] == p[1] for p in pairs):
            raise ValueError("plan must contain valid ordered edge pairs")
        if len(set(map(tuple, pairs))) != len(pairs):
            raise ValueError("plan contains duplicate edge pairs")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError("plan base_masks must be a nonempty list")
        names = set()
        context_by_id = {}
        for context in contexts:
            if not isinstance(context, dict) or set(context) < {"name", "mask"}:
                raise ValueError("invalid plan graph context")
            name, mask = context.get("name"), context.get("mask")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("plan context names must be unique nonempty strings")
            names.add(name)
            if not isinstance(mask, list) or len(mask) != 12 or any(type(v) is not int or v not in (0, 1) for v in mask):
                raise ValueError("plan context mask must have twelve binary integer entries")
            encoded = json.dumps(mask, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            context_id = name + "_" + hashlib.sha256(encoded).hexdigest()[:12]
            context_by_id[context_id] = context
        expected_edges = [[source, target] for source in range(4) for target in range(4) if source != target]
        if plan["edge_order"] != expected_edges:
            raise ValueError("plan edge order differs from fixed four-worker source-major order")
        if protocols != {plan["protocol_hash"]} or not _hash_ok(plan["protocol_hash"]):
            raise ValueError("plan protocol_hash differs from observations")
        if not _hash_ok(plan["data_sha256"]):
            raise ValueError("invalid plan input-data SHA256")
        payloads = plan["question_payload_sha256"]
        if not isinstance(payloads, list) or len(payloads) != len(qids) or any(not _hash_ok(v) for v in payloads):
            raise ValueError("plan question-payload hashes must cover every planned question")
        if not isinstance(plan["model_identity"], dict) or not plan["model_identity"] or not plan["model_manifest"]:
            raise ValueError("plan lacks a frozen model identity/file manifest")
        if plan["feedback_timing"] != "all_assigned_outputs_frozen_before_labels_opened":
            raise ValueError("unexpected planned feedback timing")
        if plan["sampling_pairing"] != "same_question_seed_node_across_all_graphs":
            raise ValueError("unexpected planned node-randomness pairing")
        if not isinstance(plan["config"], dict):
            raise ValueError("plan config must be an object")
        sources = plan["source_manifest"]
        source_keys = {"src/graph_credit/executor.py", "src/graph_credit/backend.py",
                       "src/graph_credit/scoring.py", "scripts/run_pilot.py", "scripts/analyze_pilot.py"}
        if not isinstance(sources, dict) or not source_keys <= set(sources):
            raise ValueError("source manifest lacks required executor/backend/scoring/pilot/statistics modules")
        encoded_sources = json.dumps(sources, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if hashlib.sha256(encoded_sources).hexdigest() != plan["source_manifest_sha256"]:
            raise ValueError("source_manifest_sha256 does not match canonical frozen source manifest")
        for name, item in sources.items():
            if not isinstance(item, dict) or not _hash_ok(item.get("sha256")) or type(item.get("bytes")) is not int or item["bytes"] < 1:
                raise ValueError(f"invalid source-manifest entry: {name}")
        scorer = plan["scorer_identity"]
        if not isinstance(scorer, dict) or not isinstance(scorer.get("callable"), str) or ":" not in scorer["callable"]:
            raise ValueError("missing explicit frozen scoring callable")
        scorer_module = "src/" + scorer["callable"].split(":")[0].replace(".", "/") + ".py"
        if scorer_module not in sources or scorer.get("module_sha256") != sources[scorer_module]["sha256"] or scorer.get("module_bytes") != sources[scorer_module]["bytes"]:
            raise ValueError("scorer identity differs from frozen source module")
        expected_keys = {(qid, context_id, e, f) for qid in qids for context_id in context_by_id for e, f in pairs}
        observed_keys = {(qid, context_id, e, f) for task, qid, context_id, e, f in groups}
        missing_groups = [list(key) for key in sorted(expected_keys - observed_keys)]
        unexpected = sorted(observed_keys - expected_keys)
        if unexpected:
            errors.append(f"observations contain {len(unexpected)} groups outside frozen design; first={unexpected[:3]}")
        for group, seed_rows in groups.items():
            task, qid, context_id, e, f = group
            if (qid, context_id, e, f) not in expected_keys:
                continue
            context = context_by_id[context_id]
            expected_family = FAMILY_ALIASES.get(context["name"], context["name"])
            if family_by_group[group] != expected_family:
                errors.append(f"{group}: context family differs from frozen plan")
            extra_seeds = sorted(set(seed_rows) - set(seeds))
            missing_seeds = sorted(set(seeds) - set(seed_rows))
            if extra_seeds:
                errors.append(f"{group}: unplanned generation seeds {extra_seeds}")
            if missing_seeds:
                missing_groups.append([qid, context_id, e, f, "missing_seeds", missing_seeds])
            for seed, cells in seed_rows.items():
                missing_cells = [cell for cell in CELLS if cell not in cells or cells[cell].get("quality") is None]
                if missing_cells:
                    missing_groups.append([qid, context_id, e, f, seed, "missing_or_unscored_cells", missing_cells])
                for cell, row in cells.items():
                    expected_mask = list(context["mask"])
                    expected_mask[e], expected_mask[f] = int(cell[0]), int(cell[1])
                    if row.get("graph_mask") != expected_mask:
                        errors.append(f"{group}, seed {seed}, cell {cell}: actual graph mask differs from frozen plan")
        result.update({"verified": not errors and not missing_groups,
                       "missing_group_count": len(missing_groups),
                       "missing_groups_first_25": missing_groups[:25],
                       "expected_groups": len(expected_keys),
                       "expected_scored_rows": len(expected_keys) * len(seeds) * 4,
                       "planned_questions": len(qids), "planned_seeds": seeds,
                       "planned_edge_pairs": pairs, "planned_context_names": sorted(names)})
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return result


def analyze_observations(
    rows: Sequence[Mapping[str, Any]], config: GateConfig | None = None,
    *, plan_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or GateConfig()
    config.validate()
    plan_meta = _plan_metadata(plan_path)
    errors: list[str] = []
    groups: dict[tuple[str, str, str, int, int], dict[int, dict[str, Mapping[str, Any]]]] = {}
    family_by_group: dict[tuple[str, str, str, int, int], str] = {}
    split_by_question: dict[tuple[str, str], str] = {}
    protocols: set[str] = set()
    plans: set[str] = set()
    exact_duplicates = 0
    unscored = 0
    for index, original in enumerate(rows, 1):
        row = dict(original)
        try:
            task = str(row.get("task", row.get("dataset", ""))).lower()
            task = TASK_ALIASES.get(task, task)
            qid = str(row["question_id"])
            context = str(row["context_id"])
            split = str(row["split"]).lower()
            if not task or not qid or not context or not split:
                raise ValueError("empty task/question/context/split")
            edge_e, edge_f, seed = row["edge_e"], row["edge_f"], row["seed"]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (edge_e, edge_f, seed)):
                raise ValueError("edge indices and seed must be integers")
            if edge_e == edge_f or edge_e < 0 or edge_f < 0:
                raise ValueError("two distinct nonnegative edge indices required")
            cell = str(row["cell"])
            if cell not in CELLS:
                raise ValueError("cell must be one of 00/01/10/11")
            for field, expected in (("mask_e", int(cell[0])), ("mask_f", int(cell[1]))):
                if field in row and row[field] != expected:
                    raise ValueError(f"{field} disagrees with cell")
            if "graph_mask" in row:
                mask = row["graph_mask"]
                if not isinstance(mask, list) or len(mask) <= max(edge_e, edge_f):
                    raise ValueError("graph_mask missing intervention positions")
                if any(v not in (0, 1) for v in mask):
                    raise ValueError("graph_mask must be binary")
                if (mask[edge_e], mask[edge_f]) != (int(cell[0]), int(cell[1])):
                    raise ValueError("graph_mask disagrees with intervention cell")
            if not _hash_ok(row.get("protocol_hash")):
                raise ValueError("protocol_hash must be a lowercase SHA256")
            if not _hash_ok(row.get("frozen_plan_sha256")):
                raise ValueError("frozen_plan_sha256 must be a lowercase SHA256")
            protocols.add(row["protocol_hash"])
            plans.add(row["frozen_plan_sha256"])
            qkey = (task, qid)
            if qkey in split_by_question and split_by_question[qkey] != split:
                raise ValueError("question appears in multiple splits")
            split_by_question[qkey] = split
            family = _family(row)
            group = (task, qid, context, edge_e, edge_f)
            if group in family_by_group and family_by_group[group] != family:
                raise ValueError("one context has conflicting families")
            family_by_group[group] = family
            cells = groups.setdefault(group, {}).setdefault(seed, {})
            if cell in cells:
                # Resume can append an exact row twice. Never silently accept
                # conflicting repetitions under a supposedly identical seed.
                if dict(cells[cell]) == row:
                    exact_duplicates += 1
                    continue
                raise ValueError("conflicting duplicate question/context/edge_pair/seed/cell")
            quality = row.get("quality")
            if quality is None:
                unscored += 1
            elif isinstance(quality, bool) or not isinstance(quality, (int, float)):
                raise ValueError("quality must be a finite number in [0,1]")
            elif not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError("quality must be a finite number in [0,1]")
            cells[cell] = row
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")

    if len(protocols) != 1:
        errors.append(f"expected exactly one protocol hash; found {len(protocols)}")
    if len(plans) != 1:
        errors.append(f"expected exactly one frozen plan hash; found {len(plans)}")
    if config.expected_protocol_hash and protocols != {config.expected_protocol_hash}:
        errors.append("protocol hash differs from preregistered expected_protocol_hash")
    if plan_meta["provided"] and plans != {plan_meta["sha256"]}:
        errors.append("observed frozen plan hash differs from supplied plan bytes")
    plan_coverage = _verify_plan_coverage(plan_meta["plan"], groups, family_by_group, protocols) if plan_meta["provided"] else None
    if plan_coverage:
        errors.extend(plan_coverage["errors"])

    support: list[dict[str, Any]] = []
    pair_summary: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    values_by_question: dict[tuple[str, str], list[float]] = defaultdict(list)
    means_by_question: dict[tuple[str, str], list[float]] = defaultdict(list)
    noise_by_question: dict[tuple[str, str], list[float]] = defaultdict(list)
    families_by_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    incomplete_groups = 0
    min_complete_seed_count = None
    for group in sorted(groups):
        task, qid, context, edge_e, edge_f = group
        family = family_by_group[group]
        seed_rows = groups[group]
        complete_seeds: list[int] = []
        interactions: list[float] = []
        credits_f0: list[float] = []
        credits_f1: list[float] = []
        missing_cells: dict[str, list[str]] = {}
        context_masks: set[tuple[int, ...]] = set()
        for seed in sorted(seed_rows):
            cells = seed_rows[seed]
            missing = [cell for cell in CELLS if cell not in cells or cells[cell].get("quality") is None]
            if missing:
                missing_cells[str(seed)] = missing
                continue
            if all("graph_mask" in cells[cell] for cell in CELLS):
                backgrounds = []
                for cell in CELLS:
                    mask = tuple(v for j, v in enumerate(cells[cell]["graph_mask"]) if j not in (edge_e, edge_f))
                    backgrounds.append(mask)
                    context_masks.add(mask)
                if len(set(backgrounds)) != 1:
                    errors.append(f"{group}, seed {seed}: non-intervention graph edges changed within 2x2")
            q = {cell: float(cells[cell]["quality"]) for cell in CELLS}
            d = q["11"] - q["01"] - q["10"] + q["00"]
            interactions.append(d)
            credits_f0.append(q["10"] - q["00"])
            credits_f1.append(q["11"] - q["01"])
            complete_seeds.append(seed)
        if len(context_masks) > 1:
            errors.append(f"{group}: background graph changed across independent seeds")
        # A complete one-seed smoke run lacks statistical scale, but is not a
        # partially written execution. It must be PILOT_ONLY, not INCOMPLETE.
        is_complete = not missing_cells and bool(complete_seeds)
        if not is_complete:
            incomplete_groups += 1
        if min_complete_seed_count is None:
            min_complete_seed_count = len(complete_seeds)
        else:
            min_complete_seed_count = min(min_complete_seed_count, len(complete_seeds))
        record: dict[str, Any] = {
            "task": task, "question_id": qid, "split": split_by_question[(task, qid)],
            "context_id": context, "context_family": family, "edge_pair": [edge_e, edge_f],
            "observed_seed_count": len(seed_rows), "complete_seeds": complete_seeds,
            "n_complete_seeds": len(complete_seeds), "missing_or_unscored_cells": missing_cells,
            "complete": is_complete, "interactions_by_seed": dict(zip(map(str, complete_seeds), interactions)),
        }
        if len(interactions) >= 2:
            stats = interaction_statistics(interactions)
            record.update(stats)
            mean_q0, mean_q1 = float(np.mean(credits_f0)), float(np.mean(credits_f1))
            record["mean_credit_when_f_off"] = mean_q0
            record["mean_credit_when_f_on"] = mean_q1
            record["descriptive_credit_sign_flip"] = mean_q0 * mean_q1 < 0
            if is_complete:
                key = (task, qid)
                values_by_question[key].append(float(stats["signal_squared_unbiased"]))
                means_by_question[key].append(float(stats["mean_interaction"]))
                noise_by_question[key].append(float(stats["noise_variance_of_mean"]))
                families_by_question[key].add(family)
        pkey = (task, family, edge_e, edge_f)
        agg = pair_summary.setdefault(pkey, {
            "task": task, "context_family": family, "edge_pair": [edge_e, edge_f],
            "question_ids": set(), "context_ids": set(), "groups": 0,
            "complete_groups": 0, "complete_seed_blocks": 0,
        })
        agg["question_ids"].add(qid)
        agg["context_ids"].add(context)
        agg["groups"] += 1
        agg["complete_groups"] += int(is_complete)
        agg["complete_seed_blocks"] += len(complete_seeds)
        support.append(record)

    pair_support = []
    for key in sorted(pair_summary):
        item = pair_summary[key]
        item["n_questions"] = len(item.pop("question_ids"))
        item["context_ids"] = sorted(item["context_ids"])
        pair_support.append(item)
    question_values = {key: float(np.mean(values)) for key, values in values_by_question.items()}
    strata = {key: f"{key[0]}:{split_by_question[key]}" for key in question_values}
    primary = stratified_question_bootstrap(
        question_values, strata, samples=config.bootstrap_samples,
        seed=config.bootstrap_seed, confidence=config.confidence,
    )
    reasons: list[str] = []
    scale_reasons: list[str] = []
    questions = sorted(split_by_question)
    task_counts = Counter(task for task, _ in questions)
    split_counts = Counter(split_by_question.values())
    covered_families = sorted({family for families in families_by_question.values() for family in families})
    required_questions = max(200, config.min_questions)
    required_seeds = max(5, config.min_seeds)
    if len(question_values) < required_questions:
        scale_reasons.append(f"only {len(question_values)} complete questions; need {required_questions}")
    if min_complete_seed_count is None or min_complete_seed_count < required_seeds:
        scale_reasons.append(f"minimum complete independent seeds per group={min_complete_seed_count}; need {required_seeds}")
    if set(task_counts) != {"gsm8k"} or set(task_counts) != {config.expected_task}:
        scale_reasons.append(f"task set {sorted(task_counts)} is not exclusively GSM8K and preregistered task {config.expected_task}")
    if any(split not in config.allowed_splits or split not in ("dev", "motivation") for split in split_counts):
        scale_reasons.append(f"not exclusively allowed development splits: {dict(split_counts)}")
    required_families = set(GateConfig().required_families) | set(config.required_families)
    missing_families = sorted(required_families - set(covered_families))
    if missing_families:
        scale_reasons.append(f"missing preregistered background families: {missing_families}")
    if config.require_all_families_per_question:
        deficient = [f"{key[0]}:{key[1]}" for key in question_values
                     if not required_families <= families_by_question[key]]
        if deficient:
            scale_reasons.append(f"{len(deficient)} complete questions lack all preregistered background families")
    if config.require_plan_verification and not plan_meta["provided"]:
        scale_reasons.append("frozen execution plan file was not supplied for hash verification")
    if plan_meta["provided"]:
        plan = plan_meta["plan"]
        if plan.get("qualification_only") is not False:
            scale_reasons.append("frozen plan is qualification-only or does not explicitly permit a full motivation run")
        graph = plan.get("config", {}).get("graph", plan.get("config", {}).get("executor", {})) if isinstance(plan.get("config"), dict) else {}
        if (graph.get("rounds", 2), graph.get("n_workers", 4), graph.get("message_tokens", 128)) != (2, 4, 128):
            scale_reasons.append("G1 main protocol requires two rounds, four workers and 128-token messages")
        if not graph.get("do_sample", True) or graph.get("temperature", .7) <= 0:
            scale_reasons.append("G1 stochastic credit gate requires independent sampled generations")
    # A percentile CI of identical question summaries collapses; require at
    # least two question clusters even under deliberately relaxed pilot gates.
    if len(question_values) < 2:
        scale_reasons.append("fewer than two independent question clusters")
    if errors:
        status = "INVALID"
        reasons.extend(errors)
    elif incomplete_groups or unscored or (plan_coverage and plan_coverage["missing_group_count"]):
        status = "INCOMPLETE"
        reasons.append(f"{incomplete_groups} incomplete groups; {unscored} unscored rows")
        if plan_coverage and plan_coverage["missing_group_count"]:
            reasons.append(f"{plan_coverage['missing_group_count']} planned groups or generation-seed blocks are missing")
        reasons.extend(scale_reasons)
    elif scale_reasons:
        status = "PILOT_ONLY"
        reasons.extend(scale_reasons)
    elif primary["interval"] is not None and primary["interval"][0] > config.min_effect_squared:
        status = "PASS"
        reasons.append("preregistered local G1 signal gate met; lower confidence bound exceeds frozen signal threshold")
    else:
        status = "INSUFFICIENT_SIGNAL"
        reasons.append("signal interval does not exclude the frozen threshold; this does not prove absence of graph-conditional effects")
    return {
        "schema_version": 1,
        "scientific_gate": {"name": "G1_graph_conditional_credit_signal", "status": status,
                            "reasons": reasons, "scope": "motivation evidence only; not active-acquisition or ACL-readiness evidence"},
        "gate_config": asdict(config),
        "primary_estimand": "question-weighted mean squared expected 2x2 credit interaction",
        "primary_signal": primary,
        "noise_benchmark": {
            "mean_variance_of_paired_seed_mean": float(np.mean([np.mean(v) for v in noise_by_question.values()])) if noise_by_question else None,
            "mean_signed_interaction_descriptive": float(np.mean([np.mean(v) for v in means_by_question.values()])) if means_by_question else None,
            "note": "paired seed noise is estimated within question/context/edge_pair; negative unbiased signal estimates are retained",
        },
        "support": {"input_rows": len(rows), "exact_duplicate_rows_ignored": exact_duplicates,
                    "unscored_rows": unscored, "question_count": len(questions),
                    "complete_question_count": len(question_values), "group_count": len(groups),
                    "incomplete_group_count": incomplete_groups,
                    "minimum_complete_seed_count": min_complete_seed_count,
                    "tasks": dict(task_counts), "splits": dict(split_counts),
                    "context_families": covered_families, "by_context_edge_pair": pair_support},
        "provenance": {"protocol_hashes": sorted(protocols), "frozen_plan_hashes": sorted(plans),
                       "supplied_plan_path": plan_meta["path"],
                       "supplied_plan_sha256": plan_meta["sha256"],
                       "plan_hash_verified": bool(plan_meta["provided"] and plans == {plan_meta["sha256"]}),
                       "complete_frozen_design_coverage": plan_coverage,
                       "independent_execution_qualification": "NOT_VERIFIED_BY_THIS_STATISTICAL_SCRIPT",
                       "limitations": "Row metadata and plan bytes can be checked here; actual model inputs, generation-before-label order and dataset split lineage require separate execution audit."},
        "validation_errors": errors,
        "groups": support,
    }


def _load_config(path: Path | None, overrides: Mapping[str, Any]) -> GateConfig:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path else {}
    allowed = {field.name for field in fields(GateConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown gate configuration fields: {sorted(unknown)}")
    data.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("required_families", "allowed_splits"):
        if key in data:
            data[key] = tuple(data[key])
    result = GateConfig(**data)
    result.validate()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="Frozen GateConfig JSON; included in result")
    parser.add_argument("--plan", type=Path, help="Exact preregistered frozen execution-plan JSON")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        config = _load_config(args.config, {
            "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        })
        raw = args.observations.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        result = analyze_observations(rows, config, plan_path=args.plan)
        result["provenance"].update({
            "observations_path": str(args.observations.resolve()),
            "observations_sha256": hashlib.sha256(raw).hexdigest(),
            "gate_config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None,
        })
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")
    output = args.output or args.observations.parent / "pilot_statistics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["scientific_gate"]["status"],
                      "complete_questions": result["support"]["complete_question_count"],
                      "primary_signal": result["primary_signal"], "output": str(output.resolve())}, ensure_ascii=False))
    return 2 if result["scientific_gate"]["status"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
