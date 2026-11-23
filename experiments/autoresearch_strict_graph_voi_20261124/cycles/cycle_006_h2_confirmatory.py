"""Confirmatory runner for bounded horizon-2 Graph-VOI."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from h2_fast import run_h2_fast
from strict_graph_voi_cpu import FAMILY_SPECS, make_case, run_static


METHODS = ("masked_pairwise", "topology_voi_h2", "exact_dependency_cache", "strict_certificate_first", "full_recompute")


def one_seed(seed: int, cases_per_family: int) -> dict:
    rng = np.random.default_rng(seed)
    stats = {
        method: {"n": 0, "correct": 0, "cost": 0.0, "queries": 0, "by_family": {f: {"n": 0, "correct": 0, "cost": 0.0, "queries": 0} for f in FAMILY_SPECS}}
        for method in METHODS
    }
    for family in FAMILY_SPECS:
        for drift in (0.0, 0.1, 0.4, 1.0):
            for regime in ("uniform", "heterogeneous"):
                for edit_count in (1, 2):
                    for _ in range(cases_per_family):
                        case = make_case(rng, family, drift, regime, edit_count)
                        outputs = {
                            "masked_pairwise": run_static(case, "masked_pairwise"),
                            "topology_voi_h2": run_h2_fast(case),
                            "exact_dependency_cache": run_static(case, "exact_dependency_cache"),
                            "strict_certificate_first": run_static(case, "strict_certificate_first"),
                            "full_recompute": run_static(case, "full_recompute"),
                        }
                        target = outputs["topology_voi_h2"]["target"]
                        for method, result in outputs.items():
                            s = stats[method]; sf = s["by_family"][family]
                            ok = int(result["action"] == target)
                            for item in (s, sf):
                                item["n"] += 1; item["correct"] += ok
                                item["cost"] += float(result["cost"]); item["queries"] += int(result["queries"])
    for s in stats.values():
        s["action_exact"] = s["correct"] / s["n"]
        s["mean_cost"] = s["cost"] / s["n"]
        s["mean_queries"] = s["queries"] / s["n"]
        for sf in s["by_family"].values():
            sf["action_exact"] = sf["correct"] / sf["n"]
            sf["mean_cost"] = sf["cost"] / sf["n"]
            sf["mean_queries"] = sf["queries"] / sf["n"]
    return {"schema": "mab2_p3_cycle006_h2_seed_aggregate_v1", "seed": seed, "cases_per_family": cases_per_family, "summary": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for payload in pool.map(partial(one_seed, cases_per_family=args.cases), args.seeds):
            out = args.output_dir / f"cycle_006_seed{payload['seed']}_aggregate.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(out)


if __name__ == "__main__":
    main()

