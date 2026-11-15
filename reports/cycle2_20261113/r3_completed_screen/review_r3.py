"""Independent R3 trace/certificate review; never imports production analysis.

Inputs are R3 RESULTS/PRIVATE_TRACES and source files. Optional IC support is
disabled unless explicitly supplied. No S, question text, models or GPU calls.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dirty_from_fixed_graph_episode(episode):
    pairs = [(u, v) for u in range(4) for v in range(4) if u != v]
    shift = episode // 2
    rotate = lambda pair: ((pair[0] + shift) % 4, (pair[1] + shift) % 4)
    edit = sorted(pairs.index(rotate(pairs[e])) for e in (0, 3, 4, 6, 8, 9))
    changed = [rotate(pairs[e]) for e in (1, 5, 10)]
    receivers = {v for _, v in changed}
    return [0] + [int(pairs[e][1] in receivers) for e in edit], edit


def row_cost(bitmask, dirty):
    if bitmask == 0:
        return 0
    selected = [a for a in range(7) if bitmask & (1 << a)]
    # Explicit structural prefix plus requested suffixes, including the
    # singleton-dirty shortcut defined by the frozen protocol.
    if len(selected) == 1 and dirty[selected[0]]:
        return 4
    return 3 + sum(1 + dirty[a] for a in selected)


def total_cost(row_masks, cost_lookup):
    return sum(cost_lookup[m] for m in row_masks)


def certify(observed):
    lower = [sum(int(observed[q, a]) for q in range(12) if observed[q, a] >= 0) for a in range(7)]
    upper = [lower[a] + sum(observed[q, a] < 0 for q in range(12)) for a in range(7)]
    valid = [a for a in range(7) if all(a == b or lower[a] - upper[b] >= (1 if b < a else 0) for b in range(7))]
    if len(valid) > 1:
        raise AssertionError("multiple tie-broken certificates")
    return (valid[0] if valid else None), lower, upper


def reconstruct_bank(episode_traces):
    bank = np.full((12, 7), -1, dtype=np.int8)
    paths = [episode_traces[m] for m in ("cheapest", "gap_unit", "gap_bundle")] + episode_traces["random"]
    for path in paths:
        for step in path["trace"]:
            assert len(step["cells"]) == len(step["values"])
            for (q, a), value in zip(step["cells"], step["values"]):
                assert type(q) is int and type(a) is int and 0 <= q < 12 and 0 <= a < 7
                assert type(value) is int and value in (0, 1)
                assert bank[q, a] in (-1, value), "inconsistent cross-method paid observation"
                bank[q, a] = value
    assert (bank >= 0).all(), "trace union does not cover the bank"
    return bank


def expected_gap_purchase(observed, old, masks, lookup, bundle):
    estimates = [sum(int(observed[q, a]) if observed[q, a] >= 0 else int(old[q, a]) for q in range(12)) for a in range(7)]
    leader = min(range(7), key=lambda a: (-estimates[a], a))
    _, _, upper = certify(observed)
    challenger = min((a for a in range(7) if a != leader), key=lambda a: (-upper[a], a))
    missing = [(q, a) for q in range(12) for a in range(7) if observed[q, a] < 0]
    candidates = [x for x in missing if x[1] in (leader, challenger)]
    if not candidates:
        alternatives = {a for _, a in missing}
        challenger = min(alternatives, key=lambda a: (-upper[a], a))
        candidates = [x for x in missing if x[1] in (leader, challenger)]
    def marginal(cells):
        q = cells[0][0]
        updated = masks[q]
        for _, a in cells:
            updated |= 1 << a
        return lookup[updated] - lookup[masks[q]]
    if not bundle:
        return [min(candidates, key=lambda cell: (marginal([cell]), cell))]
    groups = [[x for x in candidates if x[0] == q] for q in range(12)]
    groups = [g for g in groups if g]
    return min(groups, key=lambda g: (-Fraction(len(g), marginal(g)), g[0][0]))


def replay_path(path, bank, dirty, method, seed=None, old=None):
    assert set(path) == {"action", "cost", "revealed", "trace", "final_observed"}
    observed = np.full((12, 7), -1, dtype=np.int8)
    masks = [0] * 12
    lookup = [row_cost(mask, dirty) for mask in range(128)]
    rng = np.random.Generator(np.random.PCG64(seed)) if method == "random" else None
    paid = 0
    steps = 0
    for step in path["trace"]:
        assert certify(observed)[0] is None, "continued after already having certificate"
        cells = [tuple(cell) for cell in step["cells"]]
        assert len(cells) in (1, 2) and len(set(cells)) == len(cells)
        assert len({q for q, _ in cells}) == 1
        if method != "gap_bundle": assert len(cells) == 1
        assert cells == sorted(cells)
        missing = [(q, a) for q in range(12) for a in range(7) if observed[q, a] < 0]
        if method == "random":
            assert cells == [missing[int(rng.integers(len(missing)))]], "random PCG64 prefix differs"
        elif method == "cheapest":
            chosen = min(missing, key=lambda x: (lookup[masks[x[0]] | (1 << x[1])] - lookup[masks[x[0]]], x))
            assert cells == [chosen], "cheapest policy differs"
        elif old is not None:
            assert cells == expected_gap_purchase(observed, old, masks, lookup, method == "gap_bundle"), "gap policy differs"
        prior = total_cost(masks, lookup)
        for (q, a), value in zip(cells, step["values"]):
            assert observed[q, a] == -1 and bank[q, a] == value
            observed[q, a] = value
            masks[q] |= 1 << a
        incremental = total_cost(masks, lookup) - prior
        assert incremental == step["incremental_cost"] and incremental > 0
        paid += incremental
        steps += 1
    action, lo, hi = certify(observed)
    assert action is not None and action == path["action"]
    assert action == min(range(7), key=lambda a: (-int(bank[:, a].sum()), a))
    assert paid == total_cost(masks, lookup) == path["cost"]
    assert int((observed >= 0).sum()) == path["revealed"]
    assert np.array_equal(observed, path["final_observed"])
    return {"cost": paid, "action": action, "revealed": int((observed >= 0).sum()), "steps": steps,
            "certificate_min_integer_margin": min(lo[action] - hi[b] - int(b < action) for b in range(7) if b != action)}


def row_pattern_oracle(bank, dirty, winner):
    """Independent one-of-patterns MILP; no prefix/discount variables as in production.

    Enumerate all 128 local reveal subsets, collapse identical certificate
    contributions to their cheapest exact-cost representative, then choose one
    pattern per question. This is an equivalent different integer formulation.
    """
    competitors = [a for a in range(7) if a != winner]
    columns = []
    for q in range(12):
        best = {}
        for mask in range(128):
            info = tuple(int(bool(mask & (1 << winner))) * int(bank[q, winner]) +
                         int(bool(mask & (1 << a))) * (1 - int(bank[q, a])) for a in competitors)
            candidate = (row_cost(mask, dirty), mask)
            if info not in best or candidate < best[info]: best[info] = candidate
        for info, (cost, mask) in sorted(best.items()):
            columns.append((q, mask, cost, info))
    matrix = lil_matrix((18, len(columns)), dtype=np.float64)
    for j, (q, _, _, info) in enumerate(columns):
        matrix[q, j] = 1
        for k, value in enumerate(info): matrix[12+k, j] = value
    lower = np.r_[np.ones(12), [12 + int(a < winner) for a in competitors]]
    upper = np.r_[np.ones(12), np.full(6, np.inf)]
    started = time.perf_counter()
    result = milp(np.array([c[2] for c in columns], dtype=float), integrality=np.ones(len(columns)),
                  bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 60.0, "mip_rel_gap": 0.0})
    assert result.success and result.x is not None and result.mip_gap == 0.0
    rounded = np.rint(result.x).astype(int)
    assert np.max(np.abs(result.x - rounded)) < 1e-7
    masks = [None] * 12
    for bit, (q, mask, _, _) in zip(rounded, columns):
        if bit:
            assert masks[q] is None
            masks[q] = mask
    assert all(mask is not None for mask in masks)
    observed = np.array([[int(bank[q, a]) if masks[q] & (1 << a) else -1 for a in range(7)] for q in range(12)])
    cost = sum(row_cost(mask, dirty) for mask in masks)
    assert certify(observed)[0] == winner and abs(result.fun - cost) < 1e-7
    return {"cost": cost, "patterns_after_exact_local_dedup": len(columns), "mip_gap": float(result.mip_gap),
            "mip_dual_bound": float(result.mip_dual_bound), "node_count": int(result.mip_node_count),
            "wall_seconds": time.perf_counter() - started,
            "independent_formulation_same_HiGHS_solver_family": True}


def review(results_path, traces_path, repo, output, ic_npz=None):
    wall, cpu = time.perf_counter(), time.process_time()
    results_path, traces_path, repo, output = map(Path, (results_path, traces_path, repo, output))
    assert not output.exists()
    pins = {"RESULTS.json": sha(results_path), "PRIVATE_TRACES.json": sha(traces_path)}
    result = json.loads(results_path.read_text(encoding="utf-8"))
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    old_banks = None
    ic_projection = []
    if ic_npz is not None:
        assert sha(ic_npz) == result["input_export_sha256"] == result["input_provenance"]["npz_sha256"]
        pins["IC_NPZ_FULL_BYTES"] = sha(ic_npz)
        allowed_metadata = {"I_A_gain", "C_A_quality", "C_B_quality", "I_A_menu_tokens",
                            "C_A_menu_tokens", "C_B_menu_tokens", "C_AB_joint_tokens", "C_AB_joint_calls",
                            "I_qid_hash", "C_qid_hash", "I_seed", "C_seed", "editable_edges", "dirty", "A_masks", "B_masks"}
        ic_projection = ["C_A_quality", "dirty", "A_masks", "B_masks", "editable_edges"]
        with np.load(ic_npz, allow_pickle=False) as archive:
            assert set(archive.files) == allowed_metadata
            projected = {name: archive[name].copy() for name in ic_projection}
        old_banks = projected["C_A_quality"]
        assert old_banks.shape == (8, 12, 7) and np.isin(old_banks, [0,1]).all()
        pairs = [(u,v) for u in range(4) for v in range(4) if u != v]
        for episode in range(8):
            shift, complement = divmod(episode, 2)
            rotated = lambda es: {((pairs[e][0]+shift)%4, (pairs[e][1]+shift)%4) for e in es}
            editable_pairs, on_pairs, changed_pairs = rotated((0,3,4,6,8,9)), rotated((0,3,6)), rotated((1,5,10))
            A = [int((pair in on_pairs) != bool(complement)) if pair in editable_pairs else 1 for pair in pairs]
            B = [0 if pair in changed_pairs else bit for pair,bit in zip(pairs,A)]
            dirty, editable = dirty_from_fixed_graph_episode(episode)
            assert np.array_equal(projected["dirty"][episode], dirty[1:])
            assert np.array_equal(projected["editable_edges"][episode], editable)
            assert np.array_equal(projected["A_masks"][episode], A)
            assert np.array_equal(projected["B_masks"][episode], B)
    source = repo / "experiments/cycle2_20261113/r3_certificate_screen"
    source_checks = {}
    for name, pin in result["source_sha256"].items():
        source_checks[name] = sha(source/name)
        assert source_checks[name] == pin
    assert sha(source/"PROTOCOL.md") == result["protocol_sha256"]
    assert len(result["episodes"]) == len(traces) == 8
    episodes = []
    total_paths = total_steps = 0
    for e, trace in enumerate(traces):
        assert set(trace) == {"cheapest", "gap_unit", "gap_bundle", "random", "oracle"}
        assert len(trace["random"]) == 256
        bank = reconstruct_bank(trace)
        dirty, editable = dirty_from_fixed_graph_episode(e)
        winner = min(range(7), key=lambda a: (-int(bank[:, a].sum()), a))
        original = result["episodes"][e]
        assert original["episode"] == e and original["bank_winner"] == winner
        assert sum(dirty) == 3 and row_cost(127, dirty) * 12 == original["full_structural_calls"] == 156
        assert (9 + 6*2) * 12 == original["old_bank_structural_calls"] == 252
        methods = {}
        for method in ("cheapest", "gap_unit", "gap_bundle"):
            replay = replay_path(trace[method], bank, dirty, method,
                                 old=None if old_banks is None else old_banks[e])
            total_paths += 1; total_steps += replay["steps"]
            assert all(replay[k] == original["methods"][method][k] for k in ("cost", "action", "revealed"))
            methods[method] = replay
        random = []
        for replica, path in enumerate(trace["random"]):
            replay = replay_path(path, bank, dirty, "random", 93173+1000*e+replica)
            total_paths += 1; total_steps += replay["steps"]; random.append(replay["cost"])
        random_mean = Fraction(sum(random), 256)
        assert float(random_mean) == original["methods"]["random"]["cost"]
        assert min(random) == original["methods"]["random"]["min"] and max(random) == original["methods"]["random"]["max"]
        assert original["methods"]["random"]["replicas"] == 256
        methods["random"] = {"mean_cost": float(random_mean), "min_cost": min(random), "max_cost": max(random), "replicas": 256,
                             "interpretation": "exact arithmetic mean of 256 frozen random traces; not exhaustive permutation expectation"}
        oracle = trace["oracle"]
        assert oracle == original["oracle"] and oracle["status"] == "OPTIMAL_CERTIFICATE_VERIFIED" and oracle["mip_gap"] == 0
        mask = np.array(oracle["mask"])
        assert mask.shape == (12,7) and np.isin(mask,[0,1]).all()
        obs = np.where(mask, bank, -1)
        certified, lo, hi = certify(obs)
        oracle_cost = sum(row_cost(sum(int(mask[q,a]) << a for a in range(7)), dirty) for q in range(12))
        assert certified == oracle["action"] == winner and oracle_cost == oracle["cost"]
        assert int(mask.sum()) == oracle["revealed"]
        independently_optimal = row_pattern_oracle(bank, dirty, winner)
        assert independently_optimal["cost"] == oracle_cost
        episodes.append({"episode": e, "winner": winner, "editable_edges": editable, "dirty": dirty,
                         "reconstructed_bank_sha256": hashlib.sha256(np.asarray(bank,dtype="<i8").tobytes()).hexdigest(),
                         "methods": methods, "oracle_cost": oracle_cost,
                         "oracle_certificate_min_integer_margin": min(lo[winner]-hi[a]-int(a<winner) for a in range(7) if a!=winner),
                         "independent_oracle": independently_optimal})
        print(json.dumps({"verified_episode": e, "paths": total_paths, "oracle_cost": oracle_cost}), flush=True)
    means = {m: sum(Fraction(ep["methods"][m]["cost"]) if m!="random" else Fraction(ep["methods"][m]["mean_cost"]) for ep in episodes)/8
             for m in ("cheapest","gap_unit","gap_bundle","random")}
    baseline = min(means[m] for m in ("cheapest","random","gap_unit"))
    oracle_mean = Fraction(sum(ep["oracle_cost"] for ep in episodes),8)
    ratio, bundle = oracle_mean/baseline, means["gap_bundle"]/baseline
    including = (252+means["gap_bundle"])/(252+baseline)
    count = sum(ep["methods"]["gap_bundle"]["cost"]<=ep["methods"]["gap_unit"]["cost"] for ep in episodes)
    nonstay = sum(ep["winner"]!=0 for ep in episodes)
    gates = {"complete_valid_oracles": True,"nonstay_winners_at_least3": nonstay>=3,
             "oracle_ratio_at_most_point80": ratio<=Fraction(4,5),"bundle_ratio_at_most_point90": bundle<=Fraction(9,10),
             "bundle_no_worse_than_gap_at_least6": count>=6,"including_bank_ratio_at_most_point97":including<=Fraction(97,100)}
    assert gates == result["gates"]
    assert {k:float(v) for k,v in means.items()} == result["mean_costs"]
    for key,value in (("oracle_mean_cost",oracle_mean),("oracle_ratio",ratio),("bundle_ratio",bundle),("including_old_bank_bundle_ratio",including)):
        assert abs(result[key]-float(value))<1e-12
    status="ELIGIBLE_FOR_REAL_COST_AND_FRESH_SERVICE_DESIGN" if all(gates.values()) else "NOT_ADMITTED_CURRENT_CERTIFICATE_ROUTE"
    assert result["status"]==status and result["S_data_used"] is False and result["new_gpu_hours"]==0
    assert not result["project_stop"] and not result["future_quality_claimed"] and not result["real_tokens_or_gpu_savings_claimed"]
    assert pins["RESULTS.json"] == sha(results_path) and pins["PRIVATE_TRACES.json"] == sha(traces_path)
    if ic_npz is not None: assert pins["IC_NPZ_FULL_BYTES"] == sha(ic_npz)
    out={"status":"PASS_INDEPENDENT_R3_TRACE_AND_ORACLE_REVIEW","original_decision":status,"input_sha256":pins,
         "review_source_sha256":sha(Path(__file__)),"production_source_hashes_verified":source_checks,
         "protocol_sha256":result["protocol_sha256"],"trajectories_replayed":total_paths,"purchase_steps_replayed":total_steps,
         "random_trace_count":8*256,"independent_row_pattern_oracles":8,"episodes":episodes,
         "means":{k:float(v) for k,v in means.items()},"oracle_mean":float(oracle_mean),"oracle_ratio":float(ratio),
         "bundle_ratio":float(bundle),"including_old_bank_bundle_ratio":float(including),"bundle_no_worse_count":count,
         "nonstay_winner_count":nonstay,"gates":gates,
         "scope":{"new_GPU":0,"S_read":False,"question_text_read":False,"raw_scoring_rechecked":False,
                  "IC_npz_read":ic_npz is not None,"IC_exact_keys_decoded":ic_projection,
                  "gap_priority_recomputed":old_banks is not None,
                  "gap_priority_limit":None if old_banks is not None else "Old C_A bank absent; priority not checked.",
                  "oracle_limit":"Independent row-pattern formulation but same scipy/HiGHS solver family; not formal theorem-prover certification.",
                  "random_limit":"256 seeded realizations, not exact all-permutations expectation.",
                  "cost_limit":"Structural calls only; old A bank adds252 equally; no real tokens/GPU/latency/future-service claim."},
         "cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-wall}
    output.mkdir(parents=True,exist_ok=False)
    (output/"REVIEW_RESULTS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value))+"\n",encoding="utf-8")
    print(json.dumps({k:out[k] for k in ("status","original_decision","trajectories_replayed","purchase_steps_replayed","oracle_ratio","bundle_ratio","bundle_no_worse_count","nonstay_winner_count","cpu_seconds","wall_seconds")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--traces",required=True)
    p.add_argument("--repo",required=True);p.add_argument("--output",required=True)
    p.add_argument("--ic-npz")
    a=p.parse_args();review(a.results,a.traces,a.repo,a.output,a.ic_npz)
