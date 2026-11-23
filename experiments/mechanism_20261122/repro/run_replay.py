"""Run the public CPU replay pipeline from a clean checkout."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MECHANISM_DIR = ROOT.parent
sys.path.insert(0, str(MECHANISM_DIR))
from conditional_real_score import run as run_real_score  # noqa: E402

DISPLAY_METHODS = {
    "stale": "stale",
    "warm_reweighted": "reweighted old-history audit / exact warm cache",
    "source_only_ctdr": "CTDR transport only",
    "tcr_token": "CTDR transport + selective ranking",
    "cold_token": "full restart",
    "max_uncertainty": "max-uncertainty",
    "periodic_recheck": "periodic",
    "random_recheck": "random",
}


def prepare_inputs() -> None:
    from prepare_inputs import build
    build()


def run_pipeline(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        result = run_real_score(manifest)
    finally:
        os.chdir(old_cwd)
    aggregate = result["aggregate"]
    table = []
    for method, label in DISPLAY_METHODS.items():
        table.append({"method": label, **aggregate[method]})
    # Declared information-theoretic lower bound. It assumes an oracle already
    # knows the post-edit decision, so it is not an executable baseline.
    table.append({
        "method": "oracle lower bound",
        "n": aggregate["cold_token"]["n"],
        "action_error": 0.0,
        "exact_set": 1.0,
        "false_drop": 0.0,
        "false_keep": 0.0,
        "queries": 0.0,
        "tokens": 0.0,
    })
    public = {
        "schema": "m60_clean_replay_v1",
        "stages": ["prepare inputs", "run worker outputs", "score old transcript",
                   "run post-edit baselines", "run CTDR", "generate M60 table"],
        "manifest": str(manifest_path.name),
        "table": table,
        "raw_result": result,
    }
    output_path = ROOT / manifest.get("output", "fixture_result.json")
    output_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "fixture_manifest.json")
    parser.add_argument("--prepare", action="store_true", help="regenerate the public fixture first")
    args = parser.parse_args()
    if args.prepare:
        prepare_inputs()
    result = run_pipeline(args.manifest.resolve())
    print(json.dumps(result["table"], indent=2))


if __name__ == "__main__":
    main()

