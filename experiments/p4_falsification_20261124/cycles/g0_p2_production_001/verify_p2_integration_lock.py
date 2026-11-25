from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(lock_path: Path, cycle_dir: Path, p2_dir: Path, root_prompt: Path, input_path: Path, model_dir: Path | None = None) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    failures = []
    for name, expected in lock["cycle_files_sha256"].items():
        path = cycle_dir / name
        if not path.is_file() or sha(path).lower() != expected.lower():
            failures.append("cycle:" + name)
    for name, expected in lock["p2_files_sha256"].items():
        path = p2_dir / name
        if not path.is_file() or sha(path).lower() != expected.lower():
            failures.append("p2:" + name)
    if sha(root_prompt).lower() != lock["authoritative_prompt_sha256"].lower():
        failures.append("authoritative_prompt")
    if sha(input_path).lower() != lock["input_sha256"].lower():
        failures.append("input")
    if model_dir is not None:
        for name, expected in lock["checkpoint_sha256"].items():
            path = model_dir / name
            if not path.is_file() or sha(path).lower() != expected.lower():
                failures.append("checkpoint:" + name)
    return {
        "status": "PASS_LOCK" if not failures else "FAIL_LOCK",
        "failures": failures,
        "prompt_sha256": lock["authoritative_prompt_sha256"],
        "input_sha256": lock["input_sha256"],
        "cycle_file_count": len(lock["cycle_files_sha256"]),
        "p2_file_count": len(lock["p2_files_sha256"]),
        "checkpoint_count_checked": len(lock["checkpoint_sha256"]) if model_dir is not None else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    result = verify(args.lock, args.cycle, args.p2, args.prompt, args.input, args.model)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS_LOCK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
