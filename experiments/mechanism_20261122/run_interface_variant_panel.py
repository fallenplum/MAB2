"""Run the fixed-model interface-variant panel on assigned CUDA devices.

This runner only executes the frozen pretrained models. It does not fine-tune a
backbone or read sealed labels during policy selection.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


MODEL_NAMES = ("qwen05", "qwen14", "llama3")


def jobs(root: Path, output_root: Path):
    paths = {
        "qwen05": "/home/chen/KDBPT/KD/models/Qwen-0.5B",
        "qwen14": "/home/chen/KDBPT/KD/models/Qwen-14B",
        "llama3": "/data/Fei_data/llama-3-8B/Meta-Llama-3-8B-Instruct",
    }
    worker = root / "experiments" / "mechanism_20261122" / "conditional_mcqa_worker_interface_drift.py"
    for variant in ("short", "semantic"):
        for density in (1, 2, 4):
            for task in ("qasc", "sciq", "commonsense"):
                for name in MODEL_NAMES:
                    yield name, paths[name], worker, root / f"{task}_interface_{variant}_rows{density}_INPUT.json", output_root / f"{task}_{variant}_rows{density}_{name}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("python", help="Python executable containing torch and transformers")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 7])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_jobs = list(jobs(args.root, args.output_root))
    for start in range(0, len(all_jobs), len(args.gpus)):
        running = []
        for gpu, (name, model, worker, input_path, output_path) in zip(args.gpus, all_jobs[start:]):
            if output_path.exists() and output_path.stat().st_size > 1000:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("MAB2_MICROBATCH", "2")
            log_path = output_path.with_suffix(".log")
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen([args.python, str(worker), model, str(input_path), str(output_path)], env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, log))
        for process, log in running:
            process.wait()
            log.close()


if __name__ == "__main__":
    main()

