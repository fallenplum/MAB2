"""Create frozen relay-interface variants from an existing density panel."""

from __future__ import annotations

import json
from pathlib import Path


VARIANTS = {
    "short": "Protocol v2.",
    "semantic": (
        "You are a reviewer. Re-evaluate this relay under the updated graph "
        "context and return one hypothesis token."
    ),
}


def make_variant(input_path: Path, output_path: Path, suffix: str) -> None:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    for row in rows:
        relay = [""] * 4
        for index in row.get("interface_drift_rows", ()):
            relay[int(index)] = suffix
        row["new_relay_interface"] = relay
    output_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def main(root: Path) -> None:
    for variant, suffix in VARIANTS.items():
        for density in (1, 2, 4):
            for task in ("qasc", "sciq", "commonsense"):
                source = root / f"{task}_interface_drift_rows{density}_INPUT.json"
                target = root / f"{task}_interface_{variant}_rows{density}_INPUT.json"
                make_variant(source, target, suffix)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))

