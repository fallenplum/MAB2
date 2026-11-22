import json
import sys
from pathlib import Path


src = Path(sys.argv[1])
out = Path(sys.argv[2])
changed = tuple(int(v) for v in sys.argv[3].split(",") if v)
rows = json.loads(src.read_text(encoding="utf-8"))
for row in rows:
    suffixes = ["" for _ in range(4)]
    for index in changed:
        suffixes[index] = (
            "Relay protocol v2. Return one hypothesis token after applying the evidence."
        )
    row["new_relay_interface"] = suffixes
    row["interface_drift_rows"] = list(changed)
out.write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True), encoding="utf-8")

