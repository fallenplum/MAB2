import json
import sys
from pathlib import Path


base = Path(sys.argv[1])
out = Path(sys.argv[2])
changed = [int(v) for v in sys.argv[3].split(",") if v]
manifest = {
    "schema": "m60_conditional_interface_drift_manifest_v1",
    "panels": {},
    "workers": [0, 1, 2],
    "changed_rows": changed,
    "graph_manifest": {
        "edited_edges": [f"source_to_relay_{i}" for i in changed],
        "edge_to_relay_descendants": {
            f"source_to_relay_{i}": [i] for i in changed
        },
        "relay_interface_change": "fixed_source_distribution_protocol_v2",
    },
    "output": str(out / "M60_INTERFACE_DRIFT_RESULT.json"),
    "interpretation": (
        "Interface drift stress panel. Source vectors and terminal prompts are fixed. "
        "New relay prompts append a frozen protocol-v2 suffix on graph-descendant rows. "
        "Gold labels are used only by the replay scorer for reporting."
    ),
}
for task in ("qasc", "sciq", "commonsense"):
    folder = out / task
    folder.mkdir(parents=True, exist_ok=True)
    manifest["panels"][task] = str(folder)
    sealed_src = base / f"m60_conditional_server_{task}" / "SEALED.json"
    (folder / "SEALED.json").write_bytes(sealed_src.read_bytes())
    input_src = base / f"{task}_interface_drift_rows{len(changed)}_INPUT.json"
    (folder / "INPUT_PATH.txt").write_text(str(input_src), encoding="utf-8")
print(json.dumps(manifest, indent=2))

