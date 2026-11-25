import json
from pathlib import Path
r=json.loads(Path('experiments/e0_real_workflow_20261126/results/E0_BASELINE_REPLAY.json').read_text(encoding='utf-8'))
for m in ['new_cold','new_warm','canonical_full','strict_gate_cold','canonical_strict_gate']:
 ds=[x[m] for x in r['rows'] if x[m]['status']!='INVALID']
 costs=[x['costs'] for x in r['rows']]
 key={'new_cold':'new_cold_input_tokens','new_warm':'new_warm_input_tokens','canonical_full':'canonical_full_input_tokens_replay','strict_gate_cold':'strict_gate_cold_input_tokens_replay','canonical_strict_gate':'canonical_strict_gate_input_tokens_replay'}[m]
 print(m,'certified',sum(x['status']=='CERTIFIED' for x in ds),'unresolved',sum(x['status']=='UNRESOLVED' for x in ds),'tokens',sum(c[key] for c in costs))

