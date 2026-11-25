from pathlib import Path
import json
root=Path('experiments/e0_real_workflow_20261126'); r=json.loads((root/'results/E0_BASELINE_REPLAY.json').read_text(encoding='utf-8'))
for m in ['canonical_full','strict_gate_cold','canonical_strict_gate','stale_action']:
 changed=[]
 for row in r['rows']:
  a=row['new_cold'].get('decision'); b=row[m].get('decision') if isinstance(row[m],dict) else row[m].get('decision')
  if row['new_cold']['status']!='INVALID' and row[m].get('status')!='INVALID' and a!=b: changed.append(row['group_id'])
 print(m,'changed_vs_cold',len(changed),changed[:10])

