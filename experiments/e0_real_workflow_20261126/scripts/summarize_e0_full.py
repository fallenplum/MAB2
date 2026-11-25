import json
from pathlib import Path
p=Path('experiments/e0_real_workflow_20261126/results/E0_FULL_32.json'); o=json.loads(p.read_text(encoding='utf-8'))
for n in ['old_events','new_cold_events','new_warm_events']:
 ev=o[n]; print(n,'events',len(ev),'cache_hits',sum(e['cache_hit'] for e in ev),'input_tokens',sum(e['input_tokens'] for e in ev),'output_tokens',sum(e['output_tokens'] for e in ev),'latency',round(sum(e['latency_seconds'] for e in ev),2),'unresolved',sum(e['label'] is None for e in ev))
for n in ['old_decisions','new_cold_decisions','new_warm_decisions']:
 d=o[n]; print(n,'groups',len(d),'invalid',sum(v['status']=='INVALID' for v in d.values()),'certified',sum(v['status']=='CERTIFIED' for v in d.values()),'unresolved',sum(v['status']=='UNRESOLVED' for v in d.values()))

