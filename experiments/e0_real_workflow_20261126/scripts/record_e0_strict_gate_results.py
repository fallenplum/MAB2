from pathlib import Path
import json
from datetime import datetime,timezone
root=Path('experiments/e0_real_workflow_20261126'); now=datetime.now(timezone.utc).isoformat()
def load(n): return json.loads((root/n).read_text(encoding='utf-8'))
def save(n,o): (root/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
g=load('GATE_RECEIPTS.json'); g['status']='INCONCLUSIVE_STRICT_GATE_ACTION_RISK'; g['gates']['D_strict_gate']='PASS_DEVELOPMENT_ACTUAL'; g['gates']['strong_flat']='NOT_RUN'; g['gates']['stale_quality']='NOT_RUN'; g['next']='Complete strong-flat/RAG and stale quality baselines before any positive mechanism claim.'; save('GATE_RECEIPTS.json',g)
d=load('E0_DECISION.json'); d['status']='INCONCLUSIVE_STRICT_GATE_ACTION_RISK'; d['conclusion']='Full development run and actual strict-gate baselines completed. Strict gates save substantial verifier cost but show more action errors than full cold on the locked development groups.'; d['recommendation']='Finish strong flat/RAG and stale baselines. If their quality ordering confirms this pattern, freeze strict certificate optimization and report the negative mechanism result.'; save('E0_DECISION.json',d)
r=load('results/E0_D_BASELINE_CERTIFICATE.json'); r['status']='DEVELOPMENT_FULL_WITH_ACTUAL_STRICT_GATE'; r['strict_gate_summary']='results/E0_STRICT_GATE_SUMMARY.json'; r['baseline_comparison']='results/E0_BASELINE_COMPARISON_SUMMARY.json'; r['strict_gate_action_errors']=17; r['full_cold_action_errors']=7; save('results/E0_D_BASELINE_CERTIFICATE.json',r)
for name,status in [('results/E0_E_HEADROOM.json','NOT_RUN_STRONG_BASELINE_PENDING'),('results/E0_A_WORKFLOW_VALUE.json','DEVELOPMENT_FULL_NO_FLAT_BASELINE'),('results/E0_B_INTERFACE_STABILITY.json','DEVELOPMENT_FULL_RAW_CANONICAL_PARTIAL')]:
 o=load(name); o['status']=status; save(name,o)
(root/'ISSUE_REGISTRY.jsonl').open('a',encoding='utf-8').write(json.dumps({'id':'E0-009','severity':'MAJOR','status':'OPEN','issue':'Actual cold and canonical strict-gate baselines reduced tokens but caused 17/30 action errors versus 7/30 for full cold on development groups.'},ensure_ascii=False)+'\n')
log=root/'PROGRESS_LOG.jsonl'; log.write_text(log.read_text(encoding='utf-8')+''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in [
 {'timestamp_utc':now,'event':'E0_STRICT_GATE_COLD_32','status':'PASS_E0_STRICT_GATE_DEV','input_tokens':33861,'decision_errors':17},
 {'timestamp_utc':now,'event':'E0_STRICT_GATE_CANONICAL_32','status':'PASS_E0_STRICT_GATE_DEV','input_tokens':18448,'decision_errors':17},
 {'timestamp_utc':now,'event':'E0 strict gate interpretation','status':'INCONCLUSIVE_ACTION_RISK','next':'strong-flat/stale baselines'}
]),encoding='utf-8')
print('recorded')

