from pathlib import Path
import json
from datetime import datetime,timezone
root=Path('experiments/e0_real_workflow_20261126'); now=datetime.now(timezone.utc).isoformat()
def load(n): return json.loads((root/n).read_text(encoding='utf-8'))
def save(n,o): (root/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
summary=load('results/E0_FULL_32_SUMMARY.json')
g=load('GATE_RECEIPTS.json'); g['status']='INCONCLUSIVE_FULL_DEVELOPMENT_PARTIAL_BASELINES'; g['gates']['workflow_trace']='PASS_E0_FULL_DEVELOPMENT_32_GROUPS'; g['gates']['D_smoke']='PASS_DEVELOPMENT_ONLY'; g['gates']['D_full']='PASS_DEVELOPMENT_PARTIAL_BASELINES'; g['gates']['F_baselines']='NOT_RUN'; g['next']='Run matched canonical-cache, stale, strict-gate and strong-flat baselines.'; save('GATE_RECEIPTS.json',g)
d=load('E0_DECISION.json'); d['status']='INCONCLUSIVE_DIRECTIONAL_DEVELOPMENT'; d['conclusion']='The full 32-group development run completed. Raw reuse is limited to 70 of 1073 new events, with one valid decision change and two invalid new decompositions. This is not enough to claim graph-conditional reuse benefit.'; d['recommendation']='Complete matched canonical, stale, strict-gate and strong-flat baselines before deciding whether the task supports further graph optimization.'; save('E0_DECISION.json',d)
for name,status,extra in [
 ('results/E0_A_WORKFLOW_VALUE.json','DEVELOPMENT_FULL_RUN_NO_QUALITY_BASELINE',{'groups_completed':32,'note':'No strong flat/RAG comparison yet'}),
 ('results/E0_B_INTERFACE_STABILITY.json','DEVELOPMENT_FULL_RUN_RAW_AND_CANONICAL_PREDIAGNOSTIC',{'summary':'results/E0_FULL_32_SUMMARY.json'}),
 ('results/E0_C_PROPAGATION_JOINT_RECOVERY.json','NOT_APPLICABLE_NO_NONTRIVIAL_SHARED_ANCESTOR_GRAPH',{'note':'Current fact-verification workflow is topic fan-out; no shared dynamic ancestor was established'}),
 ('results/E0_D_BASELINE_CERTIFICATE.json','DEVELOPMENT_FULL_RUN_PARTIAL_BASELINES',{'summary':'results/E0_FULL_32_SUMMARY.json','valid_groups':summary['groups_valid_for_pairwise_decision'],'invalid_groups':summary['invalid_groups']}),
 ('results/E0_E_HEADROOM.json','NOT_RUN_STRONG_BASELINE_PENDING',{'note':'No clairvoyant headroom until matched baselines are complete'})
]:
 o=load(name); o['status']=status; o.update(extra); save(name,o)
log=root/'PROGRESS_LOG.jsonl'; log.write_text(log.read_text(encoding='utf-8')+''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in [
 {'timestamp_utc':now,'event':'E0_FULL_32','status':'PASS_E0_SMOKE_DEV','groups':32,'valid_pairwise_groups':summary['groups_valid_for_pairwise_decision']},
 {'timestamp_utc':now,'event':'E0_FULL_32 interpretation','status':'INCONCLUSIVE_DIRECTIONAL_DEVELOPMENT','raw_cache_hits':summary['new_warm']['cache_hits'],'decision_changes':len(summary['decision_changes'])},
 {'timestamp_utc':now,'event':'E0 next baseline gate','status':'OPEN','next':'canonical/stale/strict-gate/strong-flat'}
]),encoding='utf-8')
print('recorded')

