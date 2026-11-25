from pathlib import Path
import json
from datetime import datetime,timezone
root=Path('experiments/e0_real_workflow_20261126'); now=datetime.now(timezone.utc).isoformat()
def load(n): return json.loads((root/n).read_text(encoding='utf-8'))
def save(n,o): (root/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
g=load('GATE_RECEIPTS.json'); g['gates']['workflow_trace']='PASS_E0_SMOKE_DEV_4_GROUPS'; g['gates']['D_smoke']='PASS_DEVELOPMENT_ONLY'; g['status']='INCONCLUSIVE_FULL_E0_OPEN'; g['next']='Run all locked groups after smoke review; retain invalid prior run and parser fix.'; save('GATE_RECEIPTS.json',g)
d=load('E0_DECISION.json'); d['status']='INCONCLUSIVE_SMOKE_ONLY'; d['conclusion']='Official assets and the first four-group E0 smoke are available, but full 32-group F/D evidence and independent confirmation are not complete.'; d['smoke_artifact']='results/E0_SMOKE_4_SUMMARY.json'; d['recommendation']='Review the smoke ledger and then run the remaining locked groups under the same protocol. Do not treat the four-group development result as an ACL claim.'; save('E0_DECISION.json',d)
r=load('results/E0_D_BASELINE_CERTIFICATE.json'); r['status']='DEVELOPMENT_SMOKE_ONLY'; r['smoke_summary']='results/E0_SMOKE_4_SUMMARY.json'; r['groups_completed']=4; r['groups_remaining']=28; r['quality_metric']='posthoc fact-label accuracy only'; r['note']='No full E0 baseline comparison or independent confirmation.'; save('results/E0_D_BASELINE_CERTIFICATE.json',r)
issues=root/'ISSUE_REGISTRY.jsonl'; issues.write_text(issues.read_text(encoding='utf-8')+'{"id":"E0-008","severity":"MAJOR","status":"CLOSED_AFTER_RERUN","issue":"Initial 4-group smoke parser misread split token UN SUPPORTED. The run was excluded and rerun with normalized verdict parsing."}\n',encoding='utf-8')
log=root/'PROGRESS_LOG.jsonl'; log.write_text(log.read_text(encoding='utf-8')+''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in [
 {'timestamp_utc':now,'event':'Official knowledge source downloaded and SQLite checked','status':'PASS','document_count':6187531,'groups_smoke_checked':4},
 {'timestamp_utc':now,'event':'E0_SMOKE_4','status':'INVALID_IMPLEMENTATION_BUG','failure':'split unsupported parser'},
 {'timestamp_utc':now,'event':'E0_SMOKE_4_RERUN_001','status':'PASS_E0_SMOKE_DEV','groups':4,'model':'Qwen2.5-3B-Instruct'},
 {'timestamp_utc':now,'event':'E0 stage decision','status':'INCONCLUSIVE_SMOKE_ONLY','next':'run remaining 28 locked groups'}
]),encoding='utf-8')
print('updated')

