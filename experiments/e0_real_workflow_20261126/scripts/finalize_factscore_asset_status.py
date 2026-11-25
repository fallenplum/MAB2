from pathlib import Path
import json
from datetime import datetime,timezone
root=Path('experiments/e0_real_workflow_20261126'); now=datetime.now(timezone.utc).isoformat()
def load(n): return json.loads((root/n).read_text(encoding='utf-8'))
def save(n,o): (root/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
inv=load('ASSET_INVENTORY.json'); inv['status']='OFFICIAL_ASSETS_AVAILABLE_SMOKE_COMPLETE'; inv['knowledge_source']={'status':'PASS_OFFICIAL_DB_READONLY','path':'E:\\图结构演化\\data\\raw\\factscore\\enwiki-20230401.db','bytes':21924696064,'document_count':6187531,'selected_smoke_topics_verified':4}; inv['server']['status']='NOT_NEEDED_FOR_LOCAL_SMOKE'; save('ASSET_INVENTORY.json',inv)
receipt=load('FACTSCORE_DOWNLOAD_RECEIPT.json'); receipt['knowledge_source']={'status':'DOWNLOADED_SQLITE_VALID','file':'enwiki-20230401.db','path':'E:\\图结构演化\\data\\raw\\factscore\\enwiki-20230401.db','bytes':21924696064,'document_count':6187531,'selected_smoke_topics_verified':4,'sha256':'NOT_COMPLETED_AFTER_SIZE_VALIDATION'}; save('FACTSCORE_DOWNLOAD_RECEIPT.json',receipt)
man=load('DATA_MANIFEST.json'); man['official_knowledge_source']='AVAILABLE_LOCAL_E_DRIVE'; man['knowledge_source_path']='E:\\图结构演化\\data\\raw\\factscore\\enwiki-20230401.db'; save('DATA_MANIFEST.json',man)
g=load('GATE_RECEIPTS.json'); g['gates']['knowledge_source']='PASS_OFFICIAL_DB_READONLY'; g['status']='INCONCLUSIVE_SMOKE_ONLY_FULL_E0_OPEN'; g['next']='Run all remaining locked groups and complete F/D baseline ledger.'; save('GATE_RECEIPTS.json',g)
d=load('E0_DECISION.json'); d['status']='INCONCLUSIVE_SMOKE_ONLY'; d['conclusion']='The official FActScore code, labeled data, and knowledge source are available. Four-group smoke passed after a parser fix, while the full 32-group E0 evaluation remains incomplete.'; d['recommendation']='Run the remaining 28 locked groups with the same model, database, ledger, and posthoc scorer. Do not claim full E0 or ACL readiness yet.'; save('E0_DECISION.json',d)
log=root/'PROGRESS_LOG.jsonl'; log.write_text(log.read_text(encoding='utf-8')+json.dumps({'timestamp_utc':now,'event':'Knowledge source download and SQLite validation','status':'PASS','bytes':21924696064,'document_count':6187531},ensure_ascii=False)+'\n',encoding='utf-8')
print('updated')

