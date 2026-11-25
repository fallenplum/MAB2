from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
root=Path('experiments/e0_real_workflow_20261126')
files={}
for p in sorted(root.rglob('*')):
    excluded = {'E0_SMOKE_4.json','E0_SMOKE_4_RERUN_001.json','E0_SMOKE_4_POSTHOC.json','E0_FULL_32.json','E0_STRICT_GATE_COLD_32.json','E0_STRICT_GATE_CANONICAL_32.json','inspect_smoke_json.py','debug_smoke_json.py','inspect_factscore_db_download.py','parse_factscore_db_download.py','probe_factscore_download.py'}
    if p.is_file() and p.name!='ARTIFACT_SHA256.json' and '__pycache__' not in p.parts and 'external' not in p.parts and p.name not in excluded and not p.name.startswith('amend_'):
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        files[str(p.relative_to(root)).replace('\\','/')]=h
(root/'ARTIFACT_SHA256.json').write_text(json.dumps({'schema':'mab2_e0_artifact_sha256_v2','generated_at_utc':datetime.now(timezone.utc).isoformat(),'external_raw_data_excluded':True,'files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('hashed',len(files))

