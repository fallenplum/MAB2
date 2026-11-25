from pathlib import Path
import hashlib, json
from datetime import datetime, timezone

root=Path('experiments/e0_real_workflow_20261126')
source=root/'external/factscore_official/data/data/labeled/InstructGPT.jsonl'
salt='mab2-e0-20261126-factscore-topic-v1'
rows=[]
for row_index,line in enumerate(source.open(encoding='utf-8')):
    obj=json.loads(line)
    topic=obj['topic']
    digest=hashlib.sha256((salt+'|'+topic).encode()).hexdigest()
    rows.append({'group_id':'e0g_'+digest[:16],'topic_sha256':hashlib.sha256(topic.encode()).hexdigest(),'row_index':row_index,'source_file':'data/labeled/InstructGPT.jsonl','output_chars':len(obj.get('output',''))})
selected=sorted(rows,key=lambda x:x['group_id'])[:32]
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
manifest={'schema':'mab2_e0_topic_selection_v1','status':'LOCKED_DEVELOPMENT_GROUPS','selected_from':'official FActScore labeled data only','source_model':'InstructGPT','selection_rule':'sort by SHA256(salt|topic), take first 32 unique topics','salt':salt,'source_file_sha256':source_sha,'group_count':len(selected),'groups':selected,'labels_excluded_from_manifest':True,'online_gold_access':False,'generated_at_utc':datetime.now(timezone.utc).isoformat()}
(root/'TOPIC_SELECTION_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source_sha256':source_sha,'group_count':len(selected),'first_groups':selected[:5]},ensure_ascii=False,indent=2))

