from pathlib import Path
import json,hashlib
root=Path('experiments/e0_real_workflow_20261126'); sel=json.loads((root/'TOPIC_SELECTION_MANIFEST.json').read_text()); ids={x['group_id'] for x in sel['groups']}; salt=sel['salt']
def gid(t): return 'e0g_'+hashlib.sha256((salt+'|'+t).encode()).hexdigest()[:16]
for name in ['InstructGPT.jsonl','ChatGPT.jsonl','PerplexityAI.jsonl']:
 bad=[]
 for i,line in enumerate((root/'external/factscore_official/data/data/labeled'/name).open(encoding='utf-8')):
  row=json.loads(line); g=gid(row['topic'])
  if g in ids and not row.get('annotations'): bad.append((g,i,row['topic']))
 print(name,'selected_missing',len(bad),bad)

