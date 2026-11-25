from pathlib import Path
import json
root=Path('experiments/e0_real_workflow_20261126/external/factscore_official/data/data/labeled')
for name in ['InstructGPT.jsonl','ChatGPT.jsonl','PerplexityAI.jsonl']:
 n=0; none=0; facts=0
 for line in (root/name).open(encoding='utf-8'):
  row=json.loads(line); n+=1
  if not row.get('annotations'): none+=1
  facts += sum(len(a.get('model-atomic-facts',[])) for a in (row.get('annotations') or []))
 print(name,n,'annotations_none',none,'facts',facts)

