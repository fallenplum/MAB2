"""Fresh input-only selection. Never decodes answer or supporting-facts columns."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';old=root/'artifacts/private/cycle2_20261113';out=root/'artifacts/private/cycle3_20261116/annotation_input_v1'
sys.path.insert(0,str(r/'experiments/cycle2_20261113/hotpot_selection_v1'));import select_hotpot as h
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=old/'hotpot_selection_preparation/SOURCE_DESCRIPTOR_FROZEN.json'
assert sha(source)=='5cf1902c112089f8b787dc8bb1a6a46df285003fb9318e0134082f1c5d148456'
exp_path=root/'artifacts/private/cycle3_20261116/EXPOSURES_UNION_AFTER_FORMAL6_SELECTION.json';exp=json.loads(exp_path.read_bytes())
ids=set(exp['ids']);groups=set(exp['question_group_hashes']);contents=set(exp['content_hashes'])
poolpath=old/'closure_input_pool_v1/POOL.jsonl'
for line in poolpath.read_bytes().splitlines():
    v=h.validate_row(json.loads(line));ids.add(v['id']);groups.add(h.question_group_sha256(v['question']));contents.add(h.content_sha256(v))
paths=h._source_preflight(json.loads(source.read_bytes()),source)
by_group={};excluded=set();seen=set()
for raw in h.iter_allowed_rows(paths):
    v=h.validate_row(raw);assert v['id'] not in seen;seen.add(v['id']);g=h.question_group_sha256(v['question']);by_group.setdefault(g,[]).append(v)
    if v['id'] in ids or g in groups or h.content_sha256(v) in contents:excluded.add(g)
eligible=sorted([min(v,key=lambda x:x['id'].encode()) for g,v in by_group.items() if g not in excluded and all(len(x['context']['title'])==10 for x in v)],key=lambda x:x['id'].encode())
indices=np.random.Generator(np.random.PCG64(261116701)).permutation(len(eligible))[:96]
selected=[eligible[int(i)] for i in indices];assert len(selected)==96 and all(len(x['context']['title'])==10 for x in selected)
out.mkdir(parents=True,exist_ok=False)
with (out/'QUESTIONS.jsonl').open('xb') as f:
    for v in selected:f.write(h.canonical_bytes(v))
rows=[]
for i,v in enumerate(selected):rows.append(dict(index=i,id=v['id'],input_sha256=hashlib.sha256(h.canonical_bytes(v)).hexdigest(),question_group_sha256=h.question_group_sha256(v['question']),content_sha256=h.content_sha256(v),removed_upstream='U' if i%2==0 else 'V'))
write(out/'SELECTION_MANIFEST.json',dict(questions=rows,sampling_seed=261116701,labels_projected=False,source_sha256=sha(source),exclusions_union_sha256=sha(exp_path),prior1024pool_sha256=sha(poolpath),known_exposure_groups=len(groups),eligible_groups=len(eligible),unknown_cross_project_exposure='UNKNOWN',official_validation_test_opened=False,selector_sha256=sha(Path(__file__))))
write(out/'RECEIPT.json',dict(selected_questions=96,questions_sha256=sha(out/'QUESTIONS.jsonl'),selection_sha256=sha(out/'SELECTION_MANIFEST.json'),no_labels=True,eligible_groups=len(eligible),prior_exclusion_groups=len(groups),source_rows=len(seen),GPU_hours=0))
print((out/'RECEIPT.json').read_text())
