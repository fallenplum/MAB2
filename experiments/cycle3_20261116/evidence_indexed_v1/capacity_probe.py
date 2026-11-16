"""CPU tokenization only; conservative full-document bounds, no selection or labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1')
import argparse,json,hashlib,sys
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--questions',required=True);a.add_argument('--output',required=True);x=a.parse_args()
plan=json.loads(Path(x.plan).read_bytes())
for p in reversed(plan['import_paths']):sys.path.insert(0,p)
from transformers import AutoTokenizer
import evidence_operator as e
import exogenous_path_operator as b
rows=[json.loads(s) for s in Path(x.questions).read_bytes().splitlines()]
records=[]
for name,model in plan['models'].items():
    t=AutoTokenizer.from_pretrained(model['path'],local_files_only=True,trust_remote_code=False)
    for i,r in enumerate(rows):
        p={k:r[k] for k in ['id','question','context']};docs=e.documents(p,range(10))
        sizes={}
        for mode,instruction in [('selector',e.SELECT),('terminal',b.FINAL_INSTRUCTION)]:
            messages=b._messages(p,docs,[],instruction)
            sizes[mode]=len(t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        for mode in ['INDEPENDENT3_U','INDEPENDENT3_M']:
            sizes[mode]=len(t.apply_chat_template(b.build_full_context_messages(p,mode),tokenize=True,add_generation_prompt=True))+1024
        records.append(dict(index=i,model=name,bounds=sizes,admitted=max(sizes.values())+256+64<=8192))
out=Path(x.output);out.mkdir(parents=True,exist_ok=False)
result=dict(status='PASS_CONSERVATIVE_TOKENIZER_BOUND' if all(r['admitted'] for r in records) else 'FAIL_INPUT_CAPACITY',rows=len(rows),checks=len(records),maximum_bound=max(max(r['bounds'].values()) for r in records),failures=[r for r in records if not r['admitted']],new_GPU_hours=0,labels_read=False,questions_sha256=hashlib.sha256(Path(x.questions).read_bytes()).hexdigest(),limitation='Subset tokenization is not strictly monotone; exact per-call admission remains mandatory')
(out/'RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
