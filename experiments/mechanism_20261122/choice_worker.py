"""Registered constrained next-token interface. Inputs exclude correctness labels."""
import argparse,hashlib,itertools,json,os,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('input');p.add_argument('output');a=p.parse_args()
    start=time.monotonic()
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    inputs=json.loads(Path(a.input).read_bytes());assert len(inputs)==96 and all(set(r)=={'key','task','stem','options'} for r in inputs)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    letters='ABCD';ids=[tok.encode(c,add_special_tokens=False) for c in letters];assert all(len(v)==1 for v in ids) and len(set(x[0] for x in ids))==4,ids
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    choices=list(itertools.product('-ABCD',repeat=2))
    with Path(a.output).open('x',encoding='utf8') as f,torch.inference_mode():
        for row in inputs:
            q=row['stem']+'\n'+'\n'.join(k+'. '+v for k,v in zip(letters,row['options']))
            messages=[q+'\nChoose your own answer. Return only A, B, C or D.']
            for left,right in choices:
                hints=[f'Adviser {k}: {v}' for k,v in [(1,left),(2,right)] if v!='-']
                messages.append('\n'.join([q]+hints+['Choose your final answer. Return only A, B, C or D.']))
            enc=[tok.apply_chat_template([{'role':'user','content':m}],tokenize=True,add_generation_prompt=True) for m in messages]
            probs=[]
            for k in range(0,len(enc),13):
                batch=tok.pad({'input_ids':enc[k:k+13]},return_tensors='pt').to('cuda:0');mask=batch['attention_mask'];pos=(mask.long().cumsum(1)-1).clamp_min(0)
                logits=model(**batch,position_ids=pos,use_cache=False).logits[:,-1,[v[0] for v in ids]].float()
                probs.extend(logits.softmax(-1).cpu().tolist())
            f.write(json.dumps(dict(key=row['key'],probs=probs,input_tokens=list(map(len,enc)),elapsed=time.monotonic()-start))+'\n');f.flush()
    print(json.dumps(dict(rows=len(inputs),elapsed=time.monotonic()-start,torch=torch.__version__,transformers=transformers.__version__,letter_ids=ids)),flush=True)
if __name__=='__main__':main()
