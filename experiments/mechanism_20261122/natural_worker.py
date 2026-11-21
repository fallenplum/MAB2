"""Paired inference workers on a shared transient spool. There are no answer keys in inputs."""
import itertools,json,time,sys,os
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
mp,fi,fo=sys.argv[1:];fo=Path(fo);which=int(fo.stem);assert which in (0,1)
data=json.loads(Path(fi).read_bytes());assert len(data)==64 and all(set(q)=={'key','task','stem','options'} for q in data)
tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
ids=[tok.encode(a,add_special_tokens=False) for a in 'ABCD'];assert all(len(x)==1 for x in ids)
def question(r):return r['stem']+'\n'+'\n'.join(c+'. '+v for c,v in zip('ABCD',r['options']))
def encode(s):return tok.apply_chat_template([{'role':'user','content':s}],tokenize=True,add_generation_prompt=True)
def pack(rows):return tok.pad({'input_ids':rows},return_tensors='pt').to('cuda:0')
with torch.inference_mode():
    src={};source_calls=0;source_tokens=0
    for j in ([0,2] if which==0 else [1]):
        torch.manual_seed(2026112201+j);torch.cuda.manual_seed_all(2026112201+j)
        for a in range(0,len(data),8):
            batch=data[a:a+8];encoded=[encode(question(r)+'\nOffer a brief reason and your recommended option to another solver. Write at most two sentences.') for r in batch];x=pack(encoded);l=x['input_ids'].shape[1]
            result=model.generate(**x,do_sample=True,top_p=.95,temperature=.8,max_new_tokens=48,pad_token_id=tok.pad_token_id)
            for r,t,ii in zip(batch,result[:,l:].tolist(),encoded):
                n=t.index(tok.eos_token_id)+1 if tok.eos_token_id in t else len(t);src.setdefault(r['key'],{})[str(j)]=tok.decode(t[:n],skip_special_tokens=True).strip();source_tokens+=len(ii)+n;source_calls+=1
    partial=fo.parent/f'source{which}.partial';partial.write_text(json.dumps(src),encoding='utf8');os.replace(partial,fo.parent/f'source{which}.json')
    other=fo.parent/f'source{1-which}.json';deadline=time.monotonic()+600
    while not other.exists():
        if (fo.parent/f'{1-which}.done').exists() or time.monotonic()>deadline:raise RuntimeError('Peer source failed')
        time.sleep(1)
    other_src=json.loads(other.read_bytes())
    orders=[o for n in range(4) for o in itertools.permutations(range(3),n)]
    with fo.open('x',encoding='utf8') as f:
        for r in data:
            messages={**src[r['key']],**other_src[r['key']]};assert set(messages)=={'0','1','2'}
            encoded=[encode('\n'.join([question(r)]+[f'Adviser {j+1}: '+messages[str(j)] for j in o]+['Choose your final answer. Return only A, B, C or D.'])) for o in orders]
            batch=pack(encoded);pos=(batch['attention_mask'].long().cumsum(1)-1).clamp_min(0);probs=model(**batch,position_ids=pos,use_cache=False).logits[:,-1,[x[0] for x in ids]].float().softmax(-1).cpu().tolist()
            f.write(json.dumps(dict(key=r['key'],probs=probs,input_tokens=list(map(len,encoded))))+'\n');f.flush()
    print(json.dumps(dict(source_calls=source_calls,source_tokens=source_tokens,source_empty=sum(not x for m in src.values() for x in m.values()),receiver_calls=64*16)),flush=True)
