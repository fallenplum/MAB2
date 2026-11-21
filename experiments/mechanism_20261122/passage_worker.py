import itertools,json,sys
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

def main(mp,inp,out):
    data=json.loads(Path(inp).read_bytes());assert len(data)==128 and all(set(v)=={'key','question','sections'} for v in data)
    tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(x)==1 for x in ids)
    def code(s):return tok.apply_chat_template([dict(role='user',content=s)],tokenize=True,add_generation_prompt=True)
    def batch(xs):return tok.pad({'input_ids':xs},return_tensors='pt').to('cuda:0')
    with torch.inference_mode(),Path(out).open('x',encoding='utf8') as f:
        for row in data:
            secs=row['sections'];q=row['question'];assert len(secs)==4
            context='\n'.join('Section '+k+': '+v for k,v in zip('ABCD',secs))
            task=context+'\nQuestion: '+q
            pre=[code(task+'\nAs the first researcher select the section most likely to contain the answer. Reply A, B, C or D.')]
            for a in (-1,0,1,2,3):
                first='' if a<0 else '\nThe first researcher recommends section '+'ABCD'[a]+'.'
                pre.append(code(task+first+'\nAs the second researcher select the section most likely to contain the answer. Reply A, B, C or D.'))
            x=batch(pre);z=model(**x,position_ids=(x.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
            probs=z[:,[v[0] for v in ids]].float().softmax(-1).cpu().tolist();texts=[]
            for a,b in itertools.product(range(4),repeat=2):
                texts.append(code('First retrieved excerpt: '+secs[a]+'\nSecond retrieved excerpt: '+secs[b]+'\nQuestion: '+q+
                   '\nReturn only the shortest text span that answers the question. If the excerpts do not support an answer, return NO_EVIDENCE.'))
            x=batch(texts);gen=model.generate(**x,do_sample=False,max_new_tokens=32,pad_token_id=tok.pad_token_id)
            cells=[]
            for seq,ts in zip(texts,gen[:,x.input_ids.shape[1]:].tolist()):
                n=ts.index(tok.eos_token_id)+1 if tok.eos_token_id in ts else len(ts)
                cells.append(dict(text=tok.decode(ts[:n],skip_special_tokens=True).strip(),prefill=len(seq),decode=n))
            f.write(json.dumps(dict(key=row['key'],probs=probs,prefix_tokens=list(map(len,pre)),terminal=cells))+'\n');f.flush()
    print(json.dumps(dict(questions=len(data),prefix_calls=6*len(data),terminal_calls=16*len(data),torch=torch.__version__,transformers=transformers.__version__)),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
