"""Measure conditional choices before and after removal of an evidence edge."""
import itertools
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer


def prompts(row):
    options=row['options']
    problem=row['stem']+'\n'+'\n'.join(c+'. '+v for c,v in zip('ABCDEFGH',options))
    groups=[options[2*i]+' or '+options[2*i+1] for i in range(4)]
    menu='\nHypotheses:\n'+'\n'.join(c+'. '+pair for c,pair in zip('ABCD',groups))
    sender=[problem+menu+('\nRetrieved fact: '+note if note else '')+
            '\nFirst researcher: choose the hypothesis most likely to contain the answer. Return A, B, C or D.'
            for note in (row['first_fact'],'',row['control_fact'])]
    relay=[problem+menu+'\nRetrieved fact: '+row['second_fact']+
           ('\nFirst researcher suggests: '+groups[i] if i>=0 else '')+
           '\nSecond researcher: choose the hypothesis most likely to contain the answer. Return A, B, C or D.'
           for i in (-1,0,1,2,3)]
    terminal=[problem+'\nFirst researcher suggests: '+groups[i]+'\nSecond researcher suggests: '+groups[j]+
              '\nSelect the single best answer to the original question. Return only its letter A through H.'
              for i,j in itertools.product(range(4),repeat=2)]
    return sender+relay+terminal


def main(model_path,input_path,output_path):
    rows=json.loads(Path(input_path).read_bytes())
    assert len(rows)==256 and all(set(r)=={'key','stem','options','first_fact','second_fact','control_fact'} for r in rows)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCDEFGH']
    assert all(len(v)==1 for v in ids)
    with Path(output_path).open('x',encoding='utf8') as out,torch.inference_mode():
        for row in rows:
            seq=[tok.apply_chat_template([dict(role='user',content=t)],tokenize=True,add_generation_prompt=True)
                 for t in prompts(row)]
            x=tok.pad({'input_ids':seq},return_tensors='pt').to('cuda:0')
            pos=(x.attention_mask.long().cumsum(1)-1).clamp_min(0)
            z=model(**x,position_ids=pos,use_cache=False).logits[:,-1]
            probs=z[:8,[v[0] for v in ids[:4]]].float().softmax(-1).cpu().tolist()
            answer=z[8:,[v[0] for v in ids]].argmax(-1).cpu().tolist()
            out.write(json.dumps(dict(key=row['key'],probs=probs,terminal=answer,tokens=list(map(len,seq))))+'\n')
            out.flush()
    print(json.dumps(dict(questions=len(rows),requests=24*len(rows),torch=torch.__version__,transformers=transformers.__version__)),flush=True)


if __name__=='__main__':main(*sys.argv[1:])
