import itertools,json,sys
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer


def main(model_path,input_path,output_path):
    rows=json.loads(Path(input_path).read_bytes())
    assert len(rows)==256 and all(set(r)=={'key','stem','options','support','control'} for r in rows)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    ids=[tok.encode(v,add_special_tokens=False) for v in 'ABCD'];assert all(len(x)==1 for x in ids)
    with Path(output_path).open('x',encoding='utf8') as out,torch.inference_mode():
        for row in rows:
            problem=row['stem']+'\n'+'\n'.join(c+'. '+v for c,v in zip('ABCD',row['options']))
            texts=[]
            for context in ('',row['support'],row['control']):
                texts.append(problem+('\nResearch note: '+context if context else '')+'\nChoose your own answer. Return only A, B, C or D.')
            for pair in itertools.product('-ABCD',repeat=2):
                notes=[f'Adviser {j+1}: {v}' for j,v in enumerate(pair) if v!='-']
                texts.append('\n'.join([problem,*notes,'Choose your final answer. Return only A, B, C or D.']))
            seq=[tok.apply_chat_template([dict(role='user',content=s)],tokenize=True,add_generation_prompt=True) for s in texts]
            x=tok.pad({'input_ids':seq},return_tensors='pt').to('cuda:0')
            logits=model(**x,position_ids=(x.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
            probs=logits[:,[r[0] for r in ids]].float().softmax(-1).cpu().tolist()
            out.write(json.dumps(dict(key=row['key'],probs=probs,input_tokens=list(map(len,seq))))+'\n');out.flush()
    print(json.dumps(dict(questions=len(rows),sequences=len(rows)*28,torch=torch.__version__,transformers=transformers.__version__)),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
