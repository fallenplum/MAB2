"""Generate both sender realizations and all paired downstream counterfactuals."""
import json
import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer


def main(model_path,input_path,output_path):
    inputs=json.loads(Path(input_path).read_bytes())
    assert len(inputs)==256 and all(set(r)=={'key','task','options','stem'} for r in inputs)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    def run(prompts,limit,sample):
        seq=[tok.apply_chat_template([dict(role='user',content=s)],tokenize=True,add_generation_prompt=True) for s in prompts]
        x=tok.pad(dict(input_ids=seq),return_tensors='pt').to('cuda:0')
        args=dict(temperature=.8,top_p=.95) if sample else {}
        generated=model.generate(**x,do_sample=sample,max_new_tokens=limit,pad_token_id=tok.pad_token_id,**args)
        result=[]
        for orig,ids in zip(seq,generated[:,x.input_ids.shape[1]:].tolist()):
            n=ids.index(tok.eos_token_id)+1 if tok.eos_token_id in ids else len(ids)
            result.append(dict(text=tok.decode(ids[:n],skip_special_tokens=True).strip(),prefill=len(orig),decode=n))
        return result
    with torch.inference_mode(), Path(output_path).open('x',encoding='utf8') as out:
        for offset in range(0,len(inputs),16):
            batch=inputs[offset:offset+16]
            q=[r['stem']+'\n'+'\n'.join(c+'. '+v for c,v in zip('ABCDE',r['options'])) for r in batch]
            source=[]
            for seed in (1771,6959):
                torch.manual_seed(seed+offset);torch.cuda.manual_seed_all(seed+offset)
                source.append(run([s+'\nGive your recommended answer letter followed by one brief reason for another solver.' for s in q],48,True))
            middle=[]
            for i in (-1,0,1):
                prompts=[s+('' if i<0 else '\nFirst analyst: '+source[i][j]['text'])+'\nAs the second analyst, give your recommended answer letter followed by one brief reason.' for j,s in enumerate(q)]
                middle.append(run(prompts,48,False))
            final={}
            for i in (-1,0,1):
                for b in range(3):
                    prompts=[s+('' if i<0 else '\nFirst analyst: '+source[i][j]['text'])+
                             '\nSecond analyst: '+middle[b][j]['text']+
                             '\nGive your final choice. Start with the option letter, then briefly explain.' for j,s in enumerate(q)]
                    final[f'{i},{b}']=run(prompts,24,False)
            for j,row in enumerate(batch):
                out.write(json.dumps(dict(key=row['key'],source=[s[j] for s in source],middle=[b[j] for b in middle],
                                          final={k:rows[j] for k,rows in final.items()}))+'\n')
            out.flush()
    print(json.dumps(dict(questions=len(inputs),source_sequences=2*len(inputs),relay_sequences=3*len(inputs),
                          final_sequences=9*len(inputs),torch=torch.__version__,transformers=transformers.__version__)),flush=True)


if __name__=='__main__':main(*sys.argv[1:])
