import itertools,json,sys
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

STRATEGIES=('look up an entry or count matching entries','add or subtract quantities or compare differences',
            'multiply, divide, or convert a rate','sort, group, or compute a summary statistic')
def main(model_path,input_path,output_path):
    p=Path(input_path);data=json.loads(p.read_bytes());template=json.loads((p.parent/'public_prompt.json').read_bytes())
    assert len(data)==128 and all(set(r)=={'key','task','stem'} for r in data)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(v)==1 for v in ids)
    def encode(user,system=None):
        messages=[dict(role='user',content=user)]
        if system:messages.insert(0,dict(role='system',content=system))
        try:return tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        except Exception as exc:
            if system and ('role' in str(exc).lower() or 'system' in str(exc).lower()):
                return tok.apply_chat_template([dict(role='user',content=system+'\n\n'+user)],tokenize=True,add_generation_prompt=True)
            raise
    pack=lambda texts:tok.pad({'input_ids':texts},return_tensors='pt').to('cuda:0')
    menu='\n'.join(c+'. '+m for c,m in zip('ABCD',STRATEGIES))
    with torch.inference_mode(),Path(output_path).open('x',encoding='utf8') as out:
        for row in data:
            problem=row['stem']
            common=problem+'\nPossible solution strategies:\n'+menu
            prompts=[common+'\nAs the first analyst, choose the single most useful strategy. Answer with one letter.']
            for i in (-1,0,1,2,3):
                msg='' if i<0 else '\nFirst analyst recommends: '+STRATEGIES[i]+'.'
                prompts.append(common+msg+'\nAs the second analyst, choose the single most useful strategy. Answer with one letter.')
            codes=[encode(t) for t in prompts];x=pack(codes)
            z=model(**x,position_ids=(x.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
            prob=z[:,[v[0] for v in ids]].float().softmax(-1).cpu().tolist()
            final={};texts=[];conditions=[]
            for a in (-1,0,1,2,3):
                for b in range(4):
                    raw=template['last' if a<0 else 'user'].replace('{query}',problem).replace('{second}',STRATEGIES[b])
                    if a>=0:raw=raw.replace('{first}',STRATEGIES[a])
                    texts.append(encode(raw,template['system']));conditions.append(f'{a},{b}')
            x=pack(texts)
            gen=model.generate(**x,do_sample=False,max_new_tokens=160,pad_token_id=tok.pad_token_id)
            for name,orig,tokens in zip(conditions,texts,gen[:,x.input_ids.shape[1]:].tolist()):
                n=tokens.index(tok.eos_token_id)+1 if tok.eos_token_id in tokens else len(tokens)
                final[name]=dict(text=tok.decode(tokens[:n],skip_special_tokens=True).strip(),prefill=len(orig),decode=n)
            out.write(json.dumps(dict(key=row['key'],probs=prob,policy_inputs=list(map(len,codes)),final=final))+'\n');out.flush()
    print(json.dumps(dict(questions=len(data),policy_calls=6*len(data),final_calls=20*len(data),
                          torch=torch.__version__,transformers=transformers.__version__)),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
