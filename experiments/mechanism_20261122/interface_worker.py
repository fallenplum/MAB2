"""A finite semantic channel with a full measured terminal response table."""
import itertools,json,sys
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

def main(model_path,input_path,output_path):
    inputs=json.loads(Path(input_path).read_bytes())
    assert len(inputs)==256 and all(set(r)=={'key','task','options','stem'} for r in inputs)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    with torch.inference_mode(),Path(output_path).open('x',encoding='utf8') as out:
        for row in inputs:
            alphabet='ABCDE'[:len(row['options'])]
            ids=[tok.encode(c,add_special_tokens=False) for c in alphabet]
            assert len(alphabet) in (4,5) and all(len(x)==1 for x in ids)
            base=row['stem']+'\n'+'\n'.join(c+'. '+s for c,s in zip(alphabet,row['options']))
            tail='Return only one option letter: '+', '.join(alphabet)+'.'
            texts=[base+'\nChoose your own answer. '+tail]
            for u,v in itertools.product('-'+alphabet,repeat=2):
                hints=[f'Adviser {j}: {x}' for j,x in [(1,u),(2,v)] if x!='-']
                texts.append('\n'.join([base,*hints,'Choose your final answer. '+tail]))
            seq=[tok.apply_chat_template([{'role':'user','content':s}],tokenize=True,add_generation_prompt=True) for s in texts]
            batch=tok.pad(dict(input_ids=seq),return_tensors='pt').to('cuda:0');mask=batch['attention_mask']
            pred=model(**batch,position_ids=(mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
            vals=pred[:,[x[0] for x in ids]].float()
            out.write(json.dumps(dict(key=row['key'],probs=vals.softmax(-1).cpu().tolist(),input_tokens=list(map(len,seq))))+'\n');out.flush()
    print(json.dumps(dict(questions=len(inputs),sequences=sum(1+(len(r['options'])+1)**2 for r in inputs),
                          torch=torch.__version__,transformers=transformers.__version__)),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
