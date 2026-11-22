import itertools,json,sys
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

def main(model_path,input_path,output_path):
    rows=json.loads(Path(input_path).read_bytes());assert len(rows)==192
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    n=len(rows[0]['options']);assert n in (4,8);groups=n//2
    ids=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(n)];assert all(len(v)==1 for v in ids)
    def code(text):return tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
    with torch.inference_mode(),Path(output_path).open('x',encoding='utf8') as out:
        for row in rows:
            menu='\n'.join(chr(65+i)+'. '+v for i,v in enumerate(row['options']));problem=row['stem']+'\n'+menu
            source=[problem+('\nEvidence: '+v if v else '')+'\nChoose the best answer. Return one letter.' for v in (row.get('first_fact',''),' ',row.get('control_fact',''))]
            relay=[problem+'\nEvidence: '+row.get('second_fact','')+('\\nFirst researcher proposes option '+chr(65+i)+'.' if i>=0 else '')+'\nChoose the best answer. Return one letter.' for i in (-1,*range(groups))]
            terminals=[problem+'\nFirst researcher proposes option '+chr(65+i)+'.\nSecond researcher proposes option '+chr(65+j)+'.\nReturn only the best answer letter.' for i,j in itertools.product(range(groups),repeat=2)]
            seq=[code(v) for v in source+relay+terminals];x=tok.pad({'input_ids':seq},return_tensors='pt').to('cuda:0')
            pos=(x.attention_mask.long().cumsum(1)-1).clamp_min(0);z=model(**x,position_ids=pos,use_cache=False).logits[:,-1]
            probs=z[:6,[v[0] for v in ids]].float().softmax(1).cpu().tolist();labels=z[6:,[v[0] for v in ids]].argmax(1).cpu().tolist()
            full={str(i):probs[i] for i in range(6)};full.update({str(6+i):labels[i] for i in range(len(labels))})
            out.write(json.dumps(dict(key=row['key'],full=full,tokens=list(map(len,seq)))+'\n');out.flush()
    print(json.dumps(dict(questions=len(rows),n_options=n,requests=len(rows)*len(seq),torch=torch.__version__,transformers=transformers.__version__)),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
