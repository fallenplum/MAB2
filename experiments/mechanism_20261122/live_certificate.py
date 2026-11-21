"""Execute all three auditing policies on hardware, with independent forward calls."""
import itertools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

STRATEGIES=('look up an entry or count matching entries','add or subtract quantities or compare differences',
            'multiply, divide, or convert a rate','sort, group, or compute a summary statistic')
POLICIES=('full','anchored','weighted')

def main(model_path,input_path,output_path):
    p=Path(input_path);inputs=json.loads(p.read_bytes());template=json.loads((p.parent/'public_prompt.json').read_bytes())
    assert len(inputs)==48 and all(set(r)=={'key','task','stem'} for r in inputs)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(v)==1 for v in ids)
    mode='native'
    try:tok.apply_chat_template([dict(role='system',content='Test.'),dict(role='user',content='Test.')],tokenize=True,add_generation_prompt=True)
    except Exception as e:
        if 'role' in str(e).lower() or 'system' in str(e).lower():mode='folded'
        else:raise
    def enc(user,system=None):
        message=[dict(role='user',content=user)]
        if system and mode=='native':message.insert(0,dict(role='system',content=system))
        elif system:message[0]['content']=system+'\n\n'+user
        return tok.apply_chat_template(message,tokenize=True,add_generation_prompt=True)
    pack=lambda texts:tok.pad({'input_ids':texts},return_tensors='pt').to('cuda:0')
    def physical_pids():
        return subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    def audit(problem,method):
        torch.cuda.synchronize();t0=time.perf_counter()
        menu='\n'.join(k+'. '+v for k,v in zip('ABCD',STRATEGIES))
        common=problem+'\nPossible solution strategies:\n'+menu
        prefix=[common+'\nAs the first analyst, choose the single most useful strategy. Answer with one letter.']
        for i in (-1,0,1,2,3):
            msg='' if i<0 else '\nFirst analyst recommends: '+STRATEGIES[i]+'.'
            prefix.append(common+msg+'\nAs the second analyst, choose the single most useful strategy. Answer with one letter.')
        prefix=[enc(s) for s in prefix];x=pack(prefix)
        raw=model(**x,position_ids=(x.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
        prob=raw[:,[v[0] for v in ids]].float().softmax(-1).cpu().numpy().astype(float);prob/=prob.sum(1,keepdims=True)
        radii=np.abs(prob[2:]-prob[1]).sum(1)/2
        weighted=float(prob[0]@radii);anchored=float(max(radii))
        diameter=float(max(abs(a-b).sum()/2 for a in prob[1:] for b in prob[1:]))
        skip=method!='full' and (weighted if method=='weighted' else anchored)<.01-1e-10
        results=[];calls=6;prefill=sum(map(len,prefix));decode=6
        if not skip:
            texts=[]
            for a,b in itertools.product(range(4),repeat=2):
                user=template['user'].replace('{query}',problem).replace('{first}',STRATEGIES[a]).replace('{second}',STRATEGIES[b])
                texts.append(enc(user,template['system']))
            x=pack(texts)
            gen=model.generate(**x,do_sample=False,max_new_tokens=160,pad_token_id=tok.pad_token_id)
            calls+=16;prefill+=sum(map(len,texts))
            for ts in gen[:,x.input_ids.shape[1]:].tolist():
                n=ts.index(tok.eos_token_id)+1 if tok.eos_token_id in ts else len(ts)
                results.append(tok.decode(ts[:n],skip_special_tokens=True).strip());decode+=n
        torch.cuda.synchronize()
        return dict(probs=prob.tolist(),skip=skip,weighted=weighted,anchored=anchored,diameter=diameter,
                    results=results,seqs=calls,prefill=prefill,decode=decode,wall_seconds=time.perf_counter()-t0)
    with torch.inference_mode(),Path(output_path).open('x',encoding='utf8') as out:
        # Full warm-up on an artificial input independent of the selected questions. Charge it in residency and collection.
        warm=audit('Number of marbles | 4\nHow many marbles are there?','full')
        orders=list(itertools.permutations(POLICIES))
        for idx,row in enumerate(inputs):
            other=any(pid!=str(os.getpid()) for pid in physical_pids())
            order=orders[idx%6];trial={method:audit(row['stem'],method) for method in order}
            other|=any(pid!=str(os.getpid()) for pid in physical_pids())
            out.write(json.dumps(dict(key=row['key'],arm_order=order,other_process_observed=other,policies=trial))+'\n');out.flush()
    print(json.dumps(dict(questions=len(inputs),template_mode=mode,torch=torch.__version__,transformers=transformers.__version__,
                          warmup={k:warm[k] for k in ('wall_seconds','seqs','prefill','decode')})),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
