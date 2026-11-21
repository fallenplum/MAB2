"""Physically run restart, exact-key memoization, and eager transport."""
import itertools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from blind_scheduler import Audit
from qasc_worker import prompts


def decision(a):
    unknown=a.source[~a.known].sum()*(1-a.base.min())
    w=np.where(a.known[:,None]&~a.seen,a.w,0)
    lo=np.minimum(w,0).sum()-unknown;hi=np.maximum(w,0).sum()+unknown
    mass=np.array([a.observed.get(j,0.) for j in range(8)])
    if a.seen.all() or np.all((mass+lo>.01+1e-12)|(mass+hi<.01-1e-12)):
        return list(map(int,np.flatnonzero(mass>.01)))
    return None


def main(mp,input_path,output_path):
    qs=json.loads(Path(input_path).read_bytes());assert len(qs)==144 and all('answer' not in q for q in qs)
    tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    letters=[tok.encode(c,add_special_tokens=False) for c in 'ABCDEFGH'];assert all(len(v)==1 for v in letters)
    ids=[v[0] for v in letters];bt=8
    cooked=[[tok.apply_chat_template([dict(role='user',content=s)],tokenize=True,add_generation_prompt=True) for s in prompts(q)] for q in qs]
    widths=[max(len(seq[j]) for seq in cooked for j in rg) for rg in (range(8),range(8,24))]
    def collect(queries):
        output={};calls=0;tokens=0;slots=0
        for final in (False,True):
            todo=[(q,j) for q,j in queries if (j>=8)==final]
            for t in range(0,len(todo),bt):
                batch=todo[t:t+bt];seq=[cooked[q][j] for q,j in batch]
                padded=seq+[seq[-1]]*(bt-len(seq));width=widths[int(final)]
                x=tok.pad({'input_ids':padded},padding='max_length',max_length=width,return_tensors='pt').to('cuda:0')
                z=model(**x,position_ids=(x.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
                if final:values=z[:,ids].argmax(1).cpu().tolist()
                else:
                    probs=z[:,ids[:4]].float().softmax(1).cpu().numpy().astype(float)
                    values=(probs/probs.sum(1,keepdims=True)).tolist()
                output.update(zip(batch,values));calls+=len(batch);tokens+=sum(len(s)+1 for s in seq);slots+=bt*(width+1)
        return output,np.array([calls,tokens,slots],int)
    def stage(block,mode,old=None):
        torch.cuda.synchronize();start=time.perf_counter()
        ref,ct=collect([(q,j) for q in block for j in ((0,3) if mode=='old' else (1,))])
        engines={};history={q:{} for q in block};done={}
        for q in block:
            source=ref[q,0] if mode=='old' else ref[q,1]
            base=ref[q,3] if mode=='old' else old[0][q,3]
            cost=np.array([len(s)+1 for s in cooked[q]])
            a=Audit(source,base,cost[4:8],cost[8:].reshape(4,4),'joint');engines[q]=a
            if mode=='warm':
                for action,val in old[1][q].items():a.update(action,val)
        while len(done)<len(block):
            orders={}
            for q in block:
                if q in done:continue
                a=engines[q]
                while True:
                    d=decision(a)
                    if d is not None:
                        done[q]=d;break
                    action=a.next()
                    if mode=='memo' and action in old[1][q]:a.update(action,old[1][q][action]);continue
                    i,j=action;orders[q]=(action,4+i if j is None else 8+4*i+j);break
            values,c=collect([(q,j) for q,(_,j) in orders.items()]);ct+=c
            for q,(action,j) in orders.items():
                val=values[q,j];history[q][action]=val;engines[q].update(action,val)
        torch.cuda.synchronize()
        return ref,history,done,dict(calls=int(ct[0]),logical=int(ct[1]),slots=int(ct[2]),seconds=time.perf_counter()-start)
    def foreign():
        p=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        return any(v!=str(os.getpid()) for v in p.split())
    with torch.inference_mode(),Path(output_path).open('x',encoding='utf8') as f:
        # Exclude one kernel warm-up from per-service timings, include it in the GPU ledger.
        collect([(0,0),(0,8)]);orders=list(itertools.permutations(('cold','memo','warm')))
        for bi in range(18):
            block=list(range(8*bi,8*bi+8));shared=foreign();old=stage(block,'old')
            results={method:stage(block,method,old) for method in orders[bi%6]}
            torch.cuda.synchronize();start=time.perf_counter();full,ct=collect([(q,i) for q in block for i in (0,1,3,*range(4,24))])
            torch.cuda.synchronize();full_time=time.perf_counter()-start;shared|=foreign()
            for q in block:
                x={str(j):val for (k,j),val in full.items() if k==q}
                def trace(h):
                    return [[int(i),None if j is None else int(j),v] for (i,j),v in h.items()]
                entry=dict(key=qs[q]['key'],block=bi,full=x,old=old[2][q],old_source=old[0][q,0],base=old[0][q,3],old_trace=trace(old[1][q]),
                           outputs={method:res[2][q] for method,res in results.items()},
                           source={method:res[0][q,1] for method,res in results.items()},
                           traces={method:trace(res[1][q]) for method,res in results.items()})
                if q==block[0]:
                    entry.update(foreign=shared,order=orders[bi%6],costs={'old':old[3],**{m:v[3] for m,v in results.items()},
                        'full':dict(calls=int(ct[0]),logical=int(ct[1]),slots=int(ct[2]),seconds=full_time)})
                f.write(json.dumps(entry)+'\n')
            f.flush()
    print(json.dumps(dict(questions=len(qs),widths=widths,torch=torch.__version__,transformers=transformers.__version__)),flush=True)


if __name__=='__main__':main(*sys.argv[1:])
