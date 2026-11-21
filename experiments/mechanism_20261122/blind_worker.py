"""Real batched, gold-blind adaptive audits of free numerical answer distributions."""
import ast,itertools,json,os,re,subprocess,sys,time
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from blind_scheduler import Audit,exhaustive

STRATEGIES=('look up an entry or count matching entries','add or subtract quantities or compare differences',
            'multiply, divide, or convert a rate','sort, group, or compute a summary statistic')
BATCH=8;LIMIT=512
def authors(folder):
    env={'re':re,'np':np}
    for fname,name in [('utilities.py','extract_prediction'),('run_gpt3.py','normalize_answer')]:
        tree=ast.parse((folder/fname).read_text(encoding='utf8'));f=next(n for n in tree.body if getattr(n,'name','')==name)
        exec(compile(ast.Module(body=[f],type_ignores=[]),'<official-grader>','exec'),env)
    def label(text,unit):
        try:
            v=Decimal(env['normalize_answer'](env['extract_prediction'](text,[],[]),unit))
            return str(v.normalize()) if v.is_finite() else 'INVALID'
        except (InvalidOperation,ZeroDivisionError,ValueError,IndexError):return 'INVALID'
    return label


def main(mp,inputfile,outputfile):
    folder=Path(inputfile).parent;data=json.loads(Path(inputfile).read_bytes());template=json.loads((folder/'public_prompt.json').read_bytes())
    assert len(data)==48 and all(set(r)=={'key','task','stem','unit'} for r in data)
    canonical=authors(folder)
    tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
    try:
        tok.apply_chat_template([dict(role='system',content='Test.'),dict(role='user',content='Test.')],tokenize=True,add_generation_prompt=True);native=True
    except Exception as e:
        if 'role' not in str(e).lower() and 'system' not in str(e).lower():raise
        native=False
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(v)==1 for v in ids)
    def code(s,sy=None):
        m=[dict(role='user',content=s)]
        if sy and native:m.insert(0,dict(role='system',content=sy))
        if sy and not native:m[0]['content']=sy+'\n\n'+s
        return tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True)
    def prepare(r):
        menu='\n'.join(c+'. '+s for c,s in zip('ABCD',STRATEGIES));base=r['stem']+'\nPossible solution strategies:\n'+menu
        pre=[code(base+'\nAs the first analyst, choose the single most useful strategy. Answer with one letter.')]
        for i in (-1,0,1,2,3):
            ctx='' if i<0 else '\nFirst analyst recommends: '+STRATEGIES[i]+'.'
            pre.append(code(base+ctx+'\nAs the second analyst, choose the single most useful strategy. Answer with one letter.'))
        final=[code(template['user'].replace('{query}',r['stem']).replace('{first}',STRATEGIES[i]).replace('{second}',STRATEGIES[j]),template['system'])
               for i,j in itertools.product(range(4),repeat=2)]
        return pre,final
    t=time.perf_counter();prompts=[prepare(r) for r in data];prep=time.perf_counter()-t
    width=[max(len(s) for pair in prompts for s in pair[g]) for g in (0,1)]
    def execute(requests,kind,records,counters):
        for offset in range(0,len(requests),BATCH):
            chunk=requests[offset:offset+BATCH];seq=[prompts[q][kind][j] for q,j in chunk]
            padded=seq+[seq[-1]]*(BATCH-len(seq))
            batch=tok.pad({'input_ids':padded},padding='max_length',max_length=width[kind],return_tensors='pt').to('cuda:0')
            counters['logical_prefill']+=sum(map(len,seq));counters['slots_prefill']+=BATCH*width[kind];counters['logical_calls']+=len(seq)
            counters['batches']+=1
            if kind==0:
                z=model(**batch,position_ids=(batch.attention_mask.long().cumsum(1)-1).clamp_min(0),use_cache=False).logits[:,-1]
                probs=z[:,[t[0] for t in ids]].float().softmax(-1).cpu().numpy().astype(float)
                for (q,j),vec in zip(chunk,probs):records[q]['prefix'][j]=(vec/vec.sum()).tolist()
                counters['logical_decode']+=len(seq);counters['slots_decode']+=BATCH
            else:
                out=model.generate(**batch,do_sample=False,max_new_tokens=LIMIT,pad_token_id=tok.pad_token_id)[:,width[1]:].tolist()
                counters['slots_decode']+=BATCH*len(out[0])
                for (q,j),tokens in zip(chunk,out):
                    n=tokens.index(tok.eos_token_id)+1 if tok.eos_token_id in tokens else len(tokens)
                    text=tok.decode(tokens[:n],skip_special_tokens=True).strip()
                    records[q]['terminal'][j]=dict(label=canonical(text,data[q]['unit']),text=text,decode=n,prefill=len(prompts[q][1][j]))
                    counters['logical_decode']+=n
    def live(block,policy):
        rec={q:dict(prefix={},terminal={},keep=[]) for q in block}
        ct=dict(logical_prefill=0,slots_prefill=0,logical_decode=0,slots_decode=0,logical_calls=0,batches=0)
        torch.cuda.synchronize();start=time.perf_counter()
        if policy=='full':
            execute([(q,j) for q in block for j in range(6)],0,rec,ct)
            execute([(q,j) for q in block for j in range(16)],1,rec,ct)
            for q in block:
                p=rec[q]['prefix'];terminal=rec[q]['terminal']
                labels=[[terminal[i*4+j]['label'] for j in range(4)] for i in range(4)]
                rec[q]['keep']=sorted(exhaustive(p[0],p[1],[p[i] for i in range(2,6)],labels))
        else:
            execute([(q,j) for q in block for j in (0,1)],0,rec,ct)
            audits={q:Audit(rec[q]['prefix'][0],rec[q]['prefix'][1],[len(s)+1 for s in prompts[q][0][2:]],
                            np.reshape([len(s)+LIMIT for s in prompts[q][1]],(4,4)),policy) for q in block}
            while audits:
                orders={}
                for q,audit in list(audits.items()):
                    decision=audit.decision()
                    if decision is not None:rec[q]['keep']=sorted(decision);del audits[q]
                    else:orders[q]=audit.next()
                pre=[(q,i+2) for q,(i,j) in orders.items() if j is None]
                post=[(q,i*4+j) for q,(i,j) in orders.items() if j is not None]
                execute(pre,0,rec,ct);execute(post,1,rec,ct)
                for q,(i,j) in orders.items():
                    result=rec[q]['prefix'][i+2] if j is None else rec[q]['terminal'][i*4+j]['label']
                    audits[q].update((i,j),result)
        torch.cuda.synchronize();ct['wall_seconds']=time.perf_counter()-start
        return rec,ct
    def foreign():return any(pid!=str(os.getpid()) for pid in subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split())
    with torch.inference_mode(),Path(outputfile).open('x',encoding='utf8') as out:
        # Exercise both entry points on a dummy prompt to avoid first-use bias in the first arm.
        warm_tokens=code('Return the number 1.');x=tok.pad({'input_ids':[warm_tokens]*BATCH},return_tensors='pt').to('cuda:0')
        model(**x,use_cache=False);model.generate(**x,do_sample=False,max_new_tokens=4,pad_token_id=tok.pad_token_id)
        permutations=list(itertools.permutations(('full','joint','row_first')))
        for block in range(6):
            ix=list(range(block*8,block*8+8));order=permutations[block];results={};costs={};others=foreign()
            for strategy in order:results[strategy],costs[strategy]=live(ix,strategy)
            others|=foreign()
            for j,q in enumerate(ix):
                out.write(json.dumps(dict(key=data[q]['key'],block=block,order=order,foreign=others,
                      arms={s:results[s][q] for s in order},costs=costs if j==0 else {},
                      lengths=[list(map(len,part)) for part in prompts[q]]))+'\n')
            out.flush()
    print(json.dumps(dict(questions=len(data),prompt_preparation_seconds=prep,padded_widths=width,native_system=native,torch=torch.__version__,transformers=transformers.__version__)),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
