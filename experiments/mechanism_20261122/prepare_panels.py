"""Rebuild the two independent input manifests without exposing task instances in the repository."""
import argparse,ast,collections,hashlib,json,re,unicodedata
from decimal import Decimal
from pathlib import Path

STAGE=Path(__file__).resolve().parent
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def text_hash(text):return digest(text.encode())
def norm(text):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))
def output(path,value):
    b=json.dumps(value,ensure_ascii=False,sort_keys=True).encode()
    if path.exists() and path.read_bytes()!=b:raise ValueError('Existing output differs')
    path.write_bytes(b)


def choice(args):
    import pyarrow.parquet as pq
    record=read(STAGE/'M43_INPUT.json')
    prior=set()
    for path in args.prior:
        for obj in read(path):
            prior.add(norm(obj if isinstance(obj,str) else obj['stem']))
    if args.routing_archive:
        for path in args.routing_archive.rglob('*.json'):
            for previous in read(path).get('records',[]):
                prior.add(norm(previous['origin_query']))
    if args.obqa:
        reference=read(STAGE/'M38_INPUT_RECEIPT.json')['source']['sources']
        history=[]
        for tag,path,key in [('arc',args.arc,'question'),('obqa',args.obqa,'question_stem')]:
            assert digest(path.read_bytes())==reference[tag]['sha256']
            for x in pq.read_table(path).to_pylist():
                q=norm(x[key]);opts=x['choices']['text'];labels=x['choices']['label']
                if len(q)>12 and len(opts)==len(set(map(norm,opts)))==4 and x['answerKey'] in labels:history.append((tag,q))
        count=collections.Counter(q for tag,q in history)
        for salt,size in [('MAB2-M38-v1|',48),('M40-natural-v1|',32)]:
            cohort=[]
            for tag in ('arc','obqa'):
                ranked=sorted((q for t,q in history if t==tag and count[q]==1),key=lambda q:text_hash(salt+q))
                cohort.extend([q for q in ranked if not any(q in s for s in prior)][:size])
            prior.update(cohort)
    entries=[]
    for tag,path,n in [('arc',args.arc,4),('csqa',args.csqa,5)]:
        assert digest(path.read_bytes())==record[tag+'_sha256']
        for x in pq.read_table(path).to_pylist():
            q=norm(x['question']);labels=x['choices']['label'];opts=x['choices']['text']
            if len(q)>12 and len(opts)==len(set(map(norm,opts)))==n and labels==list('ABCDE')[:n] and x['answerKey'] in labels:
                entries.append((tag,q,x['question'],opts,labels.index(x['answerKey'])))
    counts=collections.Counter(r[1] for r in entries);inputs=[];gold={}
    for tag in ('arc','csqa'):
        rows=sorted((r for r in entries if r[0]==tag and counts[r[1]]==1),key=lambda r:text_hash('M43-public-relay-v1|'+r[1]))
        used=0
        for _,q,text,opts,y in rows:
            if any(q in s for s in prior):continue
            key=text_hash(tag+'|'+q);inputs.append(dict(key=key,task=tag,stem=text,options=opts));gold[key]=y;used+=1
            if used==128:break
    return inputs,gold,record


def numeric(args):
    record=read(STAGE/'M46_INPUT.json');assert digest(args.train.read_bytes())==record['source_sha256']
    ds=read(args.train);entries=[]
    for x in ds.values():
        if x.get('choices') or x.get('ques_type')!='free_text':continue
        value=str(x.get('answer','')).strip().replace(',','')
        if not re.fullmatch(r'-?\d+(?:\.\d+)?',value):continue
        table=x.get('table');q=x['question']
        if not isinstance(table,str) or not table.strip() or len(table)>2400 or len(q)>800:continue
        title=x.get('table_title') or '';text=(title+'\n' if title else '')+table+'\n'+q
        entries.append((text_hash(' '.join(text.lower().split())),'lower' if int(x['grade'])<=4 else 'upper',text,str(Decimal(value))))
    counts=collections.Counter(r[0] for r in entries);inputs=[];gold={}
    for tag in ('lower','upper'):
        eligible=sorted((r for r in entries if counts[r[0]]==1 and r[1]==tag),key=lambda r:text_hash('M46-relay-strategies-v1|'+r[0]))
        for key,name,text,value in eligible[:64]:inputs.append(dict(key=key,task=name,stem=text));gold[key]=value
    directory=args.agentprune/'AgentPrune';reference=read(STAGE/'M42_AUTHOR_READOUT.json')
    graph=(directory/'graph/graph.py').read_bytes().replace(b'\r\n',b'\n')
    assert reference['file_hash'] in (digest(graph),digest(graph.replace(b'\n',b'\r\n')))
    # Use only pinned literal getters; render an independent copy of the released method's output, then verify exact hash.
    tree=ast.parse((directory/'prompt/gsm8k_prompt_set.py').read_text(encoding='utf8'))
    cls=next(n for n in tree.body if getattr(n,'name','')=='GSM8KPromptSet')
    literals={f.name:ast.literal_eval(f.body[-1].value) for f in cls.body if getattr(f,'name','') in
              ('get_decision_role','get_decision_constraint','get_decision_few_shot')}
    system=literals['get_decision_role']+'.\n '+literals['get_decision_constraint']
    prefix=literals['get_decision_few_shot']+' The task is:\n\n {query}.\n At the same time, the output of other agents is as follows:\n\n'
    prompt=dict(system=system,user=prefix+'First analyst: {first}\n\nSecond analyst: {second}\n\n',last=prefix+'Second analyst: {second}\n\n')
    raw=json.dumps(prompt).encode()
    assert digest(raw)==record['prompt_sha256']
    dest=args.output/'public_prompt.json'
    if dest.exists():assert dest.read_bytes()==raw
    dest.write_bytes(raw)
    return inputs,gold,record


def science(args):
    import pyarrow.parquet as pq
    receipt=read(STAGE/'M49_INPUT.json');assert digest(args.train.read_bytes())==receipt['train_sha256']
    rows=pq.read_table(args.train).to_pylist();counts=collections.Counter(norm(row['question']) for row in rows)
    past={norm(row['stem']) for p in args.prior for row in read(p)}
    data=[];gold={};supports=set()
    for row in sorted(rows,key=lambda r:text_hash('M49-SciQ-source-support|'+norm(r['question']))):
        q=norm(row['question']);sup=norm(row['support'])
        options=[row[k].strip() for k in ('correct_answer','distractor1','distractor2','distractor3')]
        if counts[q]!=1 or q in past or sup in supports or not 20<=len(row['support'].split())<=200 or len(set(map(norm,options)))!=4 or not all(options):continue
        supports.add(sup);key=text_hash(q);opts=sorted(options,key=lambda v:text_hash(key+'|'+norm(v)))
        data.append(dict(key=key,stem=row['question'],options=opts,support=row['support']));gold[key]=opts.index(options[0])
        if len(data)==256:break
    ix=sorted(range(len(data)),key=lambda i:(len(data[i]['support'].split()),data[i]['key']))
    for i,j in zip(ix[::2],ix[1::2]):data[i]['control']=data[j]['support'];data[j]['control']=data[i]['support']
    return data,gold,receipt


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='panel',required=True)
    c=sub.add_parser('choice')
    c.add_argument('--arc',type=Path,required=True);c.add_argument('--csqa',type=Path,required=True)
    c.add_argument('--obqa',type=Path,help='Reconstruct previous public-source cohorts used for exclusion.')
    c.add_argument('--routing-archive',type=Path,help='Previously exposed LLMRouterBench arcc/test archive, used only to exclude question stems.')
    c.add_argument('--prior',type=Path,action='append',default=[])
    n=sub.add_parser('numeric')
    n.add_argument('--train',type=Path,required=True);n.add_argument('--agentprune',type=Path,required=True)
    sc=sub.add_parser('science');sc.add_argument('--train',type=Path,required=True)
    sc.add_argument('--prior',type=Path,action='append',default=[])
    for s in (c,n,sc):s.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    inputs,gold,record={'choice':choice,'numeric':numeric,'science':science}[args.panel](args)
    pending=[('INPUT.json',inputs,'input_sha256'),('SEALED.json',gold,'label_sha256')]
    for name,value,key in pending:
        if digest(json.dumps(value,ensure_ascii=False,sort_keys=True).encode())!=record[key]:
            raise ValueError(f'{name} differs from frozen manifest. For the choice panel include the exact historical exclusion list.')
    for name,value,key in pending:output(args.output/name,value)
    print(f'Verified {len(inputs)} public-source inputs and the frozen label digest.')

if __name__=='__main__':main()
