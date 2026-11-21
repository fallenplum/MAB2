import argparse,hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq


def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def prepare(train,prior):
    seen_q={norm(r['stem']) for path in prior for r in json.loads(path.read_bytes())}
    used_facts=set();inputs=[];answers={}
    rows=sorted(pq.read_table(train).to_pylist(),key=lambda r:sha('M56-communication-removal|'+r['id']))
    for row in rows:
        q=norm(row['question']);facts=[row['fact1'],row['fact2']]
        choices=row['choices'];opts=choices['text'];labels=choices['label']
        if q in seen_q or set(map(norm,facts)) & used_facts or not all(map(norm,facts)):continue
        if len(set(map(norm,opts)))!=8 or labels!=list('ABCDEFGH') or row['answerKey'] not in labels:continue
        key=sha(q);correct=opts[labels.index(row['answerKey'])]
        opts=sorted(opts,key=lambda t:sha(key+'|'+norm(t)))
        inputs.append(dict(key=key,stem=row['question'],options=opts,first_fact=facts[0],second_fact=facts[1]))
        answers[key]=str(opts.index(correct));seen_q.add(q);used_facts.update(map(norm,facts))
        if len(inputs)==256:break
    assert len(inputs)==256
    ix=sorted(range(256),key=lambda i:(len(inputs[i]['first_fact'].split()),inputs[i]['key']))
    for i,j in zip(ix[::2],ix[1::2]):
        inputs[i]['control_fact']=inputs[j]['first_fact'];inputs[j]['control_fact']=inputs[i]['first_fact']
    return inputs,answers


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('train',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--prior',type=Path,action='append',default=[])
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    receipt=json.loads(Path(__file__).with_name('M56_INPUT.json').read_bytes())
    assert hashlib.sha256(a.train.read_bytes()).hexdigest()==receipt['train_sha256']
    for name,value,k in zip(('INPUT.json','SEALED.json'),prepare(a.train,a.prior),('input_sha256','label_sha256')):
        raw=json.dumps(value,sort_keys=True,ensure_ascii=False).encode()
        assert hashlib.sha256(raw).hexdigest()==receipt[k]
        dest=a.output/name
        if dest.exists():assert dest.read_bytes()==raw
        else:dest.write_bytes(raw)
