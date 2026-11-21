"""Fetch only the registered public TRAIN artifacts, project schema and linkage counts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

SOURCES = {
    'situated_train.jsonl': ('mikejqzhang/SituatedQA', '1ef854512fe0f692ca55f92ff2feedf3d7612eda', 'data/qa_data/temp.train.jsonl', '6e76a6fd00e365efbe682f1d9c98e665337a6191'),
    'situated_raw_train.jsonl': ('mikejqzhang/SituatedQA', '1ef854512fe0f692ca55f92ff2feedf3d7612eda', 'data/raw_data/temp.train.jsonl', '4b8f53986c5db1d890e32eb382e29cec47ec5f06'),
    'timeqa_annotated_train.json': ('wenhuchen/Time-Sensitive-QA', '38b05989070c1168b2bef3d5a2656afeeba763dc', 'dataset/human_annotated_train.json', '7d6241bada55c785a2f56f8f15bc4f199460b1fc'),
    'timeqa_easy_train.json.gzip': ('wenhuchen/Time-Sensitive-QA', '38b05989070c1168b2bef3d5a2656afeeba763dc', 'dataset/human_train.easy.json.gzip', 'b956ed1d37c92a2d0f6d16383e92474b5e99399c'),
}

def fetch(directory, proxy=None):
    import requests
    directory.mkdir(exist_ok=True, parents=True)
    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies.update(http=proxy, https=proxy)
    for name, (repo, rev, path, expected) in SOURCES.items():
        dest = directory / name
        if dest.exists():
            raw = dest.read_bytes()
        else:
            response = session.get(f'https://raw.githubusercontent.com/{repo}/{rev}/{path}', timeout=(10, 60))
            response.raise_for_status()
            raw = response.content
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == expected, name
        if not dest.exists():
            with dest.open('xb') as out:
                out.write(raw)

def structure(obj, depth=0):
    if depth > 3:
        return type(obj).__name__
    if isinstance(obj, list):
        return [structure(obj[0], depth+1)] if obj else []
    if isinstance(obj, dict):
        # Free-form dictionary keys can include question IDs or entity names.
        return {k: structure(v, depth+1) for k,v in obj.items() if k.isascii() and k.islower() and k.replace('_','').isalpha() and len(k) < 30}
    return type(obj).__name__

def read_all(path):
    raw=path.read_bytes()
    if path.suffix == '.jsonl':
        return [json.loads(x) for x in raw.splitlines() if x.strip()]
    if path.suffix == '.gzip':
        if raw[:2] == b'\x1f\x8b':
            raw=gzip.decompress(raw)
        elif raw[:2]==b'PK':
            import zipfile, io
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                ns=[x for x in z.namelist() if 'train' in x and x.endswith('.json') and not x.startswith('__MACOSX')]; assert len(ns)==1
                raw=z.read(ns[0])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if exc.msg != 'Extra data':
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

def audit(directory):
    inventory={}
    for name in SOURCES:
        path=directory/name
        data=read_all(path)
        example=data[0] if isinstance(data,list) else next(iter(data.values()))
        inventory[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'outer_type':type(data).__name__, 'count':len(data), 'row_structure':structure(example)}
    rows=read_all(directory/'situated_train.jsonl'); groups={}
    for row in rows:
        key=' '.join(unicodedata.normalize('NFKC',row['question']).casefold().split()); dates=groups.setdefault(key,{})
        aliases=dates.setdefault(str(row['date']),set()); aliases.update(' '.join(str(a).casefold().split()) for a in row['answer'] if str(a).strip())
    separated=0;paired=0
    for dates in groups.values():
        vals=list(dates.values()); paired+=len(vals)>=2
        separated+=any(a and b and not (a&b) for i,a in enumerate(vals) for b in vals[i+1:])
    return {'sources':inventory,'situated_question_groups':len(groups),'situated_multi_date_groups':paired,'situated_groups_disjoint_literal_aliases_across_dates':separated,'date_type_counts':dict(Counter(x['date_type'] for x in rows)), 'other_split_reads':0, 'GPU_hours':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--proxy');parser.add_argument('--fetch',action='store_true');args=parser.parse_args()
    if args.fetch:fetch(args.directory,args.proxy)
    print(json.dumps(audit(args.directory),indent=2))
