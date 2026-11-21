"""Check registered open-numeric results against the pinned TabMWP authors' extraction and normalization."""
import ast,hashlib,json,re
from decimal import Decimal
from pathlib import Path
import numpy as np

def original_functions(external):
    names=[('utilities.py','extract_prediction'),('run_gpt3.py','normalize_answer')]
    ns={'re':re,'np':np}
    for path,name in names:
        tree=ast.parse((external/path).read_text(encoding='utf8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<public-tabmwp-'+name+'>','exec'),ns)
    return ns['extract_prediction'],ns['normalize_answer']


def score(folder,external,source):
    from table_score import parse
    ds=json.loads(source.read_bytes());units={}
    for r in ds.values():
        if not isinstance(r.get('table'),str):continue
        title=r.get('table_title') or '';txt=(title+'\n' if title else '')+r['table']+'\n'+r['question']
        key=hashlib.sha256(' '.join(txt.lower().split()).encode()).hexdigest();units[key]=r.get('unit') or ''
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    extract,normalize=original_functions(external)
    ws=[{v['key']:v for v in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in (0,1)]
    out={}
    for reader in (0,1):
        for group in ('lower','upper'):
            totals=dict(outputs=0,only_official_correct=0,only_registered_correct=0,both_correct=0,official_errors=0,
                        relay_decision_disagreements=0,relay_cases=0,registered_forced_cert=0,official_forced_cert=0)
            for row in data:
                if row['task']!=group:continue
                k=row['key'];y=Decimal(gold[k]);unit=units[k]
                official_gold=normalize(gold[k],unit);different=[]
                for a in range(-1,4):
                    for b in range(4):
                        txt=ws[reader][k]['final'][f'{a},{b}']['text']
                        registered=parse(txt)==y
                        try:official=normalize(extract(txt,[],[]),unit)==official_gold
                        except (ValueError,ZeroDivisionError,IndexError):official=False;totals['official_errors']+=1
                        totals['outputs']+=1;totals['only_official_correct']+=official and not registered
                        totals['only_registered_correct']+=registered and not official;totals['both_correct']+=registered and official
                        if a>=0:different.append((registered,official))
                observed=np.array(different).reshape(4,4,2)
                p=[np.asarray(w[k]['probs']) for w in ws];p=[z/z.sum(1,keepdims=True) for z in p]
                for a in (0,1):
                    delta=p[1-a][2:]-p[1-a][1];s=p[a][0];w=s[:,None]*delta
                    conditional=np.einsum('ab,abk->ak',delta,observed)
                    decisions=(s@conditional)>.01
                    totals['relay_cases']+=1;totals['relay_decision_disagreements']+=decisions[0]!=decisions[1]
                    if abs(w).sum()/2<=.01:
                        totals['registered_forced_cert']+=max(abs(conditional[:,0]))>=.1
                        totals['official_forced_cert']+=max(abs(conditional[:,1]))>=.1
            out[f'{group}_{reader}']=totals
    return out
