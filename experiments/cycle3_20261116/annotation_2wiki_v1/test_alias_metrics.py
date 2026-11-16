"""Synthetic alias scoring, extraction and unchanged-gate equivalence tests."""
from pathlib import Path
from collections import Counter
import ast,hashlib,importlib.util,json,re,string,sys,tempfile,random
import numpy as np
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';d=r/'experiments/cycle3_20261116/annotation_2wiki_v1'
sys.path[:0]=[str(d),str(r/'experiments/cycle2_20261113/hotpot_support')]
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
official=root/'external/2wiki_evaluator_13800e5/official_evaluate_v1_1.py'
assert hashlib.sha256(official.read_bytes()).hexdigest()=='7c09ee2ec98c7ecb57df79eb68dba22eb381ab58d50e730dfcefafb4dda31458'
tree=ast.parse(official.read_text(encoding='utf-8'));functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['normalize_answer','f1_score','exact_match_score','eval_answer','update_answer']]
ns=dict(re=re,string=string,Counter=Counter);exec(compile(ast.Module(body=functions,type_ignores=[]),'official-pure-functions','exec'),ns)
metrics=load('metrics8',d/'alias_metrics.py');ind=load('ind8',d/'audit_raw_scoring.py');extractor=load('extract8',d/'extract_labels.py')
rng=random.Random(261116801);vocab=['yes','no','noanswer','The','alpha','beta','A.','New','York','',',','é','Washington','D.C.']
checks=0
for i in range(1000):
    pred=' '.join(rng.choices(vocab,k=rng.randrange(0,6)));golds=[' '.join(rng.choices(vocab,k=rng.randrange(0,6))) for j in range(rng.randrange(1,5))]
    expected=dict(em=0.,f1=0.,prec=0.,recall=0.);ns['update_answer'](expected,pred,golds)
    actual=metrics.answer_metrics(pred,golds);other=ind.score(pred,golds)
    assert actual['em']==expected['em']==other[0] and abs(actual['f1']-expected['f1'])<1e-14 and abs(other[1]-expected['f1'])<1e-14
    checks+=2
with tempfile.TemporaryDirectory() as td:
    td=Path(td);train=[dict(_id='one',answer='canonical',answer_id='Q1'),dict(_id='two',answer='no',answer_id=None),dict(_id='three',answer='fallback',answer_id='missing'),dict(_id='unused',answer='not retained',answer_id='Q2')]
    (td/'train.json').write_text(json.dumps(train));(td/'aliases.json').write_text(json.dumps(dict(Q_id='Q1',aliases=['alias'],demonyms=['demonym']))+'\n'+json.dumps(dict(Q_id='Q2',aliases=['not retained'],demonyms=[]))+'\n')
    vault,scanned=extractor.extract(['one','two','three'],dict(train_file=dict(path=str(td/'train.json')),alias_file=dict(path=str(td/'aliases.json'))))
    assert scanned==4 and vault=={'one':['alias','canonical','demonym'],'two':['no'],'three':['fallback']}
e7=load('e7test',r/'experiments/cycle3_20261116/evidence_annotation_v1/evaluate.py');e8=load('e8test',d/'evaluate.py')
names=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_ANNOTATION','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
nr=np.random.default_rng(261116801)
for i in range(20):
    em=nr.integers(0,2,(96,2,11));f1=em.copy();tokens=nr.integers(100,50000,(96,2,11));assert e7.summarize(em,f1,tokens,names,[26111671,26111672])==e8.summarize(em,f1,tokens,names,[26111671,26111672])
receipt=dict(status='PASS_FORMAL8_SYNTHETIC_ALIAS_EXTRACTION_AND_GATE_TESTS',official_alias_EM_F1_checks=checks,independent_implementation_checks=checks,
    synthetic_extraction_cases=4,unchanged_gate_random_array_cases=20,benchmark_labels_used=False,new_GPU_calls=0,
    official_scorer_sha256=hashlib.sha256(official.read_bytes()).hexdigest())
out=r/'reports/cycle3_20261116/formal8_preparation';out.mkdir(exist_ok=True); target=out/'SYNTHETIC_TESTS.json'; assert not target.exists() or json.loads(target.read_bytes())==receipt; target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
