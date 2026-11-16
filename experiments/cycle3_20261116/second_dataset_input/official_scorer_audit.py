"""Pin official answer scorer; compare only synthetic cases, not benchmark labels."""
from pathlib import Path
import requests,hashlib,json,ast,collections,re,string,importlib.util,sys
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';out=root/'external/2wiki_evaluator_13800e5';out.mkdir(exist_ok=False)
commit='13800e5be57df1b4040b9b1588c6c811779e69e9';url=f'https://raw.githubusercontent.com/Alab-NII/2wikimultihop/{commit}/2wikimultihop_evaluate_v1.1.py'
s=requests.Session();s.trust_env=False;s.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'};response=s.get(url,timeout=(20,40));response.raise_for_status();raw=response.content;(out/'official_evaluate_v1_1.py').write_bytes(raw)
# Extract pure answer functions. Imports/file-reading eval() and __main__ are never run.
names={'normalize_answer','f1_score','exact_match_score','eval_answer','update_answer'};tree=ast.parse(raw);definitions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert {x.name for x in definitions}==names
space={'re':re,'string':string,'Counter':collections.Counter};exec(compile(ast.Module(body=definitions,type_ignores=[]),str(out/'official_evaluate_v1_1.py'),'exec'),space)
sys.path.insert(0,str(r/'experiments/cycle2_20261113/hotpot_support'));from hotpot_support import answer_metrics
cases=[('France',['France','French']),('French',['France','French']),('THE river.',['river']),('alpha beta',['alpha gamma','beta delta']),('yes',['no']),('yes no',['yes']),('',['']),('alpha alpha',['alpha']),('alpha',['alpha alpha']),('noanswer',['noanswer']),('A-B',['ab']),('unmatched',['matched'])]
checks=0
for prediction,refs in cases:
    official=dict(em=0,f1=0,prec=0,recall=0);space['update_answer'](official,prediction,refs)
    ours={k:max(answer_metrics(prediction,g)[k] for g in refs) for k in ['em','f1']}
    assert ours['em']==official['em'] and abs(ours['f1']-official['f1'])<1e-12,(prediction,ours,official);checks+=2
receipt=dict(status='PASS_SYNTHETIC_ANSWER_EM_F1_ALIAS_MAX_CONFORMANCE',repository='Alab-NII/2wikimultihop',commit=commit,url=url,source_sha256=hashlib.sha256(raw).hexdigest(),synthetic_cases=len(cases),scalar_checks=checks,official_file_io_executed=False,benchmark_labels_read=False,scope='Answer EM/F1 only; no support/evidence/joint score claim. Preserve full precision internally, presentation may round. Alias set is canonical answer plus official aliases and demonyms by answer_id.',upstream_code_redistributed=False,new_GPU_hours=0)
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');dest=r/'reports/cycle3_20261116/second_dataset_input';(dest/'OFFICIAL_SCORER_SYNTHETIC_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
