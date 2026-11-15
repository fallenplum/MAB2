"""Export pinned public code to a flat execution directory; no model/data access."""
from pathlib import Path
import argparse,hashlib,json,shutil
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(repo,out):
    assert not out.exists(),'Use a new directory; historical bundles are immutable'
    base=repo/'experiments/cycle2_20261113';d=base/'closure_revalidation_v1'
    execution=json.loads((repo/'protocol/cycle2_closure_revalidation_20261115/EXECUTION_FROZEN_V1.json').read_bytes())
    science=json.loads((repo/'protocol/cycle2_closure_revalidation_20261115/SCIENTIFIC_DESIGN_FROZEN_V1.json').read_bytes())
    mapping={p.name:p for p in d.glob('*.py') if p.name!='build_portable_closure_bundle.py'}
    mapping.update({'exogenous_path_operator.py':base/'exogenous_paths_v1/exogenous_path_operator.py','test_operator.py':base/'exogenous_paths_v1/test_operator.py','factorial_operator.py':base/'receiver_factorial_v1/factorial_operator.py','durable_backend.py':base/'durable_backend_v1/durable_backend.py','hotpot_support.py':base/'hotpot_support/hotpot_support.py'})
    mapping.update({n:repo/'src'/n for n in execution['source_sha256'] if n.startswith('graph_credit/')})
    for key,pin in execution['source_sha256'].items():assert sha(mapping[key])==pin,('execution hash mismatch',key)
    for key,pin in science['source_sha256'].items():assert sha(mapping[key])==pin,('scientific hash mismatch',key)
    out.mkdir(parents=True)
    for name,path in mapping.items():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    receipt={'status':'PINNED_CODE_BUNDLE_EXPORTED','files':{k:sha(p) for k,p in mapping.items()},'scientific_contract_sha256':sha(repo/'protocol/cycle2_closure_revalidation_20261115/SCIENTIFIC_DESIGN_FROZEN_V1.json'),'execution_contract_sha256':sha(repo/'protocol/cycle2_closure_revalidation_20261115/EXECUTION_FROZEN_V1.json'),'data_or_models_copied':False,'new_model_calls':0}
    (out/'BUNDLE_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=build(a.repo.resolve(),a.output.resolve());print(json.dumps({'status':r['status'],'files':len(r['files'])}))
if __name__=='__main__':main()
