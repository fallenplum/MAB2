"""Frozen input-only extension: original pool order, new receiver tokenizer, prior exposure exclusion."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1')
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def choose(pool,measures,old,exp,group_hash,content_hash):
    forbidden_ids=set(exp['ids'])|{q['id'] for q in old['questions']}
    forbidden_groups=set(exp['question_group_hashes'])|{q['question_group_sha256'] for q in old['questions']}
    forbidden_content=set(exp['content_hashes'])|{q['content_sha256'] for q in old['questions']}
    eligible=[i for i,(r,m) in enumerate(zip(pool,measures)) if m['ten_documents'] and max(m['lengths'].values())<=7168 and r['id'] not in forbidden_ids and group_hash(r['question']) not in forbidden_groups and content_hash(r) not in forbidden_content]
    assert len(eligible)>=96,'No pool expansion or resampling'
    return eligible[:96],len(eligible),len(forbidden_groups)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--plan-sha',required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha;p=read(a.plan)
    assert p['source_sha256']==sha(__file__) and p['decision']=='APPROVED_INPUT_ONLY_FACTORIAL'
    for item in p['pins'].values():assert sha(item['path'])==item['sha256']
    contract=read(p['pins']['contract']['path']);assert contract['status']=='FROZEN' and contract['candidate']=='C2_FORMAL4_RECEIVER_MODEL_ACCESS'
    for path in reversed(p['import_paths']):sys.path.insert(0,path)
    import exogenous_path_operator as op
    import select_hotpot as legacy
    import tokenizer_lengths as tk
    pool=[json.loads(line) for line in Path(p['pins']['pool']['path']).read_bytes().splitlines()];assert len(pool)==512
    old=read(p['pins']['old_selection']['path']);exp=read(p['pins']['exposures']['path']);oldcap=read(p['pins']['old_capacity']['path'])
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    spec=p['qwen'];dest=out/'private_tokenizer_qwen';files=tk.copy_tokenizer_only(spec,dest);tok,versions=tk.load_tokenizer(dest,spec['identity'])
    measures=[]
    for i,(row,m) in enumerate(zip(pool,oldcap['measurements'])):
        assert m['index']==i and m['input_sha256']==hashlib.sha256((op.canonical(row)+'\n').encode()).hexdigest()
        m=json.loads(json.dumps(m))
        if m['ten_documents']:
            req=op.input_only_messages({k:row[k] for k in ('id','question','context')})['INDEPENDENT3_M']
            m['lengths']['Q_FULL_FINAL']=len(tok.apply_chat_template(req['messages'],tokenize=True,add_generation_prompt=True))
        measures.append(m)
    chosen,neligible,nexp=choose(pool,measures,old,exp,legacy.question_group_sha256,legacy.content_sha256)
    shocks=np.random.Generator(np.random.PCG64(261115401)).permutation(['U']*48+['V']*48).tolist()
    write(out/'CAPACITY.json',{'measurements':measures,'tokenizer_identity':tk.tokenizer_identity(tok,files,versions),'dynamic_capacity_guarantee':False})
    with (out/'QUESTIONS.jsonl').open('xb') as f:
        for i in chosen:f.write((op.canonical(pool[i])+'\n').encode('utf-8'))
    selection={'schema':'mab2_receiver_factorial_selection_v1','contract_sha256':p['pins']['contract']['sha256'],'pool_sha256':p['pins']['pool']['sha256'],'questions_sha256':sha(out/'QUESTIONS.jsonl'),'capacity_sha256':sha(out/'CAPACITY.json'),'generation_seeds':[26111541,26111542],
        'questions':[{'index':j,'pool_index':i,'id':pool[i]['id'],'input_sha256':hashlib.sha256((op.canonical(pool[i])+'\n').encode()).hexdigest(),'question_group_sha256':legacy.question_group_sha256(pool[i]['question']),'content_sha256':legacy.content_sha256(pool[i]),'removed_upstream':shocks[j]} for j,i in enumerate(chosen)]}
    write(out/'SELECTION_MANIFEST.json',selection)
    receipt={'status':'COMPLETE_PRIVATE_CAPACITY_SELECTION_NO_GPU_ADMISSION','stats':{'selected_questions':96,'eligible_unexposed_pool_questions':neligible,'known_exposed_groups':nexp},'manifest_sha256':sha(out/'SELECTION_MANIFEST.json'),'questions_sha256':sha(out/'QUESTIONS.jsonl'),'capacity_sha256':sha(out/'CAPACITY.json'),'contract_sha256':p['pins']['contract']['sha256'],'plan_sha256':a.plan_sha,'labels_read':False,'GPU_hours':0,'official_test_opened':False,'unknown_cross_project_exposure':'UNKNOWN','static_cap':7168,'runtime_hard_cap':8192}
    write(out/'RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
