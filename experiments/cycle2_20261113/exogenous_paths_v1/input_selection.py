"""Private, input-only pool and capacity selection for an exogenous-path screen.

No label projection, model construction or GPU use. Real IO needs a frozen
contract plus a root-issued hash-bound private plan. Synthetic helpers are pure.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
LEGACY_SHA = '5edc8dc62b349be49c2c939c9c3f93ba1a8795b3c6fc7c951aec3ab9cd3d8bfd'
OLD_SELECTION_SHA = 'f64819c79fc233f7ae455d777bc6b51b2023fed7af92ae2ca6a43a9e3109105e'
OLD_EXPOSURES_SHA = '0d6a79d1f6dc0caa86d918bea9a007654fec0ba7746015953ed265a934354826'
SOURCE_SHA = '5cf1902c112089f8b787dc8bb1a6a46df285003fb9318e0134082f1c5d148456'

def require(ok, message):
    if not ok: raise ValueError(message)
def canonical(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def pinned(p,h):
    require(sha(p)==h,'Pinned bytes differ'); x=json.loads(Path(p).read_bytes())
    require(sha(p)==h,'Pinned input changed'); return x
def write(p,x):
    with Path(p).open('xb') as f: f.write(canonical(x));f.flush();os.fsync(f.fileno())
def legacy(path):
    require(sha(path)==LEGACY_SHA,'Legacy input validator differs')
    spec=importlib.util.spec_from_file_location('_path_input_legacy',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def union_exposures(old, selected, mod):
    mod.validate_exposures(old)
    require(len(selected['questions'])==384,'Previous revealed selection count differs')
    result=json.loads(json.dumps(old))
    for field,src in [('ids','id'),('question_group_hashes','question_group_sha256'),('content_hashes','content_sha256')]:
        result[field]=sorted(set(result[field]) | {q[src] for q in selected['questions']})
    result['known_records_scope'] += '; plus all 384 revealed formal-round-2 Hotpot units; unknown exposure retained'
    mod.validate_exposures(result); return result

def build_pool(rows, exp, mod, *, size=512, seed=261113111):
    """Input-only grouping and fixed random order; no capacity/quality ranking."""
    mod.validate_exposures(exp)
    ids=set(exp['ids']);qhs=set(exp['question_group_hashes']);chs=set(exp['content_hashes'])
    seen=set(); groups={};blocked=set()
    for raw in rows:
        row=mod.validate_row(raw)
        require(row['id'] not in seen,'Duplicate source ID')
        seen.add(row['id']); qh=mod.question_group_sha256(row['question'])
        groups.setdefault(qh,[]).append(row)
        if row['id'] in ids or qh in qhs or mod.content_sha256(row) in chs: blocked.add(qh)
    eligible=[min(g,key=lambda r:r['id'].encode()) for qh,g in groups.items() if qh not in blocked]
    eligible.sort(key=lambda r:r['id'].encode())
    require(len(eligible)>=size,'Too few unexposed question groups')
    order=np.random.Generator(np.random.PCG64(seed)).permutation(len(eligible))[:size]
    pool=[eligible[int(i)] for i in order]
    return pool,{'source_rows':len(seen),'question_groups':len(groups),
        'duplicate_question_groups':sum(len(g)>1 for g in groups.values()),
        'excluded_question_groups':len(blocked),'unexposed_representatives':len(eligible),
        'pool_size':len(pool),'sampling_seed':seed,'labels_read':False,
        'unknown_cross_project_exposure':'UNKNOWN','pretraining_contamination_guarantee':False}

def select_capacity(pool, measures, *, n=96, seed=261113112):
    require(len(pool)==len(measures),'Capacity coverage differs')
    qualifying=[]
    for i,(row,m) in enumerate(zip(pool,measures)):
        require(m['index']==i and m['input_sha256']==digest(row),'Capacity order or bytes differ')
        require(type(m['ten_documents']) is bool,'Explicit document predicate required')
        if not m['ten_documents']: continue
        require(m['lengths'] and all(type(v) is int and v>0 for v in m['lengths'].values()),'Bad prompt lengths')
        require(set(m['lengths'])=={'SINGLE_Q','SINGLE_M','INDEPENDENT3_U','INDEPENDENT3_V','INDEPENDENT3_R','INDEPENDENT3_M'},'Static prompt coverage differs')
        if max(m['lengths'].values())<=7168: qualifying.append(i)
    require(len(qualifying)>=n,'Fixed input pool has fewer than 96 eligible units; no expansion')
    chosen=qualifying[:n]
    require(n%2==0,'Balanced shock needs even n')
    shocks=np.random.Generator(np.random.PCG64(seed)).permutation(['U']*(n//2)+['V']*(n//2)).tolist()
    return chosen,shocks,{'capacity_eligible_in_pool':len(qualifying),'selected_questions':n,
        'static_prompt_cap':7168,'dynamic_cap':8192,'static_headroom_is_proven_dynamic_bound':False,
        'shock_counts':{'U':shocks.count('U'),'V':shocks.count('V')},'shock_seed':seed}

def validate_plan(plan_path, plan_sha, mode):
    plan=pinned(plan_path,plan_sha)
    require(plan['schema']=='mab2_exogenous_input_plan_v1' and plan['root_decision']=='APPROVED_INPUT_ONLY','Input authority missing')
    require(plan['mode']==mode and plan['GPU_admission'] is False,'Wrong input-only action')
    require(plan['source_sha256']==sha(__file__),'Selector source differs')
    contract=pinned(plan['contract']['path'],plan['contract']['sha256'])
    require(contract['status']=='FROZEN' and contract['candidate']=='C2_FORMAL3_EXOGENOUS_PATH_SCREEN','Wrong frozen candidate')
    require(contract['population']['n_questions']==96 and contract['population']['input_pool_size']==512,'Wrong fixed counts')
    require(contract['seeds']['sampling']==261113111 and contract['seeds']['shock']==261113112,'Wrong selection seeds')
    return plan,contract

def run_pool(plan_path, plan_sha):
    start=time.monotonic();plan,contract=validate_plan(plan_path,plan_sha,'pool')
    out=Path(plan['output']);require(not out.exists(),'No overwrites')
    mod=legacy(plan['legacy_source'])
    source=pinned(plan['source_descriptor'],SOURCE_SHA)
    exp=union_exposures(pinned(plan['old_exposures'],OLD_EXPOSURES_SHA),pinned(plan['old_selection'],OLD_SELECTION_SHA),mod)
    paths=mod._source_preflight(source,Path(plan['source_descriptor']))
    pool,stats=build_pool(mod.iter_allowed_rows(paths),exp,mod)
    for p,meta in paths:require(sha(p)==meta['sha256'],'Source changed during input scan')
    validate_plan(plan_path,plan_sha,'pool')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'EXPOSURES_UNION.json',exp)
    payload=b''.join(canonical(row) for row in pool)
    with (out/'POOL.jsonl').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    receipt={'status':'COMPLETE_PRIVATE_INPUT_POOL_NO_LABELS','stats':stats,
        'pool_sha256':sha(out/'POOL.jsonl'),'exposures_union_sha256':sha(out/'EXPOSURES_UNION.json'),
        'source_sha256':sha(__file__),'plan_sha256':plan_sha,'contract_sha256':plan['contract']['sha256'],
        'legacy_source_sha256':LEGACY_SHA,'source_descriptor_sha256':SOURCE_SHA,
        'old_exposures_sha256':OLD_EXPOSURES_SHA,'old_selection_sha256':OLD_SELECTION_SHA,
        'labels_projected':False,'supporting_facts_projected':False,'GPU_hours':0,
        'seconds':time.monotonic()-start,'numpy_version':np.__version__}
    write(out/'RECEIPT.json',receipt);return receipt

def run_capacity(plan_path,plan_sha):
    # Must run in the qualified server's tokenizer-only environment.
    start=time.monotonic();plan,contract=validate_plan(plan_path,plan_sha,'capacity')
    out=Path(plan['output']);require(not out.exists(),'No overwrites')
    pool_receipt=pinned(plan['pool_receipt']['path'],plan['pool_receipt']['sha256'])
    require(pool_receipt['contract_sha256']==plan['contract']['sha256'],'Pool belongs to another contract')
    require(sha(plan['pool'])==pool_receipt['pool_sha256'],'Pool bytes differ')
    pool=[json.loads(line) for line in Path(plan['pool']).read_bytes().splitlines()]
    require(len(pool)==512,'Fixed pool size differs')
    for item in plan['sources'].values():require(sha(item['path'])==item['sha256'],'CPU helper source differs')
    spec=importlib.util.spec_from_file_location('_path_tokenizer_helper',plan['sources']['tokenizer']['path'])
    tkmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tkmod;spec.loader.exec_module(tkmod)
    spec=importlib.util.spec_from_file_location('_path_operator',plan['sources']['operator']['path'])
    op=importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
    out.mkdir(parents=True,exist_ok=False); toks={};identities={}
    for name in ('qwen','mistral'):
        dest=out/('private_tokenizer_'+name)
        copied=tkmod.copy_tokenizer_only(plan['models'][name],dest)
        toks[name],versions=tkmod.load_tokenizer(dest,plan['models'][name]['identity'])
        identities[name]=tkmod.tokenizer_identity(toks[name],copied,versions)
    measures=[]
    for i,row in enumerate(pool):
        ten=len(row['context']['title'])==len(row['context']['sentences'])==10
        lengths={}
        if ten:
            payload={k:row[k] for k in ('id','question','context')}
            for key,request in op.input_only_messages(payload).items():
                tok=toks[request['model']]
                ids=tok.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True)
                require(isinstance(ids,list),'Unexpected chat encoding output')
                lengths[key]=len(ids)
        measures.append({'index':i,'input_sha256':digest(row),'ten_documents':ten,'lengths':lengths})
    chosen,shocks,stats=select_capacity(pool,measures)
    selected=[pool[i] for i in chosen]
    write(out/'CAPACITY.json',{'measurements':measures,'tokenizer_identities':identities,
        'actual_server_input_encoding':True,'no_weights_or_labels':True})
    payload=b''.join(canonical(row) for row in selected)
    with (out/'QUESTIONS.jsonl').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    mod=legacy(plan['legacy_source'])
    manifest={'schema':'mab2_exogenous_path_selection_v1','contract_sha256':plan['contract']['sha256'],
        'pool_sha256':pool_receipt['pool_sha256'],'capacity_sha256':sha(out/'CAPACITY.json'),
        'questions_sha256':sha(out/'QUESTIONS.jsonl'),'generation_seeds':[26111311,26111312],
        'questions':[{'index':j,'pool_index':i,'id':row['id'],'input_sha256':digest(row),
            'question_group_sha256':mod.question_group_sha256(row['question']),
            'content_sha256':mod.content_sha256(row),'removed_upstream':shock}
            for j,(i,row,shock) in enumerate(zip(chosen,selected,shocks))]}
    write(out/'SELECTION_MANIFEST.json',manifest)
    validate_plan(plan_path,plan_sha,'capacity')
    require(sha(plan['pool'])==pool_receipt['pool_sha256'],'Pool changed during token counting')
    receipt={'status':'COMPLETE_PRIVATE_CAPACITY_SELECTION_NO_GPU_ADMISSION','stats':stats,
        'manifest_sha256':sha(out/'SELECTION_MANIFEST.json'),'questions_sha256':sha(out/'QUESTIONS.jsonl'),
        'capacity_sha256':sha(out/'CAPACITY.json'),'plan_sha256':plan_sha,
        'contract_sha256':plan['contract']['sha256'],'source_sha256':sha(__file__),
        'seconds':time.monotonic()-start,'new_GPU_hours':0,'labels_read':False,
        'dynamic_input_guarantee':False,'fewer_than_96_or_dynamic_overflow_rule':'stop_no_resample_no_quality_selection'}
    write(out/'RECEIPT.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pool','capacity'],required=True)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args()
    r=(run_pool if a.mode=='pool' else run_capacity)(a.plan,a.plan_sha256)
    print(json.dumps({'status':r['status'],'stats':r['stats'],'new_GPU_hours':0}))
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'FAILED_INPUT_PREPARATION_NO_AUTORETRY','error_type':type(e).__name__}),file=sys.stderr)
        raise SystemExit(2)
