from pathlib import Path
from collections import Counter
import argparse,datetime,hashlib,importlib.util,io,json,subprocess,sys,time
import numpy as np

PLAN='24c108292dbdfb95970edc509e776b5f94b92a7dcc56d780696cbe48b8e6a3ff'
DESCRIPTOR='98ab9ec28259c4e08871a76e503f118d99190f7fa956e9712e116898c6c636b4'
SCORER='da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def vote(answers):
    good=[x for x in answers if x is not None]
    if not good:return None
    counts=Counter(good);maximum=max(counts.values())
    return next(x for x in answers if x is not None and counts[x]==maximum)

def vote_text(value):
    return format(value,'f') if value is not None else ''

def main():
    p=argparse.ArgumentParser()
    for arg in ('repo','plan','descriptor','run-root','analysis','independent-review','out','expected-commit'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();repo=Path(a.repo);analysis=Path(a.analysis);out=Path(a.out);run_root=Path(a.run_root)
    assert not out.exists() and Path(__file__).resolve().is_relative_to(repo.resolve())
    git=['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();assert head==a.expected_commit
    assert not subprocess.check_output(git+['status','--porcelain'],text=True).strip()
    cpu=time.process_time();start=time.time();pr=analysis/'private'
    complete=load(pr/'COMPLETE.json');assert complete['status']=='COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION'
    results_path=analysis/'public/R1_RESULTS.json';assert sha(results_path)==complete['public_result_sha256']
    primary=load(results_path);freeze_path=pr/'OUTPUTS_FROZEN.json'
    assert sha(freeze_path)==primary['provenance']['output_freeze_sha256'];freeze=load(freeze_path)
    assert freeze['status']=='COMPLETE_LABEL_FREE' and freeze['collection_plan_sha256']==PLAN
    assert load(a.independent_review)['status']=='PASS'
    assert sha(a.plan)==PLAN and sha(a.descriptor)==DESCRIPTOR
    plan=load(a.plan);descriptor=load(a.descriptor)
    unit_order=[[u for u in plan['units'] if u['phase']=='S' and u['episode']==ep] for ep in range(8)]
    assert all(len(u)==24 for u in unit_order)
    replay=pr/'REPLAY_INPUTS.npz';assert sha(replay)==complete['replay_inputs_sha256']
    with np.load(replay,allow_pickle=False) as f:S=f['S_quality'].copy();service=f['S_cost'].copy()
    scorerpath=repo/'src/graph_credit/scoring.py';assert sha(scorerpath)==SCORER
    sp=importlib.util.spec_from_file_location('architecture_original_scorer',scorerpath)
    scorer=importlib.util.module_from_spec(sp);sp.loader.exec_module(scorer)
    out.mkdir(parents=True)
    dump(out/'BEFORE_S_READ.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),'source_commit':head,
        'R1_complete_sha256':sha(pr/'COMPLETE.json'),'R1_result_sha256':sha(results_path),
        'R1_independent_review_sha256':sha(a.independent_review),'new_generation_calls':0})
    entry=descriptor['phases']['S'];labelpath=(Path(a.descriptor).parent/entry['path']).resolve()
    assert labelpath.is_relative_to(Path(a.descriptor).parent.resolve()) and sha(labelpath)==entry['sha256']
    labels=[json.loads(b) for b in labelpath.read_bytes().splitlines()];labels={r['id']:r for r in labels}
    assert len(labels)==192 and set(labels)=={u['question_id'] for row in unit_order for u in row}
    quality=np.zeros((8,24,7));valid=np.zeros((8,24,5),int);costs=np.zeros((8,7,3),np.int64)
    pins={};score_count=0
    for ep,units in enumerate(unit_order):
        for qi,u in enumerate(units):
            menu=plan['episodes'][ep]['menu'];A=next(r for r in menu if r['background']=='A' and r['action']=='stay')
            B=next(r for r in menu if r['background']=='B' and r['action']=='stay')
            graphs=[]
            for request in (A,B):
                rel=f"worker_{u['worker_index']}/units/{u['unit_id']}/{request['name']}.json"
                path=run_root/rel;assert path.resolve().is_relative_to(run_root.resolve())
                assert sha(path)==freeze['all_artifact_hashes'][rel];pins[rel]=sha(path)
                record=load(path);assert record['unit_id']==u['unit_id'] and record['request']==request
                graphs.append(record['execution'])
            parsed=[]
            for w in range(4):
                node=graphs[0]['nodes'][f'r1:w{w}'];other=graphs[1]['nodes'][f'r1:w{w}']
                assert node['input_fingerprint']==other['input_fingerprint'] and node['generation']==other['generation']
                g=node['generation'];parsed.append(scorer.extract_answer(g['text']));valid[ep,qi,w]=parsed[-1] is not None
                quality[ep,qi,w]=scorer.score_item(g['text'],labels[u['question_id']],task='gsm8k');score_count+=1
                costs[ep,w]+=[g['input_tokens'],g['output_tokens'],1]
            chosen=vote(parsed);valid[ep,qi,4]=chosen is not None
            quality[ep,qi,4]=scorer.score_item(vote_text(chosen),labels[u['question_id']],task='gsm8k');score_count+=1
            quality[ep,qi,5]=S[ep,0,qi,0];quality[ep,qi,6]=S[ep,1,qi,0]
        costs[ep,4]=costs[ep,:4].sum(0)
        costs[ep,5]=service[ep,0,0,:3].astype(np.int64)
        costs[ep,6]=service[ep,1,0,:3].astype(np.int64)
    assert score_count==960 and (costs[:,:4,2]==24).all() and (costs[:,4,2]==96).all() and (costs[:,5:,2]==216).all()
    names=['worker0','worker1','worker2','worker3','vote4','A_stay','B_stay'];episode=quality.mean(1)
    picks=np.random.default_rng(95173).integers(0,8,size=(10000,8));contrasts=[]
    for m in (5,6):
        for b in (0,4):
            d=episode[:,m]-episode[:,b];ci=np.quantile(d[picks].mean(1),[.025,.975])
            contrasts.append({'contrast':names[m]+' - '+names[b],'mean':float(d.mean()),'ci95':ci.tolist(),'episode_differences':d.tolist()})
    np.savez_compressed(out/'PRIVATE_ARRAYS.npz',quality=quality,costs=costs,parse_valid=valid)
    result={'status':'DESCRIPTIVE_ARCHITECTURE_REFERENCE_COMPLETE_NOT_A_GATE','source_commit':head,
        'methods':{name:{'accuracy':float(episode[:,j].mean()),'episode_accuracies':episode[:,j].tolist(),
            'input_tokens':int(costs[:,j,0].sum()),'output_tokens':int(costs[:,j,1].sum()),'calls':int(costs[:,j,2].sum()),
            'parse_valid_fraction':float(valid[:,:,j].mean()) if j<5 else None} for j,name in enumerate(names)},
        'contrasts':contrasts,'new_model_calls':0,'new_gpu_hours':0,'new_score_calls':score_count,
        'new_formal_rounds':0,'cost_scope':'LOGICAL_SERVICE_ONLY_NOT_COMPLETE_POLICY_BUDGET',
        'S_role':'R1_DEVELOPMENT_SERVICE_NOT_FINAL_TEST','new_method_selected':False,'project_stop':False,
        'source_graphs_sha256':hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),'scorer_sha256':SCORER,
        'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,
        'limitations':['8 related graph episodes; descriptive intervals only.',
            'Prompts and aggregation differ, so differences do not isolate communication causally.',
            'Four role instances use one model; no heterogeneous pool claim.',
            'No original R1 criterion or metric was changed.']}
    dump(out/'SOURCE_GRAPH_PINS.json',pins);dump(out/'RESULTS.json',result)
    print(json.dumps({'status':result['status'],'methods':result['methods'],'contrasts':contrasts}))
if __name__=='__main__':main()
