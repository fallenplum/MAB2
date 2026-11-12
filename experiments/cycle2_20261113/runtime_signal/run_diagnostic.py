from pathlib import Path
import argparse,hashlib,importlib.util,io,json,time,sys,subprocess
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
import joblib
from runtime_features import first_round,features_for_edge,question_yields

INPUT_PIN='7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'
SCORER_PIN='da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e'
HEURISTICS=['sender_receiver_disagreement','receiver_incoming_disagreement','source_uncertainty','inverse_receiver_degree','uniform']
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    for name in ('repo','data-root','history-dir','d2-inputs','out','expected-commit'):p.add_argument('--'+name,required=True)
    args=p.parse_args();repo=Path(args.repo);root=Path(args.data_root);history=Path(args.history_dir);out=Path(args.out)
    assert Path(__file__).resolve().is_relative_to(repo.resolve()) and not out.exists()
    git=['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();assert head==args.expected_commit
    assert not subprocess.check_output(git+['status','--porcelain'],text=True).strip()
    out.mkdir(parents=True);start=time.time();cpu=time.process_time()
    dump(out/'START.json',{'status':'STARTED','source_commit':head,'new_gpu_calls':0,'new_formal_rounds':0,
         'source_sha256':{x.name:sha(x) for x in Path(__file__).parent.iterdir() if x.is_file()}})
    inv=json.loads((history/'INVENTORY.json').read_bytes());audit=json.loads((history/'SCAN_COMPLETE.json').read_bytes())
    assert audit['status']=='SINGLE_RAW_PASS_COMPLETE' and audit['r1_graph_invariance_checks_unique']==5600
    index_raw=(history/'SELECTED_EXECUTION_INDEX.jsonl').read_bytes();assert sha_bytes(index_raw)==inv['selected_index_sha256']
    raw=Path(args.d2_inputs).read_bytes();assert sha_bytes(raw)==INPUT_PIN
    archive=np.load(io.BytesIO(raw),allow_pickle=False)
    partitions=archive['partition'];keep=np.isin(partitions,['train','tune']);indices=np.flatnonzero(keep)
    parts=partitions[keep];qids=archive['question_ids'][keep].tolist()
    assert (parts=='train').sum()==200 and (parts=='tune').sum()==80
    left=archive['left'].reshape(400,2,12)[keep];right=archive['right'].reshape(400,2,12)[keep]
    masks=archive['masks'];rowq=archive['row_question']
    assert np.array_equal(rowq[left],rowq[right]) and np.all(rowq[left]==indices[:,None,None])
    assert np.all((masks[left]!=masks[right]).sum(-1)==1)
    for edge in range(12):assert not masks[left][:,:,edge,edge].any() and masks[right][:,:,edge,edge].all()
    selected={}
    for line in index_raw.splitlines():
        row=json.loads(line)
        assert row['partition'] in ('train','tune')
        selected.setdefault((row['question_id'],row['seed']),row)
    assert len(selected)==1400 and set(qids)=={q for q,s in selected}
    dump(out/'RAW_SELECTION_FROZEN.json',{'selection_rule':'first index occurrence per question/seed, no qualities',
        'index_sha256':sha_bytes(index_raw),'files':list(selected.values()),'raw_results_read':False})
    sp=importlib.util.spec_from_file_location('fixed_numeric_normalizer',repo/'src/graph_credit/scoring.py')
    assert sha(sp.origin)==SCORER_PIN
    scoring=importlib.util.module_from_spec(sp);sp.loader.exec_module(scoring)
    first={};bytes_read=0
    for number,((qid,seed),row) in enumerate(selected.items(),1):
        path=root/row['execution_path'];assert path.resolve().is_relative_to(root.resolve())
        b=path.read_bytes();assert sha_bytes(b)==row['execution_sha256'];bytes_read+=len(b)
        x=json.loads(b);assert x['question_id']==qid and x['seed']==seed
        nodes={f'r1:w{i}':x['nodes'][f'r1:w{i}'] for i in range(4)}
        first[qid,seed]=first_round(nodes,scoring.numeric)
        if number%200==0:print(json.dumps({'status':'FIRST_ROUND_ONLY_EXTRACTION','files':number,'total':1400}),flush=True)
    gx=np.empty((280,2,12,5,26));rx=np.empty((280,2,12,5,40));hx=np.empty((280,2,12,5,5))
    for q,qid in enumerate(qids):
        for anchor in range(2):
            for edge in range(12):
                for seed in range(5):
                    g,r,h=features_for_edge(masks[left[q,anchor,edge]],edge,*first[qid,2701+seed])
                    gx[q,anchor,edge,seed]=g;rx[q,anchor,edge,seed]=r;hx[q,anchor,edge,seed]=h
    assert np.isfinite(gx).all() and np.isfinite(rx).all() and np.isfinite(hx).all()
    np.savez_compressed(out/'PRIVATE_FEATURES.npz',graph=gx,runtime=rx,heuristics=hx,partition=parts,
        question_hash=np.array([hashlib.sha256(q.encode()).hexdigest() for q in qids]))
    # Only the training targets are materialized for fitting. No tune target is consulted below.
    train=parts=='train';tune=parts=='tune'
    qualities=archive['quality_by_seed']
    yt=(qualities[right[train]]!=qualities[left[train]]).astype(int)
    assert set(np.unique(yt))=={0,1}
    yflat=yt.ravel();counts=np.bincount(yflat);weights=len(yflat)/(2*counts[yflat])
    matrices={'graph_only':gx,'full':np.concatenate([gx,rx],axis=-1),'runtime_only':rx}
    predictions={};model_params={'learning_rate':.05,'max_iter':100,'max_leaf_nodes':15,'min_samples_leaf':100,
        'l2_regularization':1.,'early_stopping':False,'random_state':94173}
    with threadpool_limits(limits=4):
        for name,x in matrices.items():
            model=HistGradientBoostingClassifier(**model_params)
            model.fit(x[train].reshape(-1,x.shape[-1]),yflat,sample_weight=weights)
            pred=model.predict_proba(x[tune].reshape(-1,x.shape[-1]))[:,list(model.classes_).index(1)]
            predictions[name]=pred.reshape(80,2,12,5)
            joblib.dump(model,out/(name+'.joblib'))
    for i,name in enumerate(HEURISTICS):predictions[name]=hx[tune,...,i]
    np.savez_compressed(out/'PREDICTIONS_FROZEN.npz',**predictions)
    dump(out/'PREDICTIONS_FROZEN.json',{'npz_sha256':sha(out/'PREDICTIONS_FROZEN.npz'),
        'features_sha256':sha(out/'PRIVATE_FEATURES.npz'),'train_target_sha256':sha_bytes(yt.tobytes()),
        'model_parameters':model_params,'tune_targets_used_for_fitting':False,'tune_targets_scored_yet':False,
        'methods':list(predictions),'training_rows':len(yflat),'training_class_counts':counts.tolist()})
    # Evaluation starts only after immutable prediction artifact was persisted.
    yv=(qualities[right[tune]]!=qualities[left[tune]]).astype(int);archive.close()
    qy={name:question_yields(scores,yv) for name,scores in predictions.items()}
    means={name:float(v.mean()) for name,v in qy.items()}
    comparators=['graph_only',*HEURISTICS];best=max(comparators,key=lambda m:means[m]);bm=means[best]
    picks=np.random.default_rng(94174).integers(0,80,size=(10000,80))
    bootstrap=qy['full'][picks].mean(1)-np.stack([qy[m][picks].mean(1) for m in comparators]).max(0)
    ci=np.quantile(bootstrap,[.05,.95]).tolist()
    support=int(yv.reshape(80,-1).any(1).sum());positive=int((qy['full']-qy[best]>1e-12).sum())
    gates={'support_questions_at_least20':support>=20,'full_minus_uniform_at_least_point02':means['full']-means['uniform']>=.02,
        'full_minus_strongest_baseline_at_least_point01':means['full']-bm>=.01,
        'bootstrap90_lower_above0':ci[0]>0,'strict_improvement_questions_at_least20':positive>=20}
    metrics={name:{'mean_top6_event_yield':means[name],'auc':float(roc_auc_score(yv.ravel(),s.ravel())),
        'average_precision':float(average_precision_score(yv.ravel(),s.ravel()))} for name,s in predictions.items()}
    np.savez_compressed(out/'PRIVATE_EVALUATION.npz',target=yv,**{'question_yield_'+m:v for m,v in qy.items()})
    result={'status':'ELIGIBLE_FOR_INDEPENDENT_RUNTIME_REVALIDATION_DESIGN' if all(gates.values()) else 'NOT_ADMITTED_CURRENT_RUNTIME_SIGNAL',
        'evidence_class':'EXPLORATORY_OLD_TRAIN_TUNE_SINGLE_SEED_EVENT_ACQUISITION','project_stop':False,
        'new_gpu_hours':0,'new_formal_rounds':0,'question_counts':{'train':200,'tune':80},
        'tune_support_questions':support,'strict_positive_questions_vs_strongest':positive,'metrics':metrics,
        'strongest_fixed_baseline':best,'full_minus_uniform':means['full']-means['uniform'],
        'full_minus_strongest_baseline':means['full']-bm,'bootstrap90_full_minus_max_baseline':ci,'gates':gates,
        'input_sha256':INPUT_PIN,'historical_index_sha256':sha_bytes(index_raw),'historical_invariance_audit_sha256':sha(history/'SCAN_COMPLETE.json'),
        'raw_files_read':len(selected),'raw_bytes_read':bytes_read,'post_first_round_features_used':False,
        'R1_or_final_test_used':False,'prediction_freeze_sha256':sha(out/'PREDICTIONS_FROZEN.json'),
        'code_commit':head,'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),
        'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,
        'limitations':['Event acquisition is not population credit prediction or future graph-edit quality.',
            'Same-count call upper bound is not equal measured token or total-system expenditure.',
            'Old tune was previously exposed; question bootstrap is exploratory conditional on this fitted model.',
            'Per-edge anchors cannot be concatenated into one common current graph.']}
    dump(out/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ('status','metrics','full_minus_uniform','full_minus_strongest_baseline','bootstrap90_full_minus_max_baseline','gates','cpu_seconds','wall_seconds')}),flush=True)
if __name__=='__main__':main()
