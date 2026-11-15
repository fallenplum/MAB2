"""Post-exposure diagnostic repair of A/B labels, never a new experiment.

Requires an explicit frozen repair receipt. Reads already frozen source graphs,
never prints text/identifiers, and writes only a fresh run_002 output directory.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from axis_mapping import verify_stays, CONDITION_BACKGROUND, BACKGROUND_AXIS

PLAN = '24c108292dbdfb95970edc509e776b5f94b92a7dcc56d780696cbe48b8e6a3ff'
DESCRIPTOR = '98ab9ec28259c4e08871a76e503f118d99190f7fa956e9712e116898c6c636b4'
SCORER = 'da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e'
SCIENCE = '5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf'
NODE_ORDER = tuple(f'r{r}:w{w}' for r in (1,2) for w in range(4)) + ('aggregate',)


def require(value, message):
    if not value: raise ValueError(message)


def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def load(path): return json.loads(Path(path).read_bytes())
def dump(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def canonical_digest(value):
    return sha_bytes(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    for name in ('freeze','repo','plan','descriptor','run-root','analysis','independent-review','old-run','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=Path(__file__).resolve().parent
    freeze=load(args.freeze)
    require(freeze['status']=='FROZEN_DIAGNOSTIC_REPAIR_AFTER_EXPOSURE','Root must freeze this diagnostic repair before real rescoring')
    require({'repair_reference.py','axis_mapping.py','original_private/reference.py','original_private/PROTOCOL.md'} <= set(freeze['source_sha256']), 'Repair freeze omits executable or original-protocol pins')
    for rel,pin in freeze['source_sha256'].items():
        path=(base/rel).resolve();require(path.is_relative_to(base),'Source path escapes stage')
        require(sha(path)==pin,'Frozen repair source hash differs: '+rel)
    require(freeze['original_protocol_target_unchanged'] is True and freeze['new_scientific_experiment'] is False,'Repair scope differs')
    require(args.out.name=='run_002' and not args.out.exists(),'Use fresh run_002, never overwrite')
    require(sha(args.plan)==PLAN and sha(args.descriptor)==DESCRIPTOR,'Plan/descriptor changed')
    science=args.repo/'protocol/cycle2_20261113/R1_PROTOCOL.json'
    scorer_path=args.repo/'src/graph_credit/scoring.py'
    require(sha(science)==SCIENCE and sha(scorer_path)==SCORER,'Frozen protocol/scorer changed')
    old=load(args.old_run/'RESULTS.json')
    require(sha(args.old_run/'RESULTS.json')==freeze['old_result_sha256'],'Old result identity differs')
    require(sha(base/'original_private/RESULTS.json')==freeze['old_result_sha256'],'Archived old result differs')
    original=load_module('arch_original_frozen_helpers',base/'original_private/reference.py')
    scorer=load_module('arch_repair_original_scorer',scorer_path)
    private=args.analysis/'private';complete=load(private/'COMPLETE.json')
    require(complete['status']=='COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION','R1 analysis incomplete')
    result_path=args.analysis/'public/R1_RESULTS.json'
    require(sha(result_path)==complete['public_result_sha256'],'R1 result binding differs')
    result=load(result_path);outputs_path=private/'OUTPUTS_FROZEN.json'
    require(sha(outputs_path)==result['provenance']['output_freeze_sha256'],'Output freeze binding differs')
    outputs=load(outputs_path)
    require(outputs['status']=='COMPLETE_LABEL_FREE' and outputs['collection_plan_sha256']==PLAN,'Wrong output freeze')
    replay=load(args.independent_review)
    require(replay['status']=='PASS' and replay['official_result_sha256']==sha(result_path),'Independent replay mismatch')
    array_raw=(private/'REPLAY_INPUTS.npz').read_bytes()
    require(sha_bytes(array_raw)==complete['replay_inputs_sha256'],'Replay array hash differs')
    with np.load(io.BytesIO(array_raw),allow_pickle=False) as bundle:
        service_quality=bundle['S_quality'];service_cost=bundle['S_cost']
    plan=load(args.plan);specs=load(science)['data']['episodes'];descriptor=load(args.descriptor)
    units=[[u for u in plan['units'] if u['phase']=='S' and u['episode']==e] for e in range(8)]
    require(all(len(row)==24 for row in units),'S unit coverage differs')
    args.out.mkdir(parents=True,exist_ok=False)
    cpu=time.process_time();wall=time.perf_counter();score_count=0
    dump(args.out/'BEFORE_S_READ.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'repair_freeze_sha256':sha(args.freeze),'old_result_sha256':freeze['old_result_sha256'],
         'original_protocol_sha256':sha(base/'original_private/PROTOCOL.md'),
         'R1_result_sha256':sha(result_path),'post_exposure_diagnostic_repair':True,
         'new_scientific_experiment':False,'new_generation_calls':0})
    try:
        entry=descriptor['phases']['S'];labelpath=(args.descriptor.parent/entry['path']).resolve()
        require(labelpath.is_relative_to(args.descriptor.parent.resolve()),'S descriptor path escapes root')
        labelraw=labelpath.read_bytes()
        require(sha_bytes(labelraw)==entry['sha256'],'S descriptor binding differs')
        labels={r['id']:r for r in (json.loads(line) for line in labelraw.splitlines())}
        require(len(labels)==192 and set(labels)=={u['question_id'] for row in units for u in row},'S labels do not match fixed units')
        quality=np.zeros((8,24,7));valid=np.zeros((8,24,5),int);costs=np.zeros((8,7,3),np.int64)
        rawq={b:np.zeros((8,24)) for b in ('A','B')};rawc={b:np.zeros((8,3),np.int64) for b in ('A','B')}
        pins={}
        for e,row in enumerate(units):
            for q,u in enumerate(row):
                graph={}
                for background in ('A','B'):
                    requests=[r for r in plan['episodes'][e]['menu'] if r['background']==background and r['action']=='stay']
                    require(len(requests)==1,'Background does not identify one stay graph')
                    request=requests[0]
                    require(request['mask']==specs[e][background],'Stay background mask differs from scientific protocol')
                    rel=f"worker_{u['worker_index']}/units/{u['unit_id']}/{request['name']}.json"
                    path=(args.run_root/rel).resolve();require(path.is_relative_to(args.run_root.resolve()),'Graph path escapes source')
                    raw=path.read_bytes();pin=sha_bytes(raw)
                    require(pin==outputs['all_artifact_hashes'][rel],'Original graph hash changed')
                    pins[rel]=pin;record=json.loads(raw);x=record['execution']
                    require(record['unit_id']==u['unit_id'] and record['request']==request,'Graph request binding differs')
                    require(canonical_digest(x)==record['execution_value_sha256'],'Graph payload digest differs')
                    require(x['question_id']==u['question_id'] and x['seed']==u['seed'] and x['mask']==request['mask'],'Graph question/seed/mask differs')
                    require(set(x['nodes'])==set(NODE_ORDER),'Incomplete nine-node graph')
                    graph[background]=x
                    rawq[background][e,q]=scorer.score_item(x['nodes']['aggregate']['generation']['text'],labels[u['question_id']],task='gsm8k');score_count+=1
                    for node in NODE_ORDER:
                        generation=x['nodes'][node]['generation']
                        require(type(generation['input_tokens']) is int and type(generation['output_tokens']) is int,'Noninteger token evidence')
                        require(generation['input_tokens']>=0 and generation['output_tokens']>=0,'Negative token evidence')
                        rawc[background][e]+=[generation['input_tokens'],generation['output_tokens'],1]
                parsed=[]
                for w in range(4):
                    node=graph['A']['nodes'][f'r1:w{w}'];other=graph['B']['nodes'][f'r1:w{w}']
                    require(node['input_fingerprint']==other['input_fingerprint'] and node['generation']==other['generation'],'Shared first-round node differs')
                    generation=node['generation'];parsed.append(scorer.extract_answer(generation['text']))
                    valid[e,q,w]=parsed[-1] is not None
                    quality[e,q,w]=scorer.score_item(generation['text'],labels[u['question_id']],task='gsm8k');score_count+=1
                    costs[e,w]+=[generation['input_tokens'],generation['output_tokens'],1]
                voted=original.vote(parsed);valid[e,q,4]=voted is not None
                quality[e,q,4]=scorer.score_item(original.vote_text(voted),labels[u['question_id']],task='gsm8k');score_count+=1
            costs[e,4]=costs[e,:4].sum(axis=0)
        mapping=verify_stays(rawq,rawc,service_quality,service_cost)
        for j,background in ((5,'A'),(6,'B')):
            quality[:,:,j]=rawq[background];costs[:,j]=rawc[background]
        require(score_count==1344,'Expected 960 original reference + 384 aggregate verification scores')
        names=['worker0','worker1','worker2','worker3','vote4','A_stay','B_stay']
        episode=quality.mean(axis=1);indices=np.random.default_rng(95173).integers(0,8,size=(10000,8));contrasts=[]
        for j in (5,6):
            for b in (0,4):
                delta=episode[:,j]-episode[:,b]
                contrasts.append({'contrast':names[j]+' - '+names[b],'mean':float(delta.mean()),
                                  'ci95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),
                                  'episode_differences':delta.tolist()})
        methods={name:{'accuracy':float(episode[:,j].mean()),'episode_accuracies':episode[:,j].tolist(),
                      'input_tokens':int(costs[:,j,0].sum()),'output_tokens':int(costs[:,j,1].sum()),
                      'calls':int(costs[:,j,2].sum()),'parse_valid_fraction':float(valid[:,:,j].mean()) if j<5 else None}
                 for j,name in enumerate(names)}
        for name in names[:5]: require(methods[name]==old['methods'][name],'Unchanged first-round reference differs')
        require(methods['A_stay']==old['methods']['B_stay'] and methods['B_stay']==old['methods']['A_stay'],'Old-vs-repaired label relation differs; investigate before release')
        for rel,pin in freeze['source_sha256'].items(): require(sha(base/rel)==pin,'Repair source changed during execution')
        require(sha(scorer_path)==SCORER,'Frozen scorer changed during execution')
        np.savez_compressed(args.out/'PRIVATE_ARRAYS.npz',quality=quality,costs=costs,parse_valid=valid)
        dump(args.out/'SOURCE_GRAPH_PINS.json',pins)
        dump(args.out/'MAPPING_VERIFICATION.json',{**mapping,'source_graphs_verified':len(pins),
             'raw_aggregate_recomputed_with_frozen_scorer':True,'all_nine_node_token_counts_recomputed':True,
             'R1_main_results_unchanged':True,'old_A_label_really_B':True,'old_B_label_really_A':True})
        new={**old,'status':'DESCRIPTIVE_ARCHITECTURE_REFERENCE_REPAIRED_AFTER_EXPOSURE_NOT_A_GATE',
             'methods':methods,'contrasts':contrasts,'new_score_calls':score_count,
             'new_model_calls':0,'new_gpu_hours':0,'new_formal_rounds':0,
             'repair_freeze_sha256':sha(args.freeze),'repair_source_sha256':sha(__file__),
             'old_result_sha256':freeze['old_result_sha256'],'original_protocol_target_unchanged':True,
             'post_exposure_diagnostic_repair':True,'new_scientific_experiment':False,
             'condition_background':CONDITION_BACKGROUND,'background_axis':BACKGROUND_AXIS,
             'original_result_source_commit':old['source_commit'],
             'source_graphs_sha256':sha_bytes(json.dumps(pins,sort_keys=True).encode()),
             'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall}
        dump(args.out/'RESULTS.json',new)
        dump(args.out/'COMPLETE.json',{'status':'COMPLETE_DIAGNOSTIC_REPAIR_NOT_NEW_EXPERIMENT',
             'result_sha256':sha(args.out/'RESULTS.json'),'mapping_verification_sha256':sha(args.out/'MAPPING_VERIFICATION.json'),
             'new_score_calls':score_count,'new_model_calls':0,'new_GPU_hours':0})
        print(json.dumps({'status':new['status'],'graphs_verified':384,'new_score_calls':score_count,'new_model_calls':0}))
    except Exception as exc:
        dump(args.out/'FAILED_RECEIPT.json',{'status':'FAILED_REPAIR_KEEP_ARTIFACTS','error_type':type(exc).__name__,
             'scores_completed':score_count,'cpu_seconds':time.process_time()-cpu,'new_model_calls':0,'new_GPU_hours':0})
        raise


if __name__=='__main__': main()
