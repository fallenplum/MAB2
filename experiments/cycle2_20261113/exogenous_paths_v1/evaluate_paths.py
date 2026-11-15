"""Frozen point-only answer evaluation after complete archive/cost/replay release."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def collect(selection,matrices):
    from point_statistics import ORDER
    from exogenous_path_operator import digest
    rows=selection['questions'];assert len(rows)==96 and len(matrices)==96
    assert len({r['id'] for r in rows})==96
    seeds=selection['generation_seeds'];assert seeds==[26111311,26111312]
    predictions=[];costs=[];finish={};blank=0
    for index,((inp,m),row) in enumerate(zip(matrices,rows)):
        assert row['index']==inp['index']==index and inp['payload']['id']==row['id']
        assert inp['shock']==row['removed_upstream']==m['removed_upstream']
        assert m['question_fingerprint']==digest(inp['payload'])
        assert m['status']=='COMPLETE_UNSCORED' and m['complete_terminal_conditions']==18
        assert m['execution_seeds']==seeds
        assert [c['key'] for c in m['conditions']]==[f'{s}/{name}' for s in seeds for name in ORDER]
        assert sum(x['calls'] for x in m['logical_completed_node_cost_by_model'].values())==60
        answer=[]
        for c in m['conditions']:
            assert c['status']=='COMPLETE'
            raw=c['nodes']['M']['generation'];assert c['answer']==raw['text'].strip()
            assert raw['finish_reason'] in ['eos','length']
            finish[raw['finish_reason']]=finish.get(raw['finish_reason'],0)+1
            blank+=not bool(c['answer']);answer.append(c['answer'])
            costs.append({'question_index':index,'condition':c['key'],'by_model':c['logical_service_cost_by_model']})
        predictions.append(answer)
    return predictions,costs,{'finish_counts':finish,'blank_final_outputs':blank}
def evaluate(selection,matrices,gold_loader):
    # Input/terminal validation must finish before invoking a gold reader.
    preds,costs,meta=collect(selection,matrices)
    vault=gold_loader();ids=[r['id'] for r in selection['questions']]
    assert set(vault)==set(ids) and all(isinstance(v,str) for v in vault.values())
    from hotpot_support import answer_metrics
    from point_statistics import summarize,ORDER
    em=np.empty((96,2,9));f1=np.empty_like(em)
    for i,(qid,answers) in enumerate(zip(ids,preds)):
        for j,answer in enumerate(answers):
            score=answer_metrics(answer,vault[qid]);em[i,j//9,j%9]=score['em'];f1[i,j//9,j%9]=score['f1']
    result=summarize(em,[r['removed_upstream'] for r in selection['questions']])
    result.update(meta);result['descriptive_F1_by_condition_seed']={n:f1[:,:,j].mean(axis=0).tolist() for j,n in enumerate(ORDER)}
    # Task scope is one fixed benchmark; seeds are not independent tasks.
    result['task']='HotpotQA distractor train, controlled evidence partition and path deletion'
    aggregates={}
    for row in costs:
        for model,values in row['by_model'].items():
            key=row['condition']+'/'+model
            if key not in aggregates:aggregates[key]={k:0 for k in values}
            for k,v in values.items():
                assert type(v) in [int,float] and np.isfinite(v) and v>=0
                aggregates[key][k]=max(aggregates[key][k],v) if k=='peak_allocated_bytes' else aggregates[key][k]+v
    result['full_logical_cost_by_condition_model']=aggregates
    return result,{'em':em.tolist(),'f1':f1.tolist(),'ids':ids,'cost_rows':costs}
def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--approval-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert sha(a.approval)==a.approval_sha;approval=read(a.approval)
    assert approval['decision']=='APPROVED_COMPLETE_ARCHIVE_POINT_SCORING'
    assert approval['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    for path,pin in approval['source_pins'].items():assert sha(path)==pin
    for path in reversed(approval['import_paths']):sys.path.insert(0,path)
    for name in ['selection','archive_manifest','replay_receipt','cost_closure']:
        item=approval[name];assert sha(item['path'])==item['sha256']
    assert read(approval['replay_receipt']['path'])['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY'
    assert read(approval['cost_closure']['path'])['actual_cost_closed'] is True
    selection=read(approval['selection']['path']);root=Path(approval['archive_root']);manifest=read(approval['archive_manifest']['path'])
    for rel,pin in manifest.items():
        f=(root/rel).resolve();assert f.is_relative_to(root.resolve()) and sha(f)==pin
    matrices=[]
    for i in range(96):
        matches=list(root.glob(f'worker_*/unit_{i:03d}/MATRIX.json'));assert len(matches)==1
        matrices.append((read(matches[0].parent/'INPUT.json'),read(matches[0])))
    def labels():
        spec=approval['vault'];assert sha(spec['path'])==spec['sha256'];return read(spec['path'])
    result,private=evaluate(selection,matrices,labels)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,obj in [('SUMMARY.json',result),('PRIVATE_ARRAYS.json',private)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    (out/'RECEIPT.json').write_text(json.dumps({'status':'COMPLETE_POINT_EVALUATION','approval_sha256':a.approval_sha,
        'summary_sha256':sha(out/'SUMMARY.json'),'private_arrays_sha256':sha(out/'PRIVATE_ARRAYS.json'),
        'bootstrap_runs':0,'new_model_calls':0},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'COMPLETE_POINT_EVALUATION','primary_D':result['primary_D'],'screen_action':result['screen_action']}))
if __name__=='__main__':main()
