"""Second implementation from accepted G1 rows; does not import production math."""
import argparse,collections,hashlib,itertools,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):
    v=np.array(v,float);rng=np.random.default_rng(83019)
    # One allocation is equivalent to the production chunked RNG stream.
    means=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return [float(v.mean()),*np.quantile(means,[.025,.975,.05,.95]).tolist()]

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--result',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=a.parse_args()
    start=time.process_time();auditp=a.root/'artifacts/audit/INDEPENDENT_G1_ACCEPTANCE.json'
    assert sha(auditp)=='e52e02e9c11ee722f43b5e9621c13b06bbb8aeb858480a8930213729cbf1f745'
    audit=json.loads(auditp.read_bytes());groups=collections.defaultdict(dict);rows=0
    for rel,pin in audit['input_sha256'].items():
        if not(rel.startswith('results/pilot/motivation_v3_20261103/shard_') and rel.endswith('/observations.jsonl')):continue
        p=a.root/rel;assert sha(p)==pin
        with p.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);rows+=1;e,f_=r['edge_e'],r['edge_f'];mask=r['graph_mask']
                key=(r['question_id'],r['context_id'],e,f_);corner=(mask[e],mask[f_],r['seed'])
                value=(r['quality'],r['logical_service_cost']['input_tokens']+r['logical_service_cost']['output_tokens'],r['logical_service_cost']['calls'])
                assert corner not in groups[key] or groups[key][corner]==value
                groups[key][corner]=value
    assert rows==60000 and len(groups)==3000
    allrows=collections.defaultdict(list);selected=collections.defaultdict(list);supports=[0,0]
    for (qid,context,e,f_),world in sorted(groups.items()):
        assert len(world)==20
        vecs={k:[] for k in ['a','b','L','h','joint_token_saving','joint_call_saving']}
        for seed in [1701,1702,1703,1704,1705]:
            x00,x01,x10,x11=[world[(*c,seed)] for c in [(0,0),(0,1),(1,0),(1,1)]]
            vecs['a'].append(x11[0]-x01[0]);vecs['b'].append(x11[0]-x10[0]);vecs['L'].append(x11[0]-x00[0])
            vecs['h'].append(x11[0]-x10[0]-x01[0]+x00[0]);vecs['joint_token_saving'].append(x11[1]-x00[1]);vecs['joint_call_saving'].append(x11[2]-x00[2])
        row={k:sum(v)/5 for k,v in vecs.items()}
        for k in ['a','b','h']:row['u_'+k]=sum(vecs[k][i]*vecs[k][j] for i,j in itertools.combinations(range(5),2))/10
        row['h_same_seed_square']=sum(v*v for v in vecs['h'])/5;row['h_mean_square']=row['h']**2
        allrows[qid].append((e,f_,row))
        eligible=[]
        for fold,(train,test) in enumerate([((0,1),(2,3,4)),((2,3,4),(0,1))]):
            if all(abs(sum(vecs[k][s] for s in train)/len(train))<=.05 for k in ['a','b']):
                supports[fold]+=1;eligible.append({k:sum(vecs[k][s] for s in test)/len(test) for k in ['a','b','L','h']})
        if eligible:selected[qid].append((e,f_,{k:sum(t[k] for t in eligible)/len(eligible) for k in ['a','b','L','h']}))
    expected=json.loads(a.result.read_bytes())['G1_original_declared_faces'];checks=[]
    def compare(got,want,label):
        err=float(np.max(np.abs(np.asarray(got)-np.asarray(want))))
        assert err<1e-12,(label,err)
        checks.append({'check':label,'max_absolute_error':err})
    compare([len(allrows),len(selected),*supports],[expected['complete_questions'],expected['selected_distinct_questions'],*expected['selected_face_counts_by_fold']],'support')
    for pair in [None,(0,3),(0,1),(0,6)]:
        # Enumerate actual represented pairs rather than assuming naming conventions.
        if pair is not None:continue
        for subset,label in [(allrows,'all_faces_question_weighted'),(selected,'zero_single_screen_crossfit_question_weighted')]:
            keys=sorted(next(iter(subset.values()))[0][2])
            for k in keys:
                values=[sum(x[2][k] for x in cells)/len(cells) for q,cells in sorted(subset.items())]
                m=expected[label][k];compare(stats(values),[m['point'],*m['interval95'],*m['interval90']],label+'/'+k)
    pairs=sorted({(e,f_) for cells in allrows.values() for e,f_,_ in cells})
    for pair in pairs:
        target=expected['by_edge_pair'][','.join(map(str,pair))]
        for subset,label in [(allrows,'all_faces_question_weighted'),(selected,'zero_single_screen_crossfit_question_weighted')]:
            filtered={q:[x[2] for x in cells if x[:2]==pair] for q,cells in subset.items()}
            filtered={q:cells for q,cells in filtered.items() if cells}
            for k,m in target[label].items():
                v=[sum(x[k] for x in cells)/len(cells) for q,cells in sorted(filtered.items())]
                compare(stats(v),[m['point'],*m['interval95'],*m['interval90']],str(pair)+'/'+label+'/'+k)
    out={'status':'PASS','scope':'Independent scalar formulas and original declared G1 rows; all main and pair metrics/support/intervals. No production math imports. Root authored second implementation, not separate human review.',
         'source_code_sha256':sha(Path(__file__)),'official_result_sha256':sha(a.result),'checks':checks,
         'cpu_seconds':time.process_time()-start,'new_gpu_hours':0,'real_r1_or_final_test_read':False}
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps({'status':'PASS','checks':len(checks),'max_absolute_error':max(x['max_absolute_error'] for x in checks),'cpu_seconds':out['cpu_seconds']}))

if __name__=='__main__':main()
