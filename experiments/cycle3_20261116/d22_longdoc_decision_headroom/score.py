import argparse,json,hashlib,time
from pathlib import Path
import numpy as np

def analyze(f1,cost,prefix,terminal):
    assert f1.shape==(16,2,12) and cost.shape==f1.shape and prefix.shape==(16,2) and terminal.shape==(16,2,3)
    actions=[3,4,5];legal_cost=prefix[:,:,None]+terminal;legal_quality=f1[:,:,actions];legal_utility=legal_quality-5e-6*legal_cost;choice=legal_utility.argmax(2);selected=np.take_along_axis(legal_utility,choice[:,:,None],2)[:,:,0];quality=np.take_along_axis(legal_quality,choice[:,:,None],2)[:,:,0];tokens=np.take_along_axis(legal_cost,choice[:,:,None],2)[:,:,0];fixed=legal_utility.mean((0,1)).argmax();support=int(np.all(selected>legal_utility[:,:,0]+1e-12,axis=1).sum());direct=f1-5e-6*cost;controls={'FLAT':6,'SINGLE_Q':7,'SINGLE_M':8,'BM25':9,'DENSE':10,'VOTE_Q3':11};gains={n:float((selected-direct[:,:,j]).mean()) for n,j in controls.items()};fixed_gain=float((selected-legal_utility[:,:,fixed]).mean());seed_gains={n:(selected-direct[:,:,controls[n]]).mean(0).tolist() for n in ['DENSE','FLAT']};seed_gains['best_fixed']=(selected-legal_utility[:,:,fixed]).mean(0).tolist();gates=dict(mean_headroom_all_controls=all(v>=.01 for v in gains.values()),mean_headroom_best_fixed=fixed_gain>=.01,seed_consistency=all(all(x>0 for x in v) for v in seed_gains.values()),raw_quality=all(quality.mean()>=f1[:,:,controls[n]].mean()-.02 for n in ['DENSE','FLAT']),paired_paper_support=support>=4)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},oracle_utility=float(selected.mean()),oracle_F1=float(quality.mean()),oracle_tokens=float(tokens.mean()),oracle_gain_by_control=gains,oracle_gain_over_best_fixed=fixed_gain,best_fixed_legal_action=['keep','delete_U','delete_V'][fixed],seed_gains=seed_gains,paired_paper_support=support,selected_action_counts={n:int((choice==i).sum()) for i,n in enumerate(['keep','delete_U','delete_V'])}),choice,legal_utility

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--formal-run',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();run=Path(a.formal_run);out=Path(a.output);assert not out.exists();start=time.perf_counter();read=lambda f:json.loads(f.read_bytes());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert read(run/'SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_OFFICIAL_RAW_SCORING_AUDIT';arrays=np.load(run/'scoring_v1/PRIVATE_ARRAYS.npz');prefix=np.full((16,2),np.nan);terminal=np.full((16,2,3),np.nan)
    for unit in (run/'archive').glob('worker_*/unit_*'):
        i=read(unit/'INPUT.json')['index'];mat=read(unit/'MATRIX.json')
        for row in mat['conditions']:
            s=[26111691,26111692].index(row['execution_seed']);c=row['condition']
            if c=='G1':prefix[i,s]=sum(v['generation']['input_tokens']+v['generation']['output_tokens'] for n,v in row['nodes'].items() if n!='M')
            if c in ['G1','G1_MINUS_EU','G1_MINUS_EV']:
                g=row['nodes']['M']['generation'];terminal[i,s,['G1','G1_MINUS_EU','G1_MINUS_EV'].index(c)]=g['input_tokens']+g['output_tokens']
    assert np.isfinite(prefix).all() and np.isfinite(terminal).all();assert np.allclose(prefix+terminal[:,:,0],arrays['tokens'][:,:,3]);result,choice,u=analyze(arrays['f1'],arrays['tokens'],prefix,terminal);result.update(status='OPTIMISTIC_HEADROOM_ONLY' if result['passed'] else 'FREEZE_CURRENT_PREFIX_DELETION_DIRECTION',papers=16,seeds=2,all_observed_prefix_cost_paid=True,oracle_feedback_cost_omitted=True,new_GPU_hours=0,new_model_calls=0,independent_confirmation=False,source_sha256=sha(Path(__file__)),parent_summary_sha256=sha(run/'scoring_v1/SUMMARY.json'),cpu_main_wall_seconds=time.perf_counter()-start);out.mkdir();np.savez_compressed(out/'PRIVATE_CEILING.npz',choice=choice,legal_utility=u,prefix_tokens=prefix,terminal_tokens=terminal);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
