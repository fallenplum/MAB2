"""Second calculation using enumerated binary reward functions and separately formed graph laws."""
import argparse,itertools,json
from pathlib import Path
import numpy as np

def verify(folder,report):
    rows=json.loads((folder/'INPUT.json').read_bytes());labels=json.loads((folder/'SEALED.json').read_bytes())
    measurements=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in (0,1)]
    claimed=json.loads(report.read_bytes())['finite'];cells={};max_error=0.;cases=0
    for row in rows:
        k=row['key'];n=len(row['options']);y=labels[k]
        tests=np.asarray(list(itertools.product((0.,1.),repeat=n)))
        tables=[]
        for m in measurements:
            p=np.asarray(m[k]['probs']);p=p/p.sum(axis=1)[:,None];tables.append(p)
        for sender in (0,1):
            src=tables[sender][0];mid=tables[1-sender][1:].reshape(n+1,n+1,n)
            enabled=src[:,None]*mid[1:,0,:];disabled=src[:,None]*mid[0,0,:]
            score_on=enabled@tests.T;score_off=disabled@tests.T
            sharp_direct=np.max(score_on-score_off,axis=1).sum()
            sharp_relay=np.max(score_on.sum(0)-score_off.sum(0))
            for reader in (0,1):
                term=tables[reader][1:].reshape(n+1,n+1,n)[1:,1:,y]
                yes=sum(enabled[i,j]*term[i,j] for i in range(n) for j in range(n))
                no=sum(disabled[i,j]*term[i,j] for i in range(n) for j in range(n))
                assert abs(yes-no)<=sharp_direct+1e-10
                cells.setdefault(row['task']+'_'+str(reader),[]).append((sharp_direct,sharp_relay,abs(yes-no),sharp_direct<=.01))
                cases+=1
    for name,vals in cells.items():
        reference=np.array([claimed[name][field] for field in ('kw','km','actual_all_abs','all_certified')])
        max_error=max(max_error,abs(np.mean(vals,axis=0)-reference).max())
    assert max_error<1e-10
    return dict(questions=len(rows),role_reader_realizations=cases,largest_aggregate_discrepancy=max_error,
                verified_by='separate laws and exhaustive enumeration of binary reward functions; no primary scoring imports')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('report',type=Path)
    args=p.parse_args();print(json.dumps(verify(args.folder,args.report),indent=2))
