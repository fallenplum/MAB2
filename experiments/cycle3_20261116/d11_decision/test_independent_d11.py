"""Artificial decision-table verification; no benchmark labels or GPU calls."""
import copy,json
import numpy as np
from audit_diagnosis import verify
from diagnose import diagnose

def main():
    rng=np.random.default_rng(2611161101)
    names=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_ANNOTATION','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
    checks=0;rejections=0
    for case in range(21):
        n=7;shocks=['U' if i%2 else 'V' for i in range(n)]
        em=rng.integers(0,2,size=(n,2,11)).astype(float);f1=rng.random((n,2,11));tokens=rng.integers(0,150000,size=(n,2,11))
        if case==0:em[:]=0;f1[:]=0;tokens[:]=0
        permutation=rng.permutation(11);ordered=[names[i] for i in permutation]
        em=em[:,:,permutation];f1=f1[:,:,permutation];tokens=tokens[:,:,permutation]
        summary,_=diagnose(em,f1,tokens,shocks,ordered)
        checks+=verify(em,f1,tokens,shocks,ordered,summary)
        for kind in ['seed','direction','cost']:
            broken=copy.deepcopy(summary)
            if kind=='seed':broken['strategies']['KEEP']['EM_by_seed'][0]+=.1
            elif kind=='direction':broken['net_edge_direction']['other']['wrong_keep']+=1
            else:broken['strategies']['CURRENT_PERFECT_INFORMATION_ORACLE']['service_tokens']+=1
            try:verify(em,f1,tokens,shocks,ordered,broken)
            except AssertionError:rejections+=1
            else:raise AssertionError('Corrupted scientific summary was accepted')
    assert rejections==63
    print(json.dumps(dict(status='PASS_ARTIFICIAL_SCALAR_AUDIT_AND_CORRUPTION_REJECTION',tables=21,scalar_checks=checks,corruptions_rejected=rejections,new_GPU_calls=0)))

if __name__=='__main__':main()
