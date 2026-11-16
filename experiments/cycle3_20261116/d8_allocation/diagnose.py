"""Prospective exposed-pilot diagnostic; fixed lexical allocation features."""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib,re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

def features(row,shock):
    terms=lambda x:re.findall(r'\w+',x.lower(),flags=re.UNICODE)
    query=set(terms(row['question']));qcount=len(terms(row['question']))
    docs=[Counter(terms(t+' '+' '.join(s))) for t,s in zip(row['context']['title'],row['context']['sentences'])]
    lengths=np.array([sum(d.values()) for d in docs],float);avg=lengths.mean();scores=np.zeros(10)
    for word in query:
        df=sum(word in d for d in docs);idf=np.log1p((10-df+.5)/(df+.5))
        for i,d in enumerate(docs):
            tf=d.get(word,0);scores[i]+=idf*(tf*2.2)/(tf+1.2*(.25+.75*lengths[i]/max(avg,1e-8)))
    scores/=scores.max()+1e-8
    order=sorted(range(10),key=lambda i:(hashlib.sha256(row['context']['title'][i].encode()).hexdigest(),i))
    shards=dict(U=order[:3],V=order[3:6],R=order[6:]);other='V' if shock=='U' else 'U'
    rank=sorted(range(10),key=lambda i:(-scores[i],i));out=[];masses=[]
    for name in [shock,other,'R']:
        ids=shards[name];v=scores[ids];mass=sum(sorted(v,reverse=True)[:2]);masses.append(mass)
        out.extend([v.max(),v.mean(),mass,sum(i in ids for i in rank[:2])/2,sum(i in ids for i in rank[:4])/4,np.log1p(lengths[ids].sum())])
    out.extend([np.log1p(qcount),float(shock=='V')]);assert len(out)==20
    return out,1 if masses[0]<=masses[1] else 2

def diagnostic(rows,selection,arrays):
    assert len(rows)==96 and arrays['em'].shape==(96,2,11)
    names=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT_EVIDENCE','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3']
    x=[];lex=[];em=[];tokens=[]
    for i,row in enumerate(rows):
        assert row['id']==selection['questions'][i]['id'];shock=selection['questions'][i]['removed_upstream'];other='V' if shock=='U' else 'U'
        f,l=features(row,shock);x.append(f);lex.append(l)
        take=[names.index(n) for n in ['G1','G1_MINUS_E'+shock,'G1_MINUS_E'+other]]
        em.append(arrays['em'][i][:,take]);tokens.append(arrays['tokens'][i][:,take])
    x=np.array(x);em=np.array(em);tokens=np.array(tokens);utility=em-5e-6*tokens;targets=utility.mean(axis=1);advantages=targets-targets[:,[0]]
    split=np.array_split(np.random.Generator(np.random.PCG64(261116801)).permutation(96),6);fold=np.empty(96,int)
    chosen={k:np.zeros(96,int) for k in ['train_constant','ridge','tree']};chosen['lexical_delete']=np.array(lex)
    chosen.update({k:np.full(96,a) for a,k in enumerate(['fixed_keep','fixed_delete_affected','fixed_delete_other'])})
    for fid,test in enumerate(split):
        train=np.array([i for i in range(96) if i not in set(test)]);fold[test]=fid
        chosen['train_constant'][test]=int(np.argmax(targets[train].mean(axis=0)))
        scale=StandardScaler().fit(x[train]);ridge=Ridge(alpha=10,solver='svd').fit(scale.transform(x[train]),advantages[train]);chosen['ridge'][test]=np.argmax(ridge.predict(scale.transform(x[test])),axis=1)
        tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=12,random_state=261116802).fit(x[train],advantages[train]);chosen['tree'][test]=np.argmax(tree.predict(x[test]),axis=1)
    base=utility[np.arange(96),:,chosen['train_constant']];base_em=em[np.arange(96),:,chosen['train_constant']];summary={}
    for name,a in chosen.items():
        u=utility[np.arange(96),:,a];acc=em[np.arange(96),:,a];cost=tokens[np.arange(96),:,a];delta=u-base
        fd=[float(delta[fold==i].mean()) for i in range(6)];sd=delta.mean(axis=0).tolist()
        summary[name]=dict(EM=float(acc.mean()),service_utility=float(u.mean()),service_tokens=float(cost.mean()),utility_gain=float(delta.mean()),positive_folds=sum(v>0 for v in fd),fold_gains=fd,seed_gains=sd,
          EM_by_seed=acc.mean(axis=0).tolist(),corrections=int(((acc==1)&(base_em==0)).sum()),destructions=int(((acc==0)&(base_em==1)).sum()),action_counts=np.bincount(a,minlength=3).tolist(),
          registered_model_signal_gate=bool(delta.mean()>=.005 and sum(v>0 for v in fd)>=4 and all(v>0 for v in sd)) if name in ['ridge','tree'] else None)
    return dict(status='COMPLETE_EXPOSED_CPU_DIAGNOSTIC',question_groups=96,seeds=2,features=20,methods=summary,new_GPU_hours=0,bootstrap_runs=0,
      interpretation='Service-only exploratory signal; does not override Formal6 failure or establish feedback-budget gain',
      training_cost_included_in_service_utility=False,independent_confirmation=False),dict(features=x,folds=fold,em=em,tokens=tokens,**{'chosen_'+k:v for k,v in chosen.items()})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--questions',required=True);a.add_argument('--selection',required=True);a.add_argument('--arrays',required=True);a.add_argument('--scoring-audit',required=True);a.add_argument('--output',required=True);z=a.parse_args()
    assert json.loads(Path(z.scoring_audit).read_bytes())['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    rows=[json.loads(s) for s in Path(z.questions).read_bytes().splitlines()];selection=json.loads(Path(z.selection).read_bytes())
    result,private=diagnostic(rows,selection,np.load(z.arrays));result['pins']={key:hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in dict(questions=z.questions,selection=z.selection,arrays=z.arrays,scoring_audit=z.scoring_audit,source=__file__).items()}
    out=Path(z.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'PRIVATE_ARRAYS.npz',**private);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'signals':{k:v['registered_model_signal_gate'] for k,v in result['methods'].items() if k in ['ridge','tree']}}))
