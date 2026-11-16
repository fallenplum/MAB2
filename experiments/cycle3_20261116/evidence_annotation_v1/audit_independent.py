"""Independent graph, evidence, RNG, vote and cost reconstruction; no gold access."""
import argparse,json,hashlib,re,string,math
from pathlib import Path

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def norm(x):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in x.lower() if c not in string.punctuation)).split())

def audit(root,expected):
    manifest=json.loads((root/'ARCHIVE_MANIFEST.json').read_bytes());checks=0;units=0;calls=0
    for rel,h in manifest.items():
        f=(root/rel).resolve();assert f.is_relative_to(root.resolve()) and hashlib.sha256(f.read_bytes()).hexdigest()==h;checks+=1
    for unit in sorted(root.glob('worker_*/unit_*')):
        assert (unit/'COMPLETE.json').exists()
        raw=json.loads((unit/'INPUT.json').read_bytes());p=raw['payload'];assert set(p)=={'id','question','context'}
        m=json.loads((unit/'MATRIX.json').read_bytes());assert m['status']=='COMPLETE_UNSCORED' and len(m['conditions'])==22
        titles=p['context']['title'];ss=p['context']['sentences'];order=sorted(range(10),key=lambda i:(hashlib.sha256(titles[i].encode()).hexdigest(),i));shards=dict(U=order[:3],V=order[3:6],R=order[6:])
        signatures={};keys=set();total=0
        for row in m['conditions']:
            name=row['condition'];seed=row['execution_seed'];assert (name,seed) not in keys;keys.add((name,seed))
            graph=name.startswith('G0') or name.startswith('G1');flat=name=='FLAT_ANNOTATION';single=name in ['SINGLE_Q','SINGLE_M'];vote=name=='VOTE_Q3';indep=name=='INDEPENDENT3_Q'
            assert graph or flat or single or vote or indep
            edges={('U','R'),('V','R'),('R','M'),('U','M'),('V','M')} if graph else {('U','M'),('V','M'),('R','M')} if flat else set()
            if name.startswith('G1'):edges.remove((raw['shock'],'R'))
            if name.endswith('_MINUS_EU'):edges.remove(('U','M'))
            if name.endswith('_MINUS_EV'):edges.remove(('V','M'))
            assert set(map(tuple,row['edges']))==edges;checks+=1
            live={'M'}
            while True:
                new=live|{u for u,v in edges if v in live}
                if live==new:break
                live=new
            expected_nodes={'M','U','V'} if vote else {'U','V','R','M'} if indep else live
            assert set(row['nodes'])==expected_nodes;checks+=1
            costs={k:dict(calls=0,input_tokens=0,output_tokens=0,model_elapsed_seconds=0.,gpu_seconds=0.) for k in ['qwen','mistral']}
            for node,v in row['nodes'].items():
                req=v['request'];gen=v['generation'];model='mistral' if (node=='R' and not vote) or name=='SINGLE_M' else 'qwen'
                assert v['model']==req['model']==model and req['node']==node
                expected_seed=int(digest(['exogenous_path_rng_v1',p['id'],seed,node])[:16],16)%(2**31-1)
                assert req['seed']==expected_seed and v['request_fingerprint']==digest(req);checks+=2
                data=json.loads(req['messages'][1]['content']);assert set(data)=={'question','documents','evidence_notes'} and data['question']==p['question'];checks+=1
                with_ids=graph or flat or single
                indices=set(range(10)) if single or indep or vote else set(shards[node]) if node!='M' else set()
                if graph or flat:
                    for source,target in edges:
                        if target==node:indices.update(int(x[1:]) for x in row['nodes'][source]['evidence_ids'])
                docs=[dict(title=titles[i],sentences=ss[i],**({'document_id':f'D{i:02d}'} if with_ids else {})) for i in sorted(indices)]
                assert data['documents']==docs;checks+=1
                parents=[n for n in ['U','V','R'] if (n,node) in edges]
                if indep and node=='M':parents=['U','V','R']
                notes=[dict(source=n,text=row['nodes'][n]['generation']['text'].strip()) for n in parents]
                assert data['evidence_notes']==notes;checks+=1
                if (graph or flat) and node!='M':
                    assert v['evidence_ids']==[f'D{i:02d}' for i in sorted(indices)];checks+=1
                assert v['wire_message']==gen['text'].strip();checks+=1
                if with_ids:assert v['available_evidence_ids']==[f'D{i:02d}' for i in sorted(indices)];checks+=1
                cost=costs[model];cost['calls']+=1
                for key in ['input_tokens','output_tokens','gpu_seconds']:cost[key]+=gen[key]
                cost['model_elapsed_seconds']+=gen['elapsed_seconds']
                fp=v['request_fingerprint']
                if v['reused']:assert fp in signatures and signatures[fp]==gen;checks+=1
                else:assert fp not in signatures;signatures[fp]=gen;checks+=1
            if vote:
                order_vote=['M','U','V'];texts=[row['nodes'][n]['generation']['text'] for n in order_vote];normalized=[norm(t) for t in texts];best=sorted(range(3),key=lambda i:(-normalized.count(normalized[i]),i))[0]
                assert row['selected_node']==order_vote[best] and row['answer']==texts[best].strip();checks+=2
            else:assert row['answer']==row['nodes']['M']['generation']['text'].strip();checks+=1
            for model,cost in costs.items():
                for key,value in cost.items():assert math.isclose(row['logical_service_cost_by_model'][model][key],value,rel_tol=1e-12,abs_tol=1e-10);checks+=1
            total+=len(row['nodes'])
        assert total==72 and len(keys)==22;calls+=total;units+=1
    assert units==expected
    return dict(status='PASS_INDEPENDENT_PERMISSION_RNG_VOTE_COST_AUDIT',units=units,logical_calls=calls,checks=checks,labels_read=False,new_GPU_hours=0,shared_operator_imported=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--expected',type=int,required=True);a.add_argument('--output',required=True);x=a.parse_args();result=audit(Path(x.archive),x.expected)
    out=Path(x.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
