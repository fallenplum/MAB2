"""First-round only features, with no gold and no intervention endpoint access."""
import itertools,math
import numpy as np
EDGES=list(itertools.permutations(range(4),2))

def first_round(nodes,numeric):
    """nodes is an allowlisted four-node dictionary; no other stages accepted."""
    if set(nodes)!={f'r1:w{i}' for i in range(4)}:raise ValueError('First-round-only boundary')
    values=[];features=[]
    for i in range(4):
        n=nodes[f'r1:w{i}'];p=n['parsed'];g=n['generation'];meta=n['message_metadata']
        val=numeric(p['answer']) if meta['format_ok'] else None
        values.append(val)
        features.append([int(meta['format_ok']),int(val is not None),math.log1p(g['output_tokens']),
                         math.log1p(meta['message_tokens']),int(bool(str(p['uncertainty']).strip()))])
    equal=np.zeros((4,4),dtype=int);different=np.zeros((4,4),dtype=int)
    for i,j in itertools.product(range(4),repeat=2):
        if values[i] is not None and values[j] is not None:
            equal[i,j]=int(values[i]==values[j]);different[i,j]=int(values[i]!=values[j])
    return np.array(features,float),equal,different

def features_for_edge(mask,edge,node_features,equal,different):
    mask=np.asarray(mask,dtype=int)
    if mask.shape!=(12,) or not np.isin(mask,[0,1]).all() or mask[edge]!=0:raise ValueError('Expected edge-off context')
    source,target=EDGES[edge]
    incoming=[s for bit,(s,t) in zip(mask,EDGES) if bit and t==target]
    outdegree=sum(bit for bit,(s,t) in zip(mask,EDGES) if s==source)
    graph=np.r_[mask,np.eye(12,dtype=int)[edge],len(incoming),outdegree].astype(float)
    other=lambda i:sum(equal[i,j] for j in range(4) if j!=i)/3
    valid_incoming=[s for s in incoming if node_features[s,1] and node_features[target,1]]
    incoming_disagreement=(sum(different[s,target] for s in valid_incoming)/len(valid_incoming)) if valid_incoming else 0.
    runtime=np.r_[node_features.ravel(),[equal[i,j] for i,j in itertools.combinations(range(4),2)],
        node_features[source],node_features[target],different[source,target],other(source),other(target),incoming_disagreement]
    heuristics=[float(different[source,target]),incoming_disagreement,float(node_features[source,4]),1/(1+len(incoming)),0.]
    return graph,runtime,np.asarray(heuristics)

def topk_weights(scores,k=6):
    """Exact expected inclusion when a cutoff tie is uniformly randomly resolved."""
    s=np.asarray(scores,float)
    if s.ndim!=1 or not np.isfinite(s).all() or not 0<k<=len(s):raise ValueError('Invalid score/rank shape')
    cutoff=np.partition(s,len(s)-k)[len(s)-k]
    above=s>cutoff;tied=s==cutoff;need=k-int(above.sum())
    weights=above.astype(float)+tied*(need/int(tied.sum()))
    assert abs(weights.sum()-k)<1e-12
    return weights

def question_yields(scores,target):
    s=np.asarray(scores);y=np.asarray(target)
    if s.shape!=y.shape or s.ndim!=4 or s.shape[1:]!=(2,12,5):raise ValueError('Question/anchor/edge/seed shape')
    s=s.transpose(0,3,1,2).reshape(len(s),5,24);y=y.transpose(0,3,1,2).reshape(len(y),5,24)
    return np.array([np.mean([np.dot(topk_weights(a),b)/6 for a,b in zip(sq,yq)]) for sq,yq in zip(s,y)])
