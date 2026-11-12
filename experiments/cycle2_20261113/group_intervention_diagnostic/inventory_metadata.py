"""Read historical metadata only; never reads quality arrays, questions or gold."""
import argparse, collections, hashlib, itertools, json
from pathlib import Path
import numpy as np

PIN='7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'

def faces_for_masks(masks):
    observed={tuple(int(x) for x in row) for row in masks}
    if not observed: return [],{}
    d=len(next(iter(observed)))
    if any(len(m)!=d or any(x not in (0,1) for x in m) for m in observed):raise ValueError('Invalid masks')
    faces=[]; counts=collections.Counter()
    for i,j in itertools.combinations(range(d),2):
        backgrounds=set()
        for mask in observed:
            b=list(mask); b[i]=b[j]=0; backgrounds.add(tuple(b))
        for b in sorted(backgrounds):
            worlds=[]
            for x,y in ((0,0),(0,1),(1,0),(1,1)):
                w=list(b); w[i]=x;w[j]=y; worlds.append(tuple(w))
            n=sum(w in observed for w in worlds);counts[n]+=1
            if n==4:faces.append({'edges':[i,j],'background':b,'worlds':worlds})
    return faces,dict(sorted(counts.items()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise SystemExit('Refuse overwrite')
    raw=a.input.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    if sha!=PIN:raise ValueError('Historical accepted D2 NPZ hash mismatch')
    del raw
    with np.load(a.input,allow_pickle=False) as z:
        masks=z['masks'];rowq=z['row_question'];qids=z['question_ids'];parts=z['partition']
        if masks.shape!=(10394,12) or rowq.shape!=(10394,) or qids.shape!=(400,) or parts.shape!=(400,):raise ValueError('Unexpected historical metadata shape')
        detail=[];hist=collections.Counter(); pairq=collections.defaultdict(set)
        for q in range(400):
            faces,counts=faces_for_masks(masks[rowq==q]);hist.update(counts)
            for f in faces:pairq[tuple(f['edges'])].add(q)
            detail.append({'question_index':q,'partition':str(parts[q]),'complete_faces':len(faces),'face_edges':[f['edges'] for f in faces]})
    out={'schema':'mab2_group_metadata_inventory_v1','source_sha256':sha,'quality_array_accessed':False,'questions_gold_read':False,
         'questions':400,'unique_graph_rows':len(masks),'complete_faces':sum(x['complete_faces'] for x in detail),
         'questions_with_complete_faces':sum(x['complete_faces']>0 for x in detail),'observed_corner_count_histogram':dict(hist),
         'pair_unique_question_support':{','.join(map(str,k)):len(v) for k,v in sorted(pairq.items())},'private_question_index_support':detail}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!='private_question_index_support'}))

if __name__=='__main__':main()
