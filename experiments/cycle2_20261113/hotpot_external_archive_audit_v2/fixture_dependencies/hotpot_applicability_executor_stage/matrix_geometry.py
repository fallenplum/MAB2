"""Independent three-round reachability calculation; no production imports."""
import json
from pathlib import Path

EDGES=tuple((source,target) for source in range(4) for target in range(4) if source!=target)
B_EDGES=frozenset((1,5,7,9))
A=(1,)*12
B=tuple(int(i in B_EDGES) for i in range(12))
NO_EDGE=(0,)*12
ROUTE=('qwen','mistral','qwen','mistral')
NODES=tuple(f'r{r}:w{w}' for r in range(1,4) for w in range(4))+('aggregate',)


def changed_closure(old,new):
    changed_targets={target for index,(_,target) in enumerate(EDGES) if old[index]!=new[index]}
    affected={f'r{r}:w{w}' for r in (2,3) for w in changed_targets}
    adjacency={node:set() for node in NODES}
    for r in (1,2):
        for w in range(4):adjacency[f'r{r}:w{w}'].add(f'r{r+1}:w{w}')
        for i,(s,t) in enumerate(EDGES):
            if old[i] or new[i]:adjacency[f'r{r}:w{s}'].add(f'r{r+1}:w{t}')
    for w in range(4):adjacency[f'r3:w{w}'].add('aggregate')
    frontier=list(affected)
    while frontier:
        node=frontier.pop()
        for descendant in adjacency[node]:
            if descendant not in affected:
                affected.add(descendant);frontier.append(descendant)
    return sorted(affected)


def without(mask,edge):
    if type(edge) is not int or edge not in B_EDGES:
        raise ValueError('probe_edge must be a zero-based source-major index in {1,5,7,9}')
    return tuple(0 if i==edge else bit for i,bit in enumerate(mask))


def geometry(edge):
    am,bm=without(A,edge),without(B,edge)
    closures={'A':list(NODES),'B':changed_closure(A,B),'NO_EDGE':changed_closure(A,NO_EDGE),
              'A_MINUS_E':changed_closure(A,am),'B_MINUS_E':changed_closure(B,bm),
              'INDEPENDENT12':[f'r{s}:w{r}' for r in range(4) for s in range(1,4)]+['aggregate'],
              'SINGLE_M':['aggregate']}
    seed1=['A','B','NO_EDGE','A_MINUS_E','B_MINUS_E','INDEPENDENT12','SINGLE_M']
    seed2=['A','B','A_MINUS_E','B_MINUS_E']
    by_model={name:{m:sum(('mistral' if node=='aggregate' else ROUTE[int(node[-1])])==m for node in nodes)
                    for m in ('qwen','mistral')} for name,nodes in closures.items()}
    return {'edge_indexing':'zero_based_source_major','probe_edge':edge,'probe_endpoints':EDGES[edge],
            'B_edges':sorted(B_EDGES),'B_endpoints':[EDGES[i] for i in sorted(B_EDGES)],
            'closures':closures,'physical_calls_by_condition':{k:len(v) for k,v in closures.items()},
            'seed1_calls':sum(len(closures[k]) for k in seed1),'seed2_calls':sum(len(closures[k]) for k in seed2),
            'total_success_physical_calls':sum(len(closures[k]) for k in seed1+seed2),
            'total_success_logical_calls':sum(1 if k=='SINGLE_M' else 13 for k in seed1+seed2),
            'total_success_model_calls':{m:sum(by_model[k][m] for k in seed1+seed2) for m in ('qwen','mistral')},
            'successful_cache_schedule_only':True,'missing_base_policy':'FRESH; actual request count can exceed 87'}


if __name__=='__main__':
    report={'status':'PASS_INDEPENDENT_GEOMETRY','per_probe_edge':[geometry(i) for i in sorted(B_EDGES)],
            'new_GPU_calls':0,'benchmark_reads':0}
    assert all(r['seed1_calls']==55 and r['seed2_calls']==32 and r['total_success_physical_calls']==87
               and r['total_success_logical_calls']==131 for r in report['per_probe_edge'])
    (Path(__file__).resolve().parent/'GEOMETRY.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'edge_cases':len(report['per_probe_edge']),
                      'seed1_calls':55,'seed2_calls':32,'physical_calls':87,'logical_calls':131}))
