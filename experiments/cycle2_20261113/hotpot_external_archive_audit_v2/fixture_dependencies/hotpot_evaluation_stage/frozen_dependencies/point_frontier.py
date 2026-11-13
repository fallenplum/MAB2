"""Prospective descriptive complete-service Pareto screen; no data IO or admission."""
import math

N=384
METHODS=('B','NO_EDGE','INDEPENDENT12','SINGLE_M')
MODELS=('qwen','mistral')
FIELDS=('calls','input_tokens','output_tokens','CUDA_event_seconds')

def review_frontier(em_success_counts,full_logical_costs,*,archive_integrity_passed,
                    unique_questions,complete_conditions,input_rejections,runtime_failures):
    """Caller must bind these inputs to an independent completed archive audit.

    Counts refer only to seed1 of four predetermined static methods. Costs are
    full service sums including reused nodes, separated by model. This helper
    cannot itself certify their source or replace the physical resident ledger.
    """
    if (archive_integrity_passed is not True or unique_questions!=N or complete_conditions!=4224
            or input_rejections!=0 or runtime_failures!=0):
        return {'status':'NOT_ASSESSED','reason':'incomplete_or_capacity_or_integrity_evidence',
                'population_dominance_claim':False,'GPU_admission_granted':False}
    if set(em_success_counts)!=set(METHODS) or set(full_logical_costs)!=set(METHODS):
        raise ValueError('Exactly four predetermined seed1 methods required')
    vectors={}
    for method in METHODS:
        count=em_success_counts[method]
        if type(count) is not int or not 0<=count<=N:raise ValueError('EM requires integer success count')
        costs=full_logical_costs[method]
        if set(costs)!=set(MODELS):raise ValueError('Per-model full costs required')
        values=[]
        for model in MODELS:
            if set(costs[model])!=set(FIELDS):raise ValueError('Cost fields differ')
            for key in FIELDS:
                value=costs[model][key]
                if value is None:
                    return {'status':'NOT_ASSESSED','reason':'unknown_full_logical_cost',
                            'population_dominance_claim':False,'GPU_admission_granted':False}
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Cost must be finite nonnegative numeric value')
                if key!='CUDA_event_seconds' and type(value) is not int:
                    raise ValueError('Calls and token costs must be integers')
                values.append(value)
            model_cost=costs[model]
            if model_cost['calls']==0 and any(model_cost[key]!=0 for key in FIELDS[1:]):
                raise ValueError('Zero calls require zero token and CUDA-event costs')
            if model_cost['calls']>0 and model_cost['input_tokens']<model_cost['calls']:
                raise ValueError('Every completed generation requires at least one input token')
        expected={'qwen':0,'mistral':384} if method=='SINGLE_M' else {'qwen':2304,'mistral':2688}
        if any(costs[m]['calls']!=expected[m] for m in MODELS):
            raise ValueError('Full logical calls differ from frozen 13/1-node service geometry')
        vectors[method]=values
    witnesses=[];comparisons={}
    for method in METHODS[1:]:
        no_less_quality=em_success_counts[method]>=em_success_counts['B']
        no_more_cost=all(a<=b for a,b in zip(vectors[method],vectors['B']))
        strictly_better=em_success_counts[method]>em_success_counts['B'] or any(a<b for a,b in zip(vectors[method],vectors['B']))
        dominates=no_less_quality and no_more_cost and strictly_better
        comparisons[method]={'success_count_difference_from_B':em_success_counts[method]-em_success_counts['B'],
            'cost_vector_difference_from_B':[a-b for a,b in zip(vectors[method],vectors['B'])],
            'weak_quality_and_cost':no_less_quality and no_more_cost,'strictly_better_component':strictly_better,
            'observed_point_dominates_B':dominates}
        if dominates:witnesses.append(method)
    return {'status':'DOMINATED' if witnesses else 'NOT_DOMINATED','scope':'observed_complete_seed1_service_point_frontier',
        'success_counts':dict(em_success_counts),'cost_vectors':vectors,
        'cost_vector_order':[m+'/'+f for m in MODELS for f in FIELDS],
        'comparisons':comparisons,'witnesses':witnesses,'sampling_uncertainty_assessed':False,
        'resident_cost_is_separate_physical_research_ledger':True,'population_dominance_claim':False,
        'GPU_admission_granted':False,'does_not_rescue_primary_gates':True}
