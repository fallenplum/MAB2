"""Independent arithmetic/request-ledger reconstruction; no frozen evaluator imports.

NumPy is used only to reproduce the declared PCG64 random baseline choices.
This is a second implementation by the research agent, not human peer review.
"""
import argparse,json,hashlib,math,statistics
from pathlib import Path
import numpy as np
METHODS=('closure_EVSI','EVSI_fixed_price','EVSI_no_cost_ranking','EVSI_old_graph_cost','random','periodic','uncertainty','graph_changed_first','stale','protect_changed','keep')
STATIC=('KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT','SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q')
FIELDS=('tokens','kernel_seconds','calls','qwen_tokens','qwen_kernel_seconds','qwen_calls','mistral_tokens','mistral_kernel_seconds','mistral_calls')
WEIGHT=1/200000
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cost(nodes):
    out=dict.fromkeys(FIELDS,0)
    for v in nodes:
        g=v['generation'];m=v['model'];assert m in ('qwen','mistral')
        for key,value in [('tokens',g['input_tokens']+g['output_tokens']),('kernel_seconds',g['gpu_seconds']),('calls',1)]:out[key]+=value;out[m+'_'+key]+=value
    return out
def add(*values):return {k:sum(v[k] for v in values) for k in FIELDS}
def score(a,prices,protect=None):
    values=[0]+[(row[2]-row[0])/sum(row)+WEIGHT*(prices[0]-prices[e+1]) for e,row in enumerate(a)]
    if protect is not None:values[protect+1]=-math.inf
    return values
def choose(a,prices,protect=None):return max(range(3),key=lambda e:(score(a,prices,protect)[e],-e))
def voi(a,prices,e):
    before=max(score(a,prices));expected=0
    for y in range(3):
        after=[list(x) for x in a];after[e][y]+=1
        expected+=a[e][y]/sum(a[e])*max(score(after,prices))
    assert expected-before>=-1e-10
    return max(0,expected-before)

def audit(private,summary,selection):
    checks=0
    def eq(a,b):
        nonlocal checks
        checks+=1
        if isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
        else:assert a==b,(a,b)
    rows=private['rows'];eq(len(rows),1360);eq(len(private['policy_traces']),80)
    indexed={(r['episode'],r['seed_index'],r['condition'],r['method']):r for r in rows};eq(len(indexed),len(rows))
    eq(private['ids'],[r['id'] for r in selection['questions']])
    reconstructed=[]
    for trace in private['policy_traces']:
        ep,s,condition=trace['episode'],trace['seed_index'],trace['condition'];start=ep*80
        changed='UV'.index(selection['episode_shocks'][ep]) if condition=='change' else None
        prefix='G1' if changed is not None else 'G0';menu=[prefix,prefix+'_MINUS_EU',prefix+'_MINUS_EV']
        records=private['records'];em=private['em'];f1=private['f1'];initial=list(range(start,start+8))
        old=[statistics.mean(cost(records[i][s][g]['nodes'].values())['tokens'] for i in initial) for g in ['G0','G0_MINUS_EU','G0_MINUS_EV']]
        prices=list(old)
        if changed is not None:prices[1+changed]-=statistics.mean(cost([records[i][s]['G0']['nodes']['UV'[changed]]])['tokens'] for i in initial)
        finals=[statistics.mean(cost([records[i][s][g]['nodes']['M']])['tokens'] for i in initial) for g in ['G0_MINUS_EU','G0_MINUS_EV']]
        alpha0=[[1/3+sum(em[i][s][g]-em[i][s]['G0']==v for i in initial) for v in [-1,0,1]] for g in ['G0_MINUS_EU','G0_MINUS_EV']]
        costs={};selected={}
        for method in METHODS:
            choice=trace['choices'][method];dec=choice['decision'];used=old if method=='EVSI_old_graph_cost' else prices
            eq(choice['price_forecasts_from_I_only'],{'current':prices,'old':old,'counterfactual_final':finals})
            paid={};score_keys=set();gold_keys=set()
            def buy_row(i,g):
                row=records[i][s][g];before=cost(paid.values())
                for v in row['nodes'].values():
                    key=v['request_fingerprint']
                    if key in paid:eq(paid[key],v)
                    else:paid[key]=v
                score_keys.add((i,g));gold_keys.add(i)
                after=cost(paid.values());return {k:after[k]-before[k] for k in FIELDS}
            for i in initial:
                for g in ['G0','G0_MINUS_EU','G0_MINUS_EV']:buy_row(i,g)
            initial_cost=cost(paid.values());eq(initial_cost,choice['initial_cost']);n_init_scores=len(score_keys);n_init_gold=len(gold_keys)
            alpha=[list(a) for a in alpha0];counts=[0,0];seen=set();seen_base=set();rng=np.random.default_rng(selection['generation_seeds'][s]+1000)
            for step,t in enumerate(dec['trace']):
                eq(t['step'],step);eq(t['before_action'],choose(alpha,used));available=[e for e in [0,1] if counts[e]<8];candidates={}
                for e in available:
                    q=counts[e];price=finals[e]+(prices[0] if method=='EVSI_fixed_price' or q not in seen_base else 0);gain=64*voi(alpha,used,e)
                    candidates[str(e)]={'q':q,'forecast_tokens':price,'expected_gain':gain,'ratio':gain/price}
                eq(candidates,t['candidates'])
                if method=='periodic':edge,q=[(e,q) for q in [0,2,4,6] for e in [0,1]][step]
                elif method=='random':edge=int(rng.choice(available));q=counts[edge]
                elif method=='graph_changed_first':
                    edge=min(available,key=lambda e:(counts[e],e)) if changed is None else changed if counts[changed]<8 else 1-changed;q=counts[edge]
                elif method=='uncertainty':
                    variance=lambda e:((alpha[e][0]+alpha[e][2])/sum(alpha[e])-((alpha[e][2]-alpha[e][0])/sum(alpha[e]))**2)/(sum(alpha[e])+1)
                    edge=max(available,key=lambda e:(variance(e),-e));q=counts[edge]
                else:
                    metric='expected_gain' if method=='EVSI_no_cost_ranking' else 'ratio';edge=max(available,key=lambda e:(candidates[str(e)][metric],-e));q=counts[edge]
                    assert candidates[str(edge)]['expected_gain']>WEIGHT*candidates[str(edge)]['forecast_tokens']
                eq([t['edge'],t['feedback_row']],[edge,q]);assert (edge,q) not in seen
                i=start+8+q;charged=add(buy_row(i,prefix),buy_row(i,menu[edge+1]));gain=em[i][s][menu[edge+1]]-em[i][s][prefix]
                eq(t['observed_gain'],gain);eq(t['charged_tokens'],charged['tokens']);alpha[edge][int(gain)+1]+=1;counts[edge]+=1;seen.add((edge,q));seen_base.add(q)
                eq(t['after_action'],choose(alpha,used))
            eq(dec['pairs'],len(seen));assert len(seen)<=8
            if method in ['random','periodic','uncertainty','graph_changed_first']:eq(len(seen),8)
            elif method.startswith('EVSI') or method=='closure_EVSI':
                if len(seen)<8:
                    options=[]
                    for edge in [0,1]:
                        if counts[edge]>=8:continue
                        price=finals[edge]+(prices[0] if method=='EVSI_fixed_price' or counts[edge] not in seen_base else 0);gain=64*voi(alpha,used,edge)
                        options.append((gain if method=='EVSI_no_cost_ranking' else gain/price,-edge,gain-WEIGHT*price))
                    assert not options or max(options)[2]<=1e-10
            expected=0 if method=='keep' else choose(alpha,used,changed if method=='protect_changed' else None)
            eq(dec['action'],expected);eq(choice['selected_condition'],menu[expected]);selected[method]=menu[expected]
            full=cost(paid.values());feedback={k:full[k]-initial_cost[k] for k in FIELDS};eq(choice['feedback_cost'],feedback)
            eq(choice['initial_score_observations'],n_init_scores);eq(choice['initial_gold_questions'],n_init_gold)
            eq(choice['feedback_score_observations'],len(score_keys)-n_init_scores);eq(choice['feedback_gold_questions'],len(gold_keys)-n_init_gold)
            costs[method]=(initial_cost,feedback)
        edge=0 if changed is None else changed
        selected.update(KEEP_NO_INIT=prefix,DELETE_CHANGED_NO_INIT=prefix+'_MINUS_E'+'UV'[edge],DELETE_OTHER_NO_INIT=prefix+'_MINUS_E'+'UV'[1-edge],SINGLE_Q='SINGLE_Q',VOTE_Q3='VOTE_Q3',INDEPENDENT3_Q='INDEPENDENT3_Q')
        for method in METHODS+STATIC:
            row=indexed[(ep,s,condition,method)];g=selected[method];eq(row['selected_condition'],g)
            zero=dict.fromkeys(FIELDS,0);ic,cc=costs.get(method,(zero,zero));service=add(*[cost(records[i][s][g]['nodes'].values()) for i in range(start+16,start+80)]);full=add(ic,cc,service)
            for key,value in [('initial_cost',ic),('feedback_cost',cc),('service_cost',service),('complete_cost',full)]:eq(row[key],value)
            eq(row['unused_token_budget'],3000000-full['tokens']);assert full['tokens']<=3000000
            correct=sum(em[i][s][g] for i in range(start+16,start+80));eq(row['correct'],correct);eq(row['accuracy'],correct/64)
            eq(row['utility'],correct/64-WEIGHT*full['tokens']/64);eq(row['F1'],statistics.mean(f1[i][s][g] for i in range(start+16,start+80)))
            for name,baseline in [('keep',prefix),('stale',selected['stale'])]:
                eq(row['corrections_vs_'+name],sum(em[i][s][g]==1 and em[i][s][baseline]==0 for i in range(start+16,start+80)))
                eq(row['breakages_vs_'+name],sum(em[i][s][g]==0 and em[i][s][baseline]==1 for i in range(start+16,start+80)))
            reconstructed.append(row)
    for condition in ['change','no_change']:
        for method in METHODS+STATIC:
            rs=[r for r in reconstructed if r['condition']==condition and r['method']==method];result=summary['results'][condition][method]
            for k in ['accuracy','utility','F1','pairs','unused_token_budget']:eq(result[k],statistics.mean(r[k] for r in rs))
            eq(result['total_tokens'],sum(r['complete_cost']['tokens'] for r in rs));eq(result['total_kernel_seconds'],sum(r['complete_cost']['kernel_seconds'] for r in rs))
            eq(result['complete_cost_totals'],{k:sum(r['complete_cost'][k] for r in rs) for k in FIELDS})
            eq(result['stage_cost_totals'],{stage:{k:sum(r[stage+'_cost'][k] for r in rs) for k in FIELDS} for stage in ['initial','feedback','service']})
            for s in range(5):
                for k in ['accuracy','utility','pairs']:eq(result['by_generation_seed'][str(s)][k],statistics.mean(r[k] for r in rs if r['seed_index']==s))
            for ep in range(8):
                for k in ['accuracy','utility','pairs']:eq(result['by_episode'][str(ep)][k],statistics.mean(r[k] for r in rs if r['episode']==ep))
            for w in [0,.005,.02,.05]:eq(result['utility_weight_sensitivity'][str(w)],statistics.mean(r['accuracy']-w/1000*r['complete_cost']['tokens']/64 for r in rs))
        a=summary['results'][condition]['closure_EVSI']
        for method in METHODS+STATIC:
            b=summary['results'][condition][method];comp=summary['comparisons'][condition][method]
            eq(comp,{'accuracy_difference':a['accuracy']-b['accuracy'],'utility_difference':a['utility']-b['utility'],'token_ratio':a['total_tokens']/b['total_tokens'],
                'utility_difference_by_seed':[a['by_generation_seed'][str(s)]['utility']-b['by_generation_seed'][str(s)]['utility'] for s in range(5)]})
    c=summary['comparisons']['change'];controls=['random','periodic','uncertainty','stale','protect_changed','EVSI_fixed_price','KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT']
    gates={'utility_over_all_maintenance_and_free_controls':all(c[m]['utility_difference']>=.005 for m in controls),'utility_vs_random_positive_four_of_five_seeds':sum(v>0 for v in c['random']['utility_difference_by_seed'])>=4,
        'single_Q_quality_and_utility':c['SINGLE_Q']['accuracy_difference']>=0 and c['SINGLE_Q']['utility_difference']>=0,'vote_Q3_utility':c['VOTE_Q3']['utility_difference']>=0,
        'independent3_Q_quality':c['INDEPENDENT3_Q']['accuracy_difference']>=-.01,'independent3_Q_full_token_cost':c['INDEPENDENT3_Q']['token_ratio']<=.9}
    eq(gates,summary['gates']);eq(summary['bootstrap_runs'],0);eq(summary['confidence_intervals_computed'],False)
    eq(summary['screen_action'],'RETAIN_SAME_METHOD_FOR_PREREGISTERED_INDEPENDENT_CONFIRMATION' if all(gates.values()) else 'STOP_FORMAL5_CANDIDATE')
    return {'status':'PASS_INDEPENDENT_COST_POLICY_AND_POINT_RECONSTRUCTION','checks':checks,'rows':len(reconstructed),'gates':gates,'human_independent_review':False,'numpy_used_only_for_PCG64_baseline':True,'new_model_calls':0,'bootstrap_runs':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--private-sha',required=True);p.add_argument('--summary',required=True);p.add_argument('--summary-sha',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert file_sha(a.private)==a.private_sha and file_sha(a.summary)==a.summary_sha and not a.output.exists()
    load=lambda p:json.loads(Path(p).read_bytes())
    result=audit(load(a.private),load(a.summary),load(a.selection));result['input_sha256']={'private':a.private_sha,'summary':a.summary_sha,'selection':file_sha(a.selection)};result['source_sha256']=file_sha(__file__)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
