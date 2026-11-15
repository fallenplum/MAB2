"""Evaluator-owned hidden counterfactual replay with an audited paid-feedback API."""
import numpy as np
import closure_policy as policy

GRAPHS=('G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV')
STATIC=('KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT','SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q')
FULL_TOKEN_CAP=3_000_000
COST_FIELDS=('tokens','kernel_seconds','calls','qwen_tokens','qwen_kernel_seconds','qwen_calls','mistral_tokens','mistral_kernel_seconds','mistral_calls')

def request_cost(record):
    g=record['generation'];model=record['model'];assert model in ['qwen','mistral']
    out=dict.fromkeys(COST_FIELDS,0);out.update(tokens=g['input_tokens']+g['output_tokens'],kernel_seconds=g['gpu_seconds'],calls=1)
    for k in ['tokens','kernel_seconds','calls']:out[model+'_'+k]=out[k]
    return out
def total(rows):
    return {k:sum(row[k] for row in rows) for k in COST_FIELDS}
def condition_cost(condition):return total([request_cost(r) for r in condition['nodes'].values()])

class PaidLedger:
    """A distinct instance for each method/episode/seed/change condition.

    Model request fingerprints are deduplicated only AFTER a requested bundle
    is executed. Cross-question fingerprints differ; service calls never reuse
    the diagnostic cache for cost discounts.
    """
    def __init__(self):self.paid={};self.events=[];self.score_observations=set();self.gold_questions=set()
    def purchase(self,condition,question_index,stage):
        charged=[]
        for node,r in condition['nodes'].items():
            key=r['request_fingerprint'];cost=request_cost(r)
            if key in self.paid:
                if self.paid[key]!=cost:raise ValueError('Same request has inconsistent recorded costs')
            else:self.paid[key]=cost;charged.append(cost)
        self.gold_questions.add(question_index);self.score_observations.add((question_index,condition['key']))
        v=total(charged);self.events.append({'stage':stage,'question_index':question_index,'condition':condition['condition'],'new_cost':v})
        return v
    def cost(self):return total(list(self.paid.values()))

def make_quotes(initial_records,changed):
    old=np.array([np.mean([condition_cost(r[g])['tokens'] for r in initial_records]) for g in GRAPHS[:3]])
    current=old.copy()
    if changed is not None:
        dead=np.mean([request_cost(r['G0']['nodes']['UV'[changed]])['tokens'] for r in initial_records])
        current[1+changed]-=dead
    final=np.array([np.mean([request_cost(r[g]['nodes']['M'])['tokens'] for r in initial_records]) for g in GRAPHS[1:3]])
    assert np.all(current>0) and np.all(final>0)
    return current,old,final

def choose_episode(initial_records,feedback_records,initial_em,feedback_em,changed,seed):
    """No service input accepted. Complete all actions before any S evaluation."""
    assert len(initial_records)==len(feedback_records)==8
    initial_gains=np.asarray([[r['G0_MINUS_EU']-r['G0'],r['G0_MINUS_EV']-r['G0']] for r in initial_em],float)
    prices,old,final=make_quotes(initial_records,changed)
    prefix='G0' if changed is None else 'G1';menu=[prefix,prefix+'_MINUS_EU',prefix+'_MINUS_EV'];results={}
    for method in policy.METHODS:
        ledger=PaidLedger()
        for i,records in enumerate(initial_records):
            for g in GRAPHS[:3]:ledger.purchase(records[g],i,'I')
        initial_cost=ledger.cost();initial_scores=len(ledger.score_observations);initial_gold=len(ledger.gold_questions);purchased=[]
        def buy(e,q):
            assert (e,q) not in purchased;purchased.append((e,q))
            before=ledger.cost()
            ledger.purchase(feedback_records[q][prefix],8+q,'C')
            ledger.purchase(feedback_records[q][menu[1+e]],8+q,'C')
            after=ledger.cost()
            return {'gain':int(feedback_em[q][menu[1+e]]-feedback_em[q][prefix]),'charged_tokens':after['tokens']-before['tokens']}
        decision=policy.run(initial_gains,prices,old,final,changed,method,seed,buy)
        assert decision['pairs']==len(purchased)<=8
        full=ledger.cost();feedback_cost={k:full[k]-initial_cost[k] for k in full}
        results[method]={'selected_condition':menu[decision['action']],'decision':decision,'initial_cost':initial_cost,'feedback_cost':feedback_cost,
            'initial_score_observations':initial_scores,'feedback_score_observations':len(ledger.score_observations)-initial_scores,
            'initial_gold_questions':initial_gold,'feedback_gold_questions':len(ledger.gold_questions)-initial_gold,'ledger':ledger.events,
            'price_forecasts_from_I_only':{'current':prices.tolist(),'old':old.tolist(),'counterfactual_final':final.tolist()}}
    return results

def evaluate_choices(choices,service_records,service_em,service_f1,changed):
    """Called only after every acquisition action has been fixed for this episode."""
    assert len(service_records)==len(service_em)==len(service_f1)==64
    prefix='G0' if changed is None else 'G1';e=0 if changed is None else changed
    statics={'KEEP_NO_INIT':prefix,'DELETE_CHANGED_NO_INIT':prefix+'_MINUS_E'+'UV'[e],
        'DELETE_OTHER_NO_INIT':prefix+'_MINUS_E'+'UV'[1-e],'SINGLE_Q':'SINGLE_Q','VOTE_Q3':'VOTE_Q3','INDEPENDENT3_Q':'INDEPENDENT3_Q'}
    zero=dict.fromkeys(COST_FIELDS,0);output={}
    old_action=choices['stale']['selected_condition']
    for method in list(policy.METHODS)+list(STATIC):
        selected=choices[method]['selected_condition'] if method in choices else statics[method]
        initial=choices[method]['initial_cost'] if method in choices else zero
        feedback=choices[method]['feedback_cost'] if method in choices else zero
        service=total([condition_cost(r[selected]) for r in service_records]);cost={k:initial[k]+feedback[k]+service[k] for k in zero}
        if cost['tokens']>FULL_TOKEN_CAP:raise ValueError('Frozen complete token cap exceeded; no silent truncation or quality imputation')
        answers=np.array([r[selected] for r in service_em]);base=np.array([r[prefix] for r in service_em]);stale=np.array([r[old_action] for r in service_em])
        row={'selected_condition':selected,'correct':int(answers.sum()),'service_questions':64,'accuracy':float(answers.mean()),
            'F1':float(np.mean([r[selected] for r in service_f1])),'initial_cost':initial,'feedback_cost':feedback,'service_cost':service,'complete_cost':cost,
            'complete_token_cap':FULL_TOKEN_CAP,'unused_token_budget':FULL_TOKEN_CAP-cost['tokens'],
            'utility':float(answers.mean()-policy.TOKEN_WEIGHT*cost['tokens']/64),
            'corrections_vs_keep':int(((answers==1)&(base==0)).sum()),'breakages_vs_keep':int(((answers==0)&(base==1)).sum()),
            'corrections_vs_stale':int(((answers==1)&(stale==0)).sum()),'breakages_vs_stale':int(((answers==0)&(stale==1)).sum()),
            'restoration_to_keep':selected==prefix and old_action!=prefix,
            'pairs':choices[method]['decision']['pairs'] if method in choices else 0,
            'initial_score_observations':choices[method]['initial_score_observations'] if method in choices else 0,
            'feedback_score_observations':choices[method]['feedback_score_observations'] if method in choices else 0,
            'initial_gold_questions':choices[method]['initial_gold_questions'] if method in choices else 0,
            'feedback_gold_questions':choices[method]['feedback_gold_questions'] if method in choices else 0}
        output[method]=row
    return output

def summarize(all_rows):
    assert len(all_rows)==8*5*2*(len(policy.METHODS)+len(STATIC))
    grouped={};comparisons={}
    for condition in ['change','no_change']:
        grouped[condition]={}
        for method in list(policy.METHODS)+list(STATIC):
            rs=[r for r in all_rows if r['condition']==condition and r['method']==method];assert len(rs)==40
            grouped[condition][method]={k:float(np.mean([r[k] for r in rs])) for k in ['accuracy','utility','F1','pairs','unused_token_budget']}
            grouped[condition][method].update(total_tokens=sum(r['complete_cost']['tokens'] for r in rs),total_kernel_seconds=sum(r['complete_cost']['kernel_seconds'] for r in rs),
                complete_cost_totals={k:sum(r['complete_cost'][k] for r in rs) for k in COST_FIELDS},
                stage_cost_totals={stage:{k:sum(r[stage+'_cost'][k] for r in rs) for k in COST_FIELDS} for stage in ['initial','feedback','service']},
                total_calls=sum(r['complete_cost']['calls'] for r in rs),corrections_vs_keep=sum(r['corrections_vs_keep'] for r in rs),breakages_vs_keep=sum(r['breakages_vs_keep'] for r in rs),
                mean_initial_tokens=float(np.mean([r['initial_cost']['tokens'] for r in rs])),mean_feedback_tokens=float(np.mean([r['feedback_cost']['tokens'] for r in rs])),mean_service_tokens=float(np.mean([r['service_cost']['tokens'] for r in rs])),
                by_generation_seed={str(s):{k:float(np.mean([r[k] for r in rs if r['seed_index']==s])) for k in ['accuracy','utility','pairs']} for s in range(5)},
                by_episode={str(e):{k:float(np.mean([r[k] for r in rs if r['episode']==e])) for k in ['accuracy','utility','pairs']} for e in range(8)},
                utility_weight_sensitivity={str(w):float(np.mean([r['accuracy']-w/1000*r['complete_cost']['tokens']/64 for r in rs])) for w in [0,.005,.02,.05]})
        a=grouped[condition]['closure_EVSI'];comparisons[condition]={}
        for m,b in grouped[condition].items():
            comparisons[condition][m]={'accuracy_difference':a['accuracy']-b['accuracy'],'utility_difference':a['utility']-b['utility'],
                'token_ratio':a['total_tokens']/b['total_tokens'],'utility_difference_by_seed':[a['by_generation_seed'][str(s)]['utility']-b['by_generation_seed'][str(s)]['utility'] for s in range(5)]}
    c=comparisons['change'];strong=['random','periodic','uncertainty','stale','protect_changed','EVSI_fixed_price','KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT']
    gates={'utility_over_all_maintenance_and_free_controls':all(c[m]['utility_difference']>=.005 for m in strong),
        'utility_vs_random_positive_four_of_five_seeds':sum(v>0 for v in c['random']['utility_difference_by_seed'])>=4,
        'single_Q_quality_and_utility':c['SINGLE_Q']['accuracy_difference']>=0 and c['SINGLE_Q']['utility_difference']>=0,
        'vote_Q3_utility':c['VOTE_Q3']['utility_difference']>=0,
        'independent3_Q_quality':c['INDEPENDENT3_Q']['accuracy_difference']>=-.01,
        'independent3_Q_full_token_cost':c['INDEPENDENT3_Q']['token_ratio']<=.9}
    return {'status':'COMPLETE_FIXED_ICS_DEVELOPMENT_REPLAY','fixed_nominee':'closure_EVSI','results':grouped,'comparisons':comparisons,'gates':gates,
        'screen_action':'RETAIN_SAME_METHOD_FOR_PREREGISTERED_INDEPENDENT_CONFIRMATION' if all(gates.values()) else 'STOP_FORMAL5_CANDIDATE',
        'utility_token_weight':policy.TOKEN_WEIGHT,'utility_weight_is_declared_tradeoff_not_money':True,'full_token_cap':FULL_TOKEN_CAP,
        'cap_may_be_nonbinding_actual_costs_and_slack_reported':True,'cache_replay_not_live_policy_deployment':True,'label_feedback_is_benchmark_oracle':True,
        'scoring_labels_financial_cost_unknown':True,'five_seeds_not_independent_datasets':True,'development_filter_not_significance':True,
        'confidence_intervals_computed':False,'bootstrap_runs':0,'automatic_method_or_paper_success':False,'other_methods_not_substitute_winners':True}
