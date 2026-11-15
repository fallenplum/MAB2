import unittest,numpy as np
import closure_replay_policies as r
import closure_policy as p

def records(q,shock=0):
    out={}
    for g in r.GRAPHS+('SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q'):
        nodes=list('UVRM') if g not in ['SINGLE_Q','VOTE_Q3'] else ['M'] if g=='SINGLE_Q' else list('MUV')
        if g=='G1_MINUS_E'+'UV'[shock]:nodes.remove('UV'[shock])
        rr={}
        for node in nodes:
            tag=g if node=='M' or not g.startswith('G') else g[:2] if node=='R' else 'common'
            rr[node]={'request_fingerprint':f'{q}/{node}/{tag}','model':'mistral' if node=='R' else 'qwen','generation':{'input_tokens':100,'output_tokens':10,'gpu_seconds':.1}}
        out[g]={'condition':g,'key':'1/'+g,'nodes':rr}
    return out
def quality(q):return {g:int((q+len(g))%3!=0) for g in r.GRAPHS+('SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q')}

class Tests(unittest.TestCase):
    def test_shared_feedback_bundles_charge_actual_unique_calls(self):
        ledger=r.PaidLedger();rs=records(0)
        for g in ['G1','G1_MINUS_EU','G1','G1_MINUS_EV']:ledger.purchase(rs[g],0,'C')
        self.assertEqual(ledger.cost()['calls'],6)
        self.assertEqual(ledger.cost()['tokens'],660)
        self.assertEqual(ledger.cost()['qwen_tokens'],550)
        self.assertEqual(ledger.cost()['mistral_tokens'],110)
        self.assertEqual(len(ledger.gold_questions),1)
    def test_initialization_and_service_are_fully_charged(self):
        rr=[records(i) for i in range(80)];ee=[quality(i) for i in range(80)]
        choices=r.choose_episode(rr[:8],rr[8:16],ee[:8],ee[8:16],0,71)
        self.assertEqual(choices['stale']['initial_cost']['calls'],48)
        self.assertEqual(choices['periodic']['feedback_cost']['calls'],24)
        vals=r.evaluate_choices(choices,rr[16:],ee[16:],ee[16:],0)
        self.assertEqual(vals['KEEP_NO_INIT']['complete_cost']['calls'],64*4)
        self.assertEqual(vals['DELETE_CHANGED_NO_INIT']['complete_cost']['calls'],64*3)
        self.assertEqual(vals['keep']['complete_cost']['calls'],48+64*4)
        self.assertEqual(vals['keep']['initial_gold_questions'],8)
        self.assertEqual(vals['KEEP_NO_INIT']['initial_gold_questions'],0)
    def test_graph_quote_difference_only_removes_disconnected_call(self):
        current,old,final=r.make_quotes([records(i) for i in range(8)],0)
        self.assertEqual(current.tolist(),[440,330,440]);self.assertEqual(old.tolist(),[440,440,440]);self.assertEqual(final.tolist(),[110,110])
    def test_all_decisions_fixed_before_service_appears(self):
        # choose_episode has no S argument; changing service scores cannot change its choices.
        rr=[records(i) for i in range(80)];ee=[quality(i) for i in range(80)]
        choices=r.choose_episode(rr[:8],rr[8:16],ee[:8],ee[8:16],0,71)
        before={m:v['selected_condition'] for m,v in choices.items()}
        for bit in [0,1]:r.evaluate_choices(choices,rr[16:],[dict.fromkeys(ee[0],bit) for _ in range(64)],[dict.fromkeys(ee[0],bit) for _ in range(64)],0)
        self.assertEqual(before,{m:v['selected_condition'] for m,v in choices.items()})
    def test_inconsistent_cache_cost_rejected(self):
        rs=records(0);ledger=r.PaidLedger();ledger.purchase(rs['G1'],0,'C');rs['G1']['nodes']['M']['generation']['output_tokens']+=1
        with self.assertRaises(ValueError):ledger.purchase(rs['G1'],0,'C')
    def test_complete_synthetic_grid_and_fixed_negative(self):
        rr=[records(i) for i in range(80)];ee=[quality(i) for i in range(80)];allrows=[]
        for ep in range(8):
            for s in range(5):
                for name,c in [('change',0),('no_change',None)]:
                    choices=r.choose_episode(rr[:8],rr[8:16],ee[:8],ee[8:16],c,71+s)
                    for method,row in r.evaluate_choices(choices,rr[16:],ee[16:],ee[16:],c).items():allrows.append({'episode':ep,'seed_index':s,'condition':name,'method':method,**row})
        summary=r.summarize(allrows);self.assertEqual(len(allrows),1360)
        self.assertEqual(summary['screen_action'],'STOP_FORMAL5_CANDIDATE');self.assertFalse(summary['automatic_method_or_paper_success'])
        self.assertEqual(summary['bootstrap_runs'],0)
if __name__=='__main__':unittest.main()
