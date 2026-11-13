"""Exact arithmetic on fabricated probabilities and costs only; no data inputs."""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    n=(F(80,100),F(60,100)); b=(F(65,100),F(70,100)); c=(F(80,100),F(75,100))
    mean=lambda v:sum(v,F(0))/len(v)
    conditional=mean(tuple(max(x,y) for x,y in zip(n,b)))
    assert mean(b)<mean(n)<conditional<mean(c)
    # A real interaction can exist even when its marginal mean vanishes.
    y={(0,0):F(7,10),(1,0):F(8,10),(0,1):F(8,10),(1,1):F(7,10)}
    credit=(y[1,0]-y[0,0],y[1,1]-y[0,1])
    assert credit==(F(1,10),F(-1,10)) and mean(credit)==0
    k=F(1000); saving=F(2)
    assert k-100*saving==800 and k-500*saving==0 and k-501*saving<0
    assert sum((1,1,13,13,13))==41
    assert sum((1,1,13,13,13,13,13,13))==80
    assert 80+4*13==132
    assert all(k-h*s>=k for h in (0,100,1000) for s in (F(0),F(-1)))
    assert k-100*saving>k-500*saving>k-501*saving
    out={'status':'PASS_EXACT_FABRICATED_LOGIC_ONLY','checks':8,
         'mean_no_edge':str(mean(n)),'mean_static_B':str(mean(b)),
         'conditional_oracle_fixed_two_action_class':str(conditional),'mean_central':str(mean(c)),
         'conditional_credit':[str(x) for x in credit],
         'break_even_equal_not_strict':500,'first_integer_strict_saving':501,
         'call_envelopes':{'41_per_question':[41*64,41*128],
                          '132_per_question':[132*64,132*128],
                          'four_config_13call':[4*13*64,4*13*128]},
         'empirical_data_accessed':False,'new_model_calls':0,'GPU_hours':0,
         'future_policy_or_all_strategy_upper_bound_claimed':False,
         'budget_or_scientific_admission':False}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(out,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'status':out['status'],'checks':8}))


if __name__=='__main__':main()
