"""Synthetic-only independent calculations; never loads R1 data or results."""
from pathlib import Path
from itertools import product
from fractions import Fraction as F
import hashlib,json
from r2_math import *
checks=[]
def check(name,value):
    assert value,name
    checks.append({"name":name,"status":"PASS"})
def mean(xs): return sum(xs)/len(xs)
def var_population(xs):
    mu=mean(xs); return mean([(x-mu)**2 for x in xs])
pairs=[(F(0),F(0)),(F(0),F(1)),(F(1),F(1)),(F(1),F(1))]
# Enumerate independent two-row old bank and two-row paired new panel.
transport=[]; cv=[]
beta=F(1,4)
for old in product([F(0),F(1)],repeat=2):
    for new in product(pairs,repeat=2):
        aa=[p[0] for p in new]; bb=[p[1] for p in new]
        transport.append(mean(old)+mean([b-a for a,b in new]))
        cv.append(mean(bb)-beta*(mean(aa)-mean(old)))
check("independent bank unbiased by enumeration",mean(transport)==F(3,4))
check("transport variance includes bank noise",var_population(transport)==F(7,32))
check("transport formula matches enumeration",abs(transport_variance(.25,.1875,2,2)-float(var_population(transport)))<1e-12)
check("CV formula independently matches enumeration",abs(control_variate_variance(.25,.1875,.125,2,2,.25)-float(var_population(cv)))<1e-12)
check("beta one is drift estimator",abs(control_variate_variance(.25,.1875,.125,2,2,1)-transport_variance(.25,.1875,2,2))<1e-12)
a=[F(-1),F(0),F(1),F(0)]; b=[F(0),F(1),F(1),F(-1)]
check("same panel collapses exactly",mean(a)+mean([y-x for x,y in zip(a,b)])==mean(b))
check("zero drift retains old-bank uncertainty",transport_variance(.25,0,12,3)==.25/12)
check("no-change pooling dominates beta one",pooled_no_change_variance(.25,12,3)<transport_variance(.25,0,12,3))
check("no-change optimal beta gives pooling",abs(control_variate_variance(.25,.25,.25,12,3,population_optimal_beta(.25,.25,12,3))-pooled_no_change_variance(.25,12,3))<1e-12)
check("equal-size high feedback can fail necessary condition",not independent_bank_necessary_condition(.25,.25,12,21))
check("zero over zero stays undefined",safe_risk_ratio(0,0)["ratio"] is None)
check("near-zero denominator remains inconclusive",safe_risk_ratio(1e-8,1e-8)["status"].startswith("INCONCLUSIVE"))
c=template_cost(6)
check("six-edge exact structural cost",c=={"direct_calls":21,"four_world_calls":37,"direct_token_upper":175616,"four_world_token_upper":309632})
check("primary common full token cap",12*c["direct_token_upper"]+3*c["four_world_token_upper"]+24*75008==4836480)
check("direct five fits while six exceed same new cap",5*c["direct_token_upper"] <= 3*c["four_world_token_upper"] < 6*c["direct_token_upper"])
check("whole new-pilot call count",(12*21+3*37+2*21+24*21)*8==7272)
check("unaffected worker does not imply unaffected credit",(1*1-0*1)==1 and (1*0-0*0)==0)
opt=from_scratch_coefficient(.25,.04,21,37)
# Continuous Lagrange allocation check at budget 10000.
budget=10000.0
ratio=sqrt(.25*37/(.04*21))
n=budget/(ratio*21+37); m=ratio*n
check("continuous allocation envelope independently matches",abs((.25/m+.04/n)*budget-opt)<1e-9)
report={"status":"PASS","checks":checks,"count":len(checks),"evidence_tier":"controlled synthetic algebra only","real_R1_IC_read":False,"R1_S_access":False,"gpu_jobs":0,"module_sha256":hashlib.sha256((Path(__file__).parent/"r2_math.py").read_bytes()).hexdigest()}
import os
report_dir=Path(os.environ.get("MAB2_SYNTHETIC_REPORT_DIR",str(Path(__file__).parent/"synthetic_test_tmp")))
report_dir.mkdir(parents=True,exist_ok=True)
with (report_dir/"SYNTHETIC_MATH_CHECKS.json").open("x",encoding="utf-8") as stream:
    stream.write(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"status":"PASS","checks":len(checks),"real_data_read":False}))

