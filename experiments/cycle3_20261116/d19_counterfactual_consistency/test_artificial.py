import importlib.util,json
from pathlib import Path
path=Path(__file__).with_name('policy.py')
if not path.exists():path=Path(__file__).with_name('_mab2_d19_policy.py')
s=importlib.util.spec_from_file_location('policy',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def trial(keys,answers,early):
 invoked=[]
 def meta(i):return dict(job_id=keys[i[0]])
 def query(i):invoked.append(i[0]);return dict(text=answers[i[0]],input_tokens=10,output_tokens=2)
 selected,records,answer=m.acquire_and_vote([(i,0) for i in range(3)],meta,query,lambda x:x,early)
 return selected,records,answer,invoked
a=trial(['a','b','c'],['yes','yes','no'],True);assert a[3]==[0,1] and a[2]=='yes'
b=trial(['a','a','c'],['yes','yes','no'],True);assert b[3]==[0,2] and len(b[1])==2 and b[2]=='yes'
c=trial(['a','b','c'],['','', 'yes'],True);assert c[3]==[0,1,2]
for answers in [['a','a','b'],['a','b','b'],['a','b','c'],['a','b','a'],['','','a']]:assert trial(['a','b','c'],answers,True)[2]==trial(['a','b','c'],answers,False)[2]
print(json.dumps(dict(status='PASS_ARTIFICIAL_CONSENSUS_CACHE_AND_STOP_TEST',duplicate_requests_not_extra_votes=True,unqueried_output_not_accessed=True,empty_answers_do_not_stop=True,early_full_equivalence=True)))
