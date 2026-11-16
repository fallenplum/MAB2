"""JSON-only RPC for isolated APPS call-based evaluation; gold stays in parent."""
import json,re,hashlib
from pathlib import Path
from code_sandbox import run_code

PRELUDE='''import sys,time,itertools,collections,math,fractions,random,heapq
from itertools import accumulate,product,permutations,combinations
from collections import Counter,OrderedDict,deque,defaultdict,ChainMap
from functools import lru_cache
from math import sqrt,sin,cos,tan,ceil,fabs,floor,gcd,exp,log,log2
from typing import List,Tuple
from heapq import *
'''

def prepare_inputs(inputs):
 # Exact author conversion, including its singleton-list behavior.
 try:
  if isinstance(inputs[0],dict):inputs=[{int(k):v for k,v in inputs[0].items()}]
 except (ValueError,TypeError,IndexError):pass
 return inputs
def prepare_expected(expected):
 try:
  if isinstance(expected,dict):expected=[{int(k):v for k,v in expected.items()}]
 except (ValueError,TypeError):pass
 try:
  if isinstance(expected[0],dict):expected=[{int(k):v for k,v in expected[0].items()}]
 except (ValueError,TypeError,IndexError,KeyError):pass
 return expected
def author_compare(output,expected):
 expected=prepare_expected(expected)
 if isinstance(output,tuple):output=list(output)
 result=output==expected
 if isinstance(expected,list) and expected:result=result or output==expected[0]
 try:
  if isinstance(output[0],tuple):result=result or [list(x) for x in output]==expected[0]
 except (ValueError,TypeError,IndexError,KeyError):pass
 return bool(result)
def pack(v):
 if v is None or type(v) in (bool,int,float,str):return ['atom',v]
 if type(v) in (list,tuple):return ['tuple' if type(v) is tuple else 'list',[pack(x) for x in v]]
 if type(v) is dict:return ['dict',[[pack(k),pack(x)] for k,x in v.items()]]
 raise TypeError('unsupported return type')
def unpack(v):
 assert isinstance(v,list) and len(v)==2;t,x=v
 if t=='atom':assert x is None or type(x) in (bool,int,float,str);return x
 if t=='list':return [unpack(z) for z in x]
 if t=='tuple':return tuple(unpack(z) for z in x)
 if t=='dict':return {unpack(k):unpack(z) for k,z in x}
 raise ValueError('unsupported typed RPC')

def wrapper(source,name):
 assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',name)
 # Only generated source and function name enter this wrapper, never expected values.
 import inspect
 helper=inspect.getsource(pack)
 return 'import json,sys,io,contextlib\n'+helper+f'''\n_program={source!r}
_name={name!r}
_args=json.loads(sys.stdin.read())
try:
 if isinstance(_args[0],dict):_args=[{{int(k):v for k,v in _args[0].items()}}]
except (ValueError,TypeError,IndexError):pass
_ns={{'__name__':'candidate'}}
with contextlib.redirect_stdout(io.StringIO()):
 exec({PRELUDE!r}+_program,_ns)
 _target=_ns['Solution']() if 'class Solution' in _program else None
 _func=getattr(_target,_name) if _target is not None else _ns[_name]
 _answer=_func(*_args)
print('__MAB2_RETURN__'+json.dumps(pack(_answer),ensure_ascii=False,allow_nan=False))
'''
def run_function(source,name,inputs,expected,root):
 r=run_code(wrapper(source,name),json.dumps(inputs,ensure_ascii=False,allow_nan=False),root)
 r.update(passed=False,returned=False)
 if r['status']=='OK':
  try:
   lines=r['stdout'].splitlines();assert len(lines)==1 and lines[0].startswith('__MAB2_RETURN__');value=unpack(json.loads(lines[0][15:]));r['passed']=author_compare(value,expected);r['returned']=True;r['typed_return']=pack(value)
  except (ValueError,TypeError,AssertionError,IndexError,KeyError):r['status']='INVALID_RETURN'
 return r

def qualify(root):
 tests=[('module','def f(a,b):return a+b',[2,3],5,True),('class','class Solution:\n def f(self,a):return a[::-1]',[[1,2,3]],[3,2,1],True),('string','def f(s):return s.upper()',['abc'],'ABC',True),('bool','def f(v):return not v',[False],True,True),('tuple','def f(a):return tuple(a)',[[1,2]],[1,2],True),('nested_tuple','def f(a):return [(1,2),(3,4)]',[0],[[[1,2],[3,4]]],True),('numeric_dict_keys','def f(d):return d',[{'1':'x'}],{'1':'x'},True),('singleton_expected','def f(a):return a+1',[2],[3],True),('false_positive_control','def f(a):return a+9',[2],3,False),('exception','def f(a):raise ValueError()',[2],3,False),('stdout_noise','def f(a):\n print("debug")\n return a',[4],4,True),('stdlib','def f(a):return Counter(a)[1]',[[1,1,2]],2,True)]
 rows=[]
 for name,source,args,expected,passed in tests:
  z=run_function(source,'f',args,expected,root);z.update(test=name,expected_pass=passed,check=z['passed']==passed and (not passed or z['returned']));rows.append(z)
 # Independent literal reference to the author function comparison, no generated code.
 comparison_checks=0
 values=[None,True,False,0,1,2.5,'x',[],[1],[1,2],[[1,2]],(1,2),[(1,2)],{1:'x'}]
 for output in values:
  for gold in values:
   exp=prepare_expected(gold);o=list(output) if isinstance(output,tuple) else output;ref=o==exp
   if isinstance(exp,list) and exp:ref=ref or o==exp[0]
   try:
    if isinstance(o[0],tuple):ref=ref or [list(x) for x in o]==exp[0]
   except Exception:pass
   assert author_compare(output,gold)==bool(ref);comparison_checks+=1
 return dict(status='PASS_FUNCTION_ADAPTER_ARTIFICIAL' if all(x['check'] for x in rows) else 'FAIL_FUNCTION_ADAPTER',tests=rows,comparison_checks=comparison_checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),official_evaluator_blob='25414d2a5371729f00e9be888c66b236d17b86c2',gold_in_execution_environment=False,limitations=['Author return-comparison permissiveness retained; not proof of full evaluator runtime parity.','Standard library prelude supplied; author numpy prelude omitted.','Fresh namespace/process for each case; author keeps module alive across cases.','Bounded Python3.8 sandbox may reject programs accepted with other dependencies or runtime limits.'],model_calls=0,GPU_hours=0)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--output',required=True);a=a.parse_args();result=qualify(a.root)
 with open(a.output,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='tests'}));print(json.dumps([dict(test=x['test'],check=x['check'],status=x['status']) for x in result['tests']]))
