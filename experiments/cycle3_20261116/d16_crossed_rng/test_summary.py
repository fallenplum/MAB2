"""Artificial gate fixtures; never evidence for scientific effectiveness."""
from pathlib import Path
import importlib.util,json
import numpy as np
def load(path):
 s=importlib.util.spec_from_file_location('crossed_score',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def checks(module):
 shape=(96,2,4,8);tokens=np.zeros(shape)
 for mask in range(8):tokens[:,:,:,mask]=100+10*mask.bit_count()
 shock=np.arange(96)%3;zero=np.zeros(shape);z=module.summarize(zero,zero,tokens,shock);assert not z['necessary_screen_passed'];assert all(v is None for d in z['correlations'].values() for v in d.values());assert z['descriptive_noise_components']['current']['fractions']['source'] is None
 stable=np.zeros(shape)
 for i in range(96):stable[i,:,:,1 if i%2==0 else 2]=1
 a=module.summarize(stable,stable,tokens,shock);assert a['necessary_screen_passed'] and a['mean_transfer_gain']>.49
 reversed_terminal=stable.copy()
 for i in range(96):reversed_terminal[i,:,2:,:]=0;reversed_terminal[i,:,2:,2 if i%2==0 else 1]=1
 b=module.summarize(reversed_terminal,reversed_terminal,tokens,shock);assert not b['headroom_gate'] and b['mean_transfer_gain']<0
 reversed_source=stable.copy()
 for i in range(96):reversed_source[i,1,:,:]=0;reversed_source[i,1,:,2 if i%2==0 else 1]=1
 c=module.summarize(reversed_source,reversed_source,tokens,shock);assert not c['credit_gate']
 return dict(status='PASS_ARTIFICIAL_CROSSED_GATE_FIXTURES',all_zero_rejected=True,stable_complementary_case_passed=True,terminal_half_reversal_rejected=True,source_reversal_rejected=True,artificial_only=True,new_model_calls=0)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();result=checks(load(a.source))
 with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
