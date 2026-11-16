"""Artificial leakage and sunk-source-cost checks, not scientific results."""
import importlib.util,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
path=Path(__file__).with_name('score.py')
if not path.exists():path=Path(__file__).with_name('_mab2_d18_score.py')
s=importlib.util.spec_from_file_location('d18',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
shape=(96,2,4,8);em=np.zeros(shape);prefix=np.broadcast_to(np.array([1000*g.bit_count() for g in range(8)]),(96,2,8)).copy();terminal=np.broadcast_to(np.array([100+20*g.bit_count() for g in range(8)]),(96,2,8)).copy();tokens=np.repeat((prefix+terminal+10)[:,:,None,:],4,axis=2);shock=np.arange(96)%3
f=dict(question=np.zeros((96,2,384)),message=np.zeros((96,2,1536)),numeric=np.zeros((96,2,15)),prefix_tokens=prefix,terminal_input_tokens=terminal)
with threadpool_limits(limits=2):
 a,private=m.evaluate_task(f,em,tokens,shock)
 altered=em.copy();altered[np.arange(96)%6==0,:,:,2]=1
 b,changed=m.evaluate_task(f,altered,tokens,shock)
for name in ['question','message_ridge','message_trees']:assert np.array_equal(private['predictions'][name][::6],changed['predictions'][name][::6])
for name in ['fixed','question_preselected']:assert np.array_equal(private['choices'][name][::6],changed['choices'][name][::6])
assert all(not x['primary_passed'] for x in a['screens'].values())
assert a['methods']['message_ridge']['service_tokens']>=3130
assert a['methods']['single']['service_tokens']==110
assert all(v['train_question_count']==80 and v['test_question_count']==16 for v in a['folds'])
print(json.dumps(dict(status='PASS_ARTIFICIAL_D18_GROUP_LEAKAGE_AND_COST_TEST',heldout_labels_do_not_change_heldout_predictions=True,all_observed_sources_charged=True,zero_quality_gain_rejected=True,new_model_calls=0)))
