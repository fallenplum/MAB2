import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import orchestration_core as core
import storage
import worker
import parent
from plan_tools import geometry_request_keys
from production_guard import kill_owned_group

BASE=Path(__file__).resolve().parent
(BASE/'cpu_tmp').mkdir(exist_ok=True)
tempfile.tempdir=str(BASE/'cpu_tmp')
GEOMETRY=json.loads(Path(os.environ.get('MAB2_GEOMETRY',str(BASE.parent/'hotpot_applicability_executor_stage/GEOMETRY.json'))).read_bytes())
KEYS=geometry_request_keys(GEOMETRY)
PIN='a'*64


def payload(row):
    return {'id':'hotpotqa_'+row['id'],'question':row['question'],
            'context':json.dumps([{'title':t,'sentences':s} for t,s in zip(row['context']['title'],row['context']['sentences'])],ensure_ascii=False,separators=(',',':'))}


def fixture(workers=2):
    rows=[{'id':f'synthetic_{i:03d}','question':f'Artificial input {i}',
           'context':{'title':['Synthetic document'],'sentences':[['No benchmark data.']]},'type':'bridge','level':'easy'} for i in range(384)]
    manifest=[{'id':r['id'],'edge_index':core.EDGES[i%4],'input_sha256':core.digest(r),
               'generation_seed1':core.SEEDS[0],'generation_seed2':core.SEEDS[1]} for i,r in enumerate(rows)]
    lengths=[{'id':r['id'],'payload_sha256':core.graph_digest(payload(r)),'qwen_tokens':100+i,'mistral_tokens':120+i} for i,r in enumerate(rows)]
    schedule=core.build_schedule(manifest,lengths,workers)
    budget={'candidate_cap_seconds':3000.,'global_available_seconds_for_attempt':2500.,
            'worker_caps_seconds':[2000/workers]*workers,'workers':[{'gpu':str(i)} for i in range(workers)],
            'disk_min_free_bytes':1,'release_guard_seconds':2.,'allocation_authorized':False,
            'prior_candidate_transactions':[{'transaction_id':'synthetic_qualification','seconds':25.,'already_in_global_ledger':True}]}
    plan={'attempt_id':'synthetic_attempt','schedule':schedule,'budget':budget,'request_keys_by_edge':KEYS,
          'runtime':{'expected_matrix_protocol_hash':'b'*64},'forecast_margin':.25}
    return rows,manifest,lengths,plan


def profiles(plan):
    keys=set(k for values in KEYS.values() for k in values)
    return [{'index':u['index'],'worker':u['worker'],'length_stratum':u['length_stratum'],
             'status':'COMPLETE','terminal_conditions':11,'cost_fields_complete':True,'missing_request_timings':0,
             'request_max_seconds':{k:.01 for k in keys},'nonnegative_wall_residual_seconds':.63,
             'unit_wall_seconds':1.5,'overflow_conditions':0} for u in plan['schedule']['units'][:8]]


def matrix_data(unit):
    geo=next(r for r in GEOMETRY['per_probe_edge'] if r['probe_edge']==unit['edge_index'])
    conditions=[]
    for key in core.TERMINALS:
        seed_label,name=key.split('/');s=int(seed_label[1:]);mask=None
        if name not in ('INDEPENDENT12','SINGLE_M'):
            bits=4095 if name.startswith('A') else 674 if name.startswith('B') else 0
            if name.endswith('MINUS_E'):bits&=~(1<<unit['edge_index'])
            mask=[(bits>>j)&1 for j in range(12)]
        trace=[{'node':n,'model':'mistral' if n=='aggregate' or int(n[-1])%2 else 'qwen',
                'generation':{'elapsed_seconds':.01,'text':'Synthetic paid output'}} for n in geo['closures'][name]]
        conditions.append({'key':key,'name':name,'seed_number':s,'seed':core.SEEDS[s-1],'mask':mask,
                           'status':'COMPLETE','request_trace':trace})
    return {'question_id':'hotpotqa_'+unit['id'],'question_fingerprint':unit['payload_sha256'],
            'seed1':unit['seed1'],'seed2':unit['seed2'],'probe_edge':unit['edge_index'],
            'matrix_protocol_hash':'b'*64,'conditions':conditions,'status':'COMPLETE',
            'scores_computed':False,'condition_set_terminal':True,
            'total_physical_cost':{'cost_fields_complete':True,'calls':87},'known_physical_cost_components':{'calls':87}}


def publish_unit(root,unit,profile=None):
    directory=root/f"worker_{unit['worker']}"/f"unit_{unit['index']:03d}";directory.mkdir(parents=True,exist_ok=False)
    raw=storage.write_new(directory/'MATRIX.json',{'private_raw_generation':'Synthetic fixture only'})
    row={'index':unit['index'],'worker':unit['worker'],'plan_sha256':PIN,'matrix_protocol_hash':'b'*64,
         'matrix_sha256':raw,'terminal_conditions':11,'terminal_keys':list(core.TERMINALS),'scores_computed':False,
         'overflow_conditions':0,'cost_fields_complete':True,'cost_profile':profile}
    storage.write_new(directory/'UNIT_RECEIPT.json',row)


class FakeProcess:
    def __init__(self,pid):self.pid=pid;self.code=None
    def poll(self):return self.code


class FakeClock:
    def __init__(self):self.now=10.;self.on_sleep=None
    def __call__(self):return self.now
    def sleep(self,seconds):
        self.now+=seconds
        if self.on_sleep:self.on_sleep()


class OrchestrationTests(unittest.TestCase):
    def test_schedule_is_reproducible_balanced_and_preserves_remaining_order(self):
        _,m,l,p=fixture(3);s=p['schedule']
        self.assertEqual(s,core.build_schedule(m,l,3));self.assertEqual(len(s['prefix_indices']),8)
        self.assertEqual([len(set(s['prefix_indices'])&set(g)) for g in s['strata']],[1]*8)
        self.assertEqual(s['remainder_indices'],[i for i in range(384) if i not in s['prefix_indices']])
        self.assertEqual(len({u['index'] for u in s['units']}),384)

    def test_forecast_includes_margin_load_idle_and_prior_once(self):
        *_,p=fixture();value=core.forecast(p['schedule'],profiles(p),KEYS,[100.,110.],p['budget'])
        self.assertEqual(value['status'],'FIT')
        self.assertAlmostEqual(sum(value['predicted_remaining_seconds_by_worker']),376*1.5*1.25+4)
        self.assertAlmostEqual(value['predicted_attempt_sum_seconds'],919.)
        self.assertAlmostEqual(value['predicted_candidate_total_seconds'],944.)

    def test_profile_identity_swap_and_missing_time_rejected(self):
        *_,p=fixture();rows=profiles(p);rows[0]['length_stratum'],rows[1]['length_stratum']=rows[1]['length_stratum'],rows[0]['length_stratum']
        with self.assertRaises(ValueError):core.forecast(p['schedule'],rows,KEYS,[0.,0.],p['budget'])
        rows=profiles(p);rows[0]['missing_request_timings']=1
        with self.assertRaises(ValueError):core.forecast(p['schedule'],rows,KEYS,[0.,0.],p['budget'])

    def test_prior_transaction_duplicate_and_unallocated_execution_rejected(self):
        *_,p=fixture();budget=copy.deepcopy(p['budget']);budget['prior_candidate_transactions']*=2
        with self.assertRaises(ValueError):core.validate_budget(budget,2)
        with self.assertRaises(ValueError):core.validate_budget(p['budget'],2,executable=True)

    def test_sum_occupancy_is_not_parallel_wall(self):
        ledger=core.ResidenceLedger([100,100],200);ledger.start(0,10);ledger.start(1,10)
        ledger.close(0,30);ledger.close(1,35)
        self.assertEqual(ledger.spent(100),[20,25]);self.assertEqual(sum(ledger.spent(100)),45)

    def test_monitor_is_an_allowlist(self):
        row={'terminal_conditions':11,'overflow_conditions':0,'cost_fields_complete':True,'question':'DO NOT EXPOSE','answer':'NO','raw':'NO','quality':1}
        text=json.dumps(core.monitor_counts([row],[2.,3.]));self.assertNotIn('DO NOT EXPOSE',text)
        for forbidden in ('answer','quality','raw'):self.assertNotIn(forbidden,text)

    def test_matrix_fingerprint_seed_name_mask_bindings(self):
        *_,p=fixture();u=p['schedule']['units'][0];d=matrix_data(u)
        self.assertTrue(core.validate_matrix_result(d,u))
        for mutate in (lambda x:x.update(question_fingerprint='c'*64),
                       lambda x:x['conditions'][0].update(seed_number=-1),
                       lambda x:x['conditions'][0].update(name='B'),
                       lambda x:x['conditions'][0].update(mask=[0]*12)):
            bad=copy.deepcopy(d);mutate(bad)
            with self.assertRaises(ValueError):core.validate_matrix_result(bad,u)

    def test_conflicting_overflow_completion_and_prefix_count_rejected(self):
        *_,p=fixture();u=p['schedule']['units'][0]
        for mutate in (lambda x:x['conditions'][0].update(status='INPUT_REJECTED'),
                       lambda x:x.update(status='COMPLETE_WITH_INPUT_REJECTIONS')):
            bad=matrix_data(u);mutate(bad)
            with self.assertRaises(ValueError):core.validate_matrix_result(bad,u)
        timing=profiles(p);timing[0]['overflow_conditions']=1
        with self.assertRaises(ValueError):core.forecast(p['schedule'],timing,KEYS,[0.,0.],p['budget'])

    def test_atomic_no_overwrite_publication(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'RELEASE.json';original=storage.os.link;observed=[]
            def link(src,dst):
                self.assertFalse(Path(dst).exists());self.assertEqual(json.loads(Path(src).read_bytes()),{'ok':True})
                observed.append(True);return original(src,dst)
            with patch.object(storage.os,'link',link):storage.write_new(target,{'ok':True})
            self.assertEqual(observed,[True])
            with self.assertRaises(FileExistsError):storage.write_new(target,{'wrong':True})
            self.assertEqual(json.loads(target.read_bytes()),{'ok':True})

    def test_kill_refuses_other_process_group(self):
        process=FakeProcess(900);calls=[]
        class OS:
            def getpgid(self,pid):return 901
            def getpgrp(self):return 800
            def killpg(self,pid,sig):calls.append(pid)
        with self.assertRaises(RuntimeError):kill_owned_group(process,900,system=OS())
        self.assertEqual(calls,[])
        class Own(OS):
            def getpgid(self,pid):return 900
        with patch('production_guard.signal.SIGKILL',9,create=True):
            self.assertTrue(kill_owned_group(process,900,system=Own()))
        self.assertEqual(calls,[900])

    def test_worker_saves_durable_generation_even_if_partial_write_fails(self):
        rows,_,_,p=fixture();u=p['schedule']['units'][0];calls=[]
        class Result:
            def to_dict(self):return matrix_data(u)
        class Failure(RuntimeError):
            partial_result=Result()
        class Matrix:
            def execute(self,*args):calls.append(1);raise Failure('synthetic')
        runtime={'collector':payload,'matrix':Matrix(),'extract_error':lambda e:[{'returned_generation':{'text':'Synthetic paid output'},'native_events':[{'call_id':'synthetic'}]}]}
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/'unit';original=worker.write_new
            def writer(path,value):
                if path.name=='PARTIAL_MATRIX.json':raise OSError('synthetic transient failure')
                return original(path,value)
            with patch.object(worker,'write_new',writer):
                with self.assertRaises(Failure):worker.execute_unit(runtime,rows[u['index']],u,directory,PIN,{})
            self.assertIn('Synthetic paid output',(directory/'DURABLE_ERROR_EVIDENCE.json').read_text())
            self.assertEqual(calls,[1]);self.assertFalse((directory/'UNIT_RECEIPT.json').exists())

    def test_worker_preserves_input_overflow_terminal(self):
        rows,_,_,p=fixture();u=p['schedule']['units'][0];d=matrix_data(u)
        d['status']='COMPLETE_WITH_INPUT_REJECTIONS';d['conditions'][0]['status']='INPUT_REJECTED';d['total_physical_cost']['cost_fields_complete']=False
        class Result:
            def to_dict(self):return d
        class Matrix:
            def execute(self,*args):return Result()
        with tempfile.TemporaryDirectory() as td:
            receipt=worker.execute_unit({'collector':payload,'matrix':Matrix(),'extract_error':lambda e:[]},rows[u['index']],u,Path(td)/'unit',PIN,{})
            self.assertEqual(receipt['terminal_conditions'],11);self.assertEqual(receipt['overflow_conditions'],1)
            self.assertFalse(receipt['cost_fields_complete'])

    def test_parent_releases_only_after_eight_then_closes_384_4224(self):
        *_,p=fixture();clock=FakeClock();processes=[FakeProcess(900),FakeProcess(901)]
        ledger=core.ResidenceLedger(p['budget']['worker_caps_seconds'],p['budget']['global_available_seconds_for_attempt'])
        for w in range(2):ledger.start(w,0.)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pp=profiles(p)
            for u,profile in zip(p['schedule']['units'][:8],pp):publish_unit(root,u,profile)
            for w in range(2):storage.write_new(root/f'worker_{w}/PREFIX_COMPLETE.json',{'worker':w,'plan_sha256':PIN,'profiles':[x for x in pp if x['worker']==w]})
            advanced=[]
            def on_sleep():
                if (root/'RELEASE_REMAINDER.json').exists() and not advanced:
                    self.assertTrue((root/'FORECAST_FROZEN.json').exists());advanced.append(True)
                    for u in p['schedule']['units'][8:]:publish_unit(root,u)
                    for w in range(2):
                        storage.write_new(root/f'worker_{w}/WORKER_COMPLETE.json',{'status':'COMPLETE_COLLECTOR_ONLY','plan_sha256':PIN})
                        processes[w].code=0
            clock.on_sleep=on_sleep
            result=parent.supervise(p,PIN,root,processes,ledger,clock=clock,sleep=clock.sleep,disk_guard=lambda *a:None,kill_group=lambda p,g:setattr(p,'code',-9))
            self.assertEqual(result['completed_units'],384);self.assertEqual(result['terminal_conditions'],4224)
            self.assertEqual(result['status'],'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT')
            self.assertFalse(result['score_release_authorized']);self.assertEqual(advanced,[True])

    def test_disk_guard_kills_owned_workers_without_release_or_retry(self):
        *_,p=fixture();clock=FakeClock();procs=[FakeProcess(900),FakeProcess(901)]
        ledger=core.ResidenceLedger([100,100],200)
        for w in range(2):ledger.start(w,0)
        kills=[]
        def kill(proc,group):kills.append(group);proc.code=-9
        def disk(*args):raise OSError('synthetic low space')
        with tempfile.TemporaryDirectory() as td:
            result=parent.supervise(p,PIN,Path(td),procs,ledger,clock=clock,sleep=clock.sleep,disk_guard=disk,kill_group=kill)
            self.assertEqual(result['status'],'INCOMPLETE_DISK_GUARD');self.assertFalse((Path(td)/'RELEASE_REMAINDER.json').exists())
            self.assertEqual(set(kills),{900,901});self.assertFalse(result['automatic_retry'])

    def test_cap_cost_is_never_clamped(self):
        *_,p=fixture();clock=FakeClock();procs=[FakeProcess(900),FakeProcess(901)]
        ledger=core.ResidenceLedger([5,5],10)
        for w in range(2):ledger.start(w,0)
        with tempfile.TemporaryDirectory() as td:
            result=parent.supervise(p,PIN,Path(td),procs,ledger,clock=clock,sleep=clock.sleep,disk_guard=lambda *a:None,kill_group=lambda p,g:setattr(p,'code',-9))
            self.assertEqual(result['status'],'INCOMPLETE_RESOURCE_HARDCAP');self.assertGreaterEqual(result['resident_sum_seconds'],20.)

    def test_unaffordable_forecast_does_not_release_or_change_sample_size(self):
        *_,p=fixture();p['budget']['worker_caps_seconds']=[50.,50.]
        clock=FakeClock();processes=[FakeProcess(900),FakeProcess(901)]
        ledger=core.ResidenceLedger([50.,50.],2500.)
        for w in range(2):ledger.start(w,0.)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pp=profiles(p)
            for u,profile in zip(p['schedule']['units'][:8],pp):publish_unit(root,u,profile)
            for w in range(2):storage.write_new(root/f'worker_{w}/PREFIX_COMPLETE.json',{'worker':w,'plan_sha256':PIN,'profiles':[x for x in pp if x['worker']==w]})
            result=parent.supervise(p,PIN,root,processes,ledger,clock=clock,sleep=clock.sleep,
                disk_guard=lambda *a:None,kill_group=lambda p,g:setattr(p,'code',-9))
            self.assertEqual(result['status'],'INCOMPLETE_RESOURCE_FORECAST')
            self.assertFalse((root/'RELEASE_REMAINDER.json').exists())
            self.assertEqual(json.loads((root/'FORECAST_FROZEN.json').read_bytes())['remaining_units'],376)
            self.assertEqual(result['completed_units'],8)

    def test_consistent_matrix_and_receipt_rewrite_is_rejected_at_close(self):
        *_,p=fixture();u=p['schedule']['units'][0]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);publish_unit(root,u);seen={}
            parent.collect_receipts(root,p,PIN,seen)
            folder=root/f"worker_{u['worker']}"/f"unit_{u['index']:03d}"
            matrix=folder/'MATRIX.json';matrix.write_bytes(core.canonical({'altered':'synthetic bytes'}))
            receipt=folder/'UNIT_RECEIPT.json';row=json.loads(receipt.read_bytes())
            row['matrix_sha256']=storage.file_sha(matrix);receipt.write_bytes(core.canonical(row))
            self.assertEqual(len(parent.collect_receipts(root,p,PIN)),1)
            with self.assertRaises(ValueError):parent.verify_first_seen_receipts(root,seen)


if __name__=='__main__':unittest.main()
