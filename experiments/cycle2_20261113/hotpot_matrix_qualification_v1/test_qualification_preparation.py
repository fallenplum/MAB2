"""Synthetic CPU integration, no model or benchmark imports/run."""
import copy,dataclasses,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
MATRIX=BASE/'hotpot_applicability_executor_stage'
sys.path[:0]=[str(MATRIX),str(BASE/'durable_backend_stage')]
from test_matrix_executor import models
from matrix_executor import HotpotApplicabilityMatrixExecutor,MatrixExecutionError
from graph_credit.executor import digest,GraphConfig
from durable_backend import DurableBackend,DurableBackendError,extract_durable_error_evidence,read_journal
from matrix_qualification_audit import verify_matrix,compare_fresh,verify_global_calls,verify_durable
from qualify_hotpot_matrix import RequestCap,checkpointed_matrix_class,validate_launch_plan

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.root=Path(self.tmp.name)
        self.protocol=json.loads((ROOT/'PROTOCOL.json').read_bytes())
        self.question={'id':'synthetic_cpu','question':'Which district holds records?',
            'context':'Pavilion records are held by Cedar Hall. Cedar Hall is in Marlow District.'}
    def tearDown(self):self.tmp.cleanup()
    def make(self,**flags):
        raw,state=models(**flags);holder={};cap=RequestCap();paths={name:self.root/(name+'.jsonl') for name in raw}
        wrapped={name:DurableBackend(b,name,paths[name],
            context_provider=lambda:{'node':holder['matrix'].router._selected.get(),'repeat':state.get('repeat')},before_request=cap) for name,b in raw.items()}
        matrix=checkpointed_matrix_class(HotpotApplicabilityMatrixExecutor)(wrapped)
        holder['matrix']=matrix;matrix.configure_checkpoint(lambda conditions:None)
        return matrix,raw,wrapped,paths,state,cap
    def run_two(self):
        matrix,raw,wrapped,paths,state,cap=self.make()
        results=[]
        for repeat in (1,2):
            state['repeat']=repeat;results.append(matrix.execute(self.question,26111301,26111302,1).to_dict())
        return results,matrix,raw,paths,cap
    def test_two_fresh_matrices_all_174_raw_generations_and_262_logical_records_close(self):
        results,matrix,raw,paths,cap=self.run_two()
        for result in results:self.assertEqual(verify_matrix(result,self.question,self.protocol,digest),87)
        self.assertEqual(compare_fresh(*results)['logical_node_records_compared'],131)
        self.assertEqual(verify_global_calls(results,matrix.router,raw,self.protocol,digest)[0]['calls'],174)
        self.assertEqual(verify_durable(results,raw,paths,digest)['canonical_generation_bindings'],174)
        self.assertEqual(cap.admitted,174)
    def test_independent_logical_reconstruction_detects_same_wrong_cost_in_both_fresh_runs(self):
        results,_,_,_,_=self.run_two()
        for key in ('input_tokens','output_tokens','received_message_tokens','self_state_tokens','model_elapsed_seconds','gpu_seconds','peak_allocated_bytes'):
            altered=copy.deepcopy(results)
            for result in altered:result['conditions'][0]['logical_service_cost'][key]+=1
            compare_fresh(*altered) # equal error is deliberately invisible to fresh comparison
            with self.assertRaises(AssertionError):verify_matrix(altered[0],self.question,self.protocol,digest)
    def test_raw_generation_tampering_is_detected_even_when_native_counts_match(self):
        results,_,raw,paths,_=self.run_two()
        path=paths['qwen'];events=[json.loads(x) for x in path.read_text().splitlines()]
        next(e for e in events if e['event']=='RETURNED')['generation']['text']='synthetic corruption'
        path.write_bytes(('\n'.join(json.dumps(e) for e in events)+'\n').encode())
        with self.assertRaises(AssertionError):verify_durable(results,raw,paths,digest)
    def test_raw_source_and_request_metadata_tampering_is_detected(self):
        results,_,raw,paths,_=self.run_two();path=paths['qwen']
        original=[json.loads(x) for x in path.read_text().splitlines()]
        for field in ('helper','repeat','cap','config'):
            events=copy.deepcopy(original)
            if field=='helper':events[0]['helper_sha256']='a'*64
            else:
                request=events[1]['request_id']
                for event in events[1:]:
                    if event['request_id']!=request:continue
                    if field=='repeat':event['context']['repeat']=99
                    elif field=='cap':event['max_new_tokens']=999
                    else:event['config']['temperature']=.12345
            path.write_bytes(('\n'.join(json.dumps(e) for e in events)+'\n').encode())
            with self.assertRaises(AssertionError):verify_durable(results,raw,paths,digest)
    def test_hard_cap_stops_175th_before_native_generation(self):
        _,matrix,raw,_,cap=self.run_two()
        before=sum(len(b.physical_calls) for b in raw.values())
        with self.assertRaises(MatrixExecutionError):matrix.execute(self.question,26111301,26111302,1)
        self.assertEqual(before,174);self.assertEqual(sum(len(b.physical_calls) for b in raw.values()),174)
        self.assertEqual(cap.admitted,174)
    def test_qualification_stops_first_overflow_instead_of_matrix_fallback(self):
        matrix,raw,_,paths,state,cap=self.make(overflow_at={1})
        with self.assertRaisesRegex(RuntimeError,'first noncomplete'):matrix.execute(self.question,26111301,26111302,1)
        self.assertEqual(len(state['requests']),1);self.assertEqual(cap.admitted,1)
        self.assertEqual(matrix.live_conditions[0].status,'INPUT_REJECTED')
        self.assertTrue(all(c.status=='NOT_RUN' for c in matrix.live_conditions[1:]))
    def test_returned_journal_failure_survives_frozen_matrix_exception_chain_and_is_not_free(self):
        matrix,raw,wrapped,paths,state,cap=self.make();backend=wrapped['qwen'];append=backend._append
        def broken(event):
            if event['event']=='RETURNED':raise OSError('synthetic disk fault')
            append(event)
        with patch.object(backend,'_append',side_effect=broken):
            with self.assertRaises(MatrixExecutionError) as caught:matrix.execute(self.question,26111301,26111302,1)
        evidence=extract_durable_error_evidence(caught.exception)
        self.assertEqual(len(evidence),1);self.assertEqual(evidence[0]['generation'],dataclasses.asdict(state['requests'][0]['generation']))
        partial=caught.exception.partial_result.to_dict()
        self.assertEqual(partial['total_physical_cost']['calls'],1)
        self.assertGreater(partial['total_physical_cost']['input_tokens'],0)
        self.assertEqual(len(raw['qwen'].physical_calls),1)
        self.assertEqual(len(read_journal(paths['qwen'])['pending_request_ids']),1)
    def test_template_has_no_launch_permission(self):
        with self.assertRaises(AssertionError):validate_launch_plan(json.loads((ROOT/'PRIVATE_PLAN.template.json').read_bytes()))
    def test_original_guard_bytes_and_frozen_dependencies_are_exact(self):
        from qualify_hotpot_matrix import FIXED_PINS,QUAL_DIR,sha
        self.assertEqual(sha(ROOT/'gpu_job_guard.py'),FIXED_PINS[QUAL_DIR+'/gpu_job_guard.py'])
        self.assertEqual(sha(ROOT/'SYNTHETIC_INPUT.json'),FIXED_PINS[QUAL_DIR+'/SYNTHETIC_INPUT.json'])
    def test_child_rejects_changed_plan_bytes_after_arming_guard_before_model_code(self):
        import qualify_hotpot_matrix as qualification
        path=self.root/'private_plan.json';path.write_text('{}')
        args=['qualify','--plan',str(path),'--child','--supervisor-pid','123','--deadline-monotonic','456',
              '--expected-plan-sha256','a'*64]
        with patch.object(sys,'argv',args),patch.object(qualification,'arm_child_deadline') as arm,patch.object(qualification,'child') as child:
            with self.assertRaisesRegex(AssertionError,'plan byte identity'):qualification.main()
            arm.assert_called_once_with(123,456.0);child.assert_not_called()
    def test_complete_child_pipeline_with_mock_factory_and_json_configuration(self):
        import qualify_hotpot_matrix as qualification
        import graph_credit.backend as native_module
        raw,state=models();full=self.question['question']+'\n\nSupplied context:\n'+self.question['context']
        for name,backend in raw.items():
            backend.identity['model_files']={'synthetic-file':{'bytes':1,'sha256':'a'*64}}
            backend.runtime={'mock_only':True}
            original=backend.encode
            backend.encode=lambda text,original=original:([0]*3744 if text==full else original(text))
        expected=HotpotApplicabilityMatrixExecutor(raw).protocol_hash
        code=self.root/'code';qual=code/qualification.QUAL_DIR;qual.mkdir(parents=True)
        input_path=qual/'SYNTHETIC_INPUT.json';input_path.write_bytes(json.dumps(self.question).encode())
        protocol=copy.deepcopy(self.protocol);protocol['synthetic_input_sha256']=qualification.sha(input_path)
        (qual/'PROTOCOL.json').write_bytes(json.dumps(protocol).encode())
        out=self.root/'out';out.mkdir()
        plan={'source_manifest':{},'expected_matrix_protocol_hash':expected,'models':{
            name:{'path':name,'model_files':b.identity['model_files'],'allowed_metadata_files':[],
                  'expected_identity_sha256':digest(b.identity)} for name,b in raw.items()}}
        def factory(path,device,physical_log):
            self.assertIsNone(physical_log);self.assertEqual(device,'cuda:0');return raw[path]
        with patch.object(qualification,'pin_sources',return_value=code),patch.object(native_module,'HFBackend',side_effect=factory):
            qualification.child(plan,out)
        result=json.loads((out/'QUALIFICATION.json').read_bytes())
        self.assertEqual(result['status'],'PASS_ENGINEERING_REVIEW_REQUIRED');self.assertEqual(result['inference_calls'],174)
        self.assertEqual(result['durable_audit']['canonical_generation_bindings'],174)

if __name__=='__main__':unittest.main(verbosity=2)
