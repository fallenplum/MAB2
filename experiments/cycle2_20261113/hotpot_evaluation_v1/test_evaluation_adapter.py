"""384 artificial records only; no dataset, weights, real labels or results."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'hotpot_applicability_executor_stage/frozen_dependencies/src'))
from graph_credit.executor import affected_closure
import evaluation_adapter as a

GOLD='red blue green violet amber black silver'
def encode(value):return (a.canonical(value)+'\n').encode()
def cost(records):return a._node_costs(records)[1]


def synthetic_fixture():
    bindings={key:a.digest(key) for key in a.BINDINGS}
    bindings.update(scorer_sha256=a.HELPER_PINS['hotpot_support'],statistics_sha256=a.HELPER_PINS['applicability_statistics'],
                    frontier_sha256=a.HELPER_PINS['point_frontier'],selector_sha256=a.SELECTOR_PIN,
                    evaluation_adapter_sha256=a.sha(Path(a.__file__).read_bytes()))
    questions=[];files={};units=[];labels={}
    answers={key:' '.join(GOLD.split()[:j]) for j,key in enumerate(tuple('S1/'+n for n in a.GRAPHS)+tuple('S2/'+n for n in a.GRAPHS))}
    answers.update({'S1/NO_EDGE':'red blue green','S1/INDEPENDENT12':'wrong','S1/SINGLE_M':GOLD})
    for index in range(384):
        qid=f'artificial_{index:03d}';edge=a.EDGES[index//96];labels[qid]=GOLD
        q={'id':qid,'question_group_sha256':a.digest(['group',index]),'input_sha256':a.digest(['raw_input',index]),
           'content_sha256':a.digest(['content',index]),'edge_index':edge,'generation_seed1':a.SEEDS[0],'generation_seed2':a.SEEDS[1]}
        questions.append(q);payload={'id':'hotpotqa_'+qid,'question':'Artificial question '+str(index),'context':'Artificial full context.'}
        unit={'index':index,'worker':index%2,'id':qid,'edge_index':edge,'seed1':a.SEEDS[0],'seed2':a.SEEDS[1],
              'input_sha256':q['input_sha256'],'payload_sha256':a.digest(payload)}
        conditions=[];prior={}
        for key in a.ORDER:
            prefix,name=key.split('/');number=int(prefix[1:]);seed=a.SEEDS[number-1];mask=a._mask(name,edge)
            protocol=bindings['independent12_protocol_hash' if name=='INDEPENDENT12' else 'single_protocol_hash' if name=='SINGLE_M' else 'graph_protocol_hash']
            source_name={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(name)
            base_key=prefix+'/'+source_name if source_name else None;records={};trace=[]
            names=('aggregate',) if name=='SINGLE_M' else a.NODES
            reuse=set()
            if base_key:reuse=set(names)-set(affected_closure(tuple(prior[base_key]['mask']),tuple(mask),3))
            for node in names:
                if node in reuse:
                    record=copy.deepcopy(prior[base_key]['nodes'][node]);record['reused']=True
                    record['source_execution_id']=prior[base_key]['execution_id'];records[node]=record;continue
                cap=256 if node=='aggregate' else 128;model=a._model(node)
                parsed={'answer':answers[key] if node=='aggregate' else 'worker','evidence':'synthetic','uncertainty':''}
                messages=[{'role':'system','content':'synthetic role'},
                          {'role':'user','content':payload['question']+'\n\nSupplied context:\n'+payload['context']+'\n'+key+'/'+node}]
                node_seed=int(a.digest([payload['id'],seed,node])[:16],16)%(2**31-1)
                generation={'text':a.canonical(parsed),'input_tokens':100 if model=='qwen' else 120,'output_tokens':10,
                    'elapsed_seconds':.03,'gpu_seconds':.01 if model=='qwen' else .02,'gpu_peak_allocated_bytes':100,
                    'finish_reason':'eos','call_id':f'{index}/{key}/{node}'}
                record={'node':node,'messages':messages,'seed':node_seed,'generation':generation,'wire_message':a.canonical(parsed),
                    'parsed':parsed,'message_metadata':{'format_ok':True,'message_tokens':10},'received_message_tokens':0,
                    'self_state_tokens':0,'reused':False,'source_execution_id':None}
                record['input_fingerprint']=a.digest({'question':unit['payload_sha256'],'node':node,'messages':messages,
                    'seed':node_seed,'protocol':protocol,'max_new_tokens':cap})
                record['output_digest']=a.digest({k:record[k] for k in a.INTEGRITY});records[node]=record
                trace.append({'node':node,'model':model,'generation':generation,'messages':messages,'seed':node_seed})
            logical=cost(records);physical=cost({t['node']:records[t['node']] for t in trace})
            execution={'execution_id':f'{index}/{key}','question_id':payload['id'],'question_fingerprint':unit['payload_sha256'],
                       'seed':seed,'protocol_hash':protocol,'answer':records['aggregate']['parsed']['answer'],
                       'cache_reuse_count':len(reuse),'logical_service_cost':logical,'physical_cost':physical}
            if mask is not None:execution.update(mask=mask,nodes=records)
            else:
                order=['aggregate'] if name=='SINGLE_M' else [f'r{s}:w{r}' for r in range(4) for s in range(1,4)]+['aggregate']
                execution.update(status='COMPLETE',control_id=('single_mistral_full_context_direct_v1_cpu_staging' if name=='SINGLE_M' else 'independent_4roles_x3samples_plus_aggregate_v2_cost_reconciliation_staging'),slots=[])
                for slot_index,node in enumerate(order):
                    r=records[node]
                    execution['slots'].append({'index':slot_index,'node':node,'role':None if node=='aggregate' else int(node[-1]),
                        'sample':None if node=='aggregate' else int(node[1]),'model':a._model(node),'attempted_backend_request':True,
                        'status':'SUCCESS','record':r,'generation':r['generation'],'messages':r['messages'],'seed':r['seed'],
                        'input_fingerprint':r['input_fingerprint']})
            condition={'key':key,'name':name,'seed_number':number,'seed':seed,'mask':mask,'status':'COMPLETE',
                'execution':execution,'partial_execution':None,'error_type':None,'failure_category':None,
                'logical_service_cost':logical,'physical_cost':physical,'request_trace':trace,'cache_source_condition':base_key}
            prior[key]=execution;conditions.append(condition)
        all_costs=[c['logical_service_cost'] for c in conditions]
        total={k:(max(c[k] for c in all_costs) if k=='peak_allocated_bytes' else sum(c[k] for c in all_costs)) for k in all_costs[0]}
        matrix={'operator_id':'single_question_hotpot_applicability_matrix_v1_cpu_staging','matrix_protocol_hash':bindings['matrix_protocol_hash'],
            'question_id':payload['id'],'question_fingerprint':unit['payload_sha256'],'seed1':a.SEEDS[0],'seed2':a.SEEDS[1],
            'probe_edge':edge,'conditions':conditions,'status':'COMPLETE','condition_set_terminal':True,'scores_computed':False,
            'all_conditions_completed':True,'total_logical_service_cost':total,'total_physical_cost':{'calls':87}}
        directory=f"worker_{unit['worker']}/unit_{index:03d}"
        inp=encode({'plan_sha256':bindings['plan_sha256'],'unit':unit,'payload':payload});mat=encode(matrix)
        receipt={'schema':'hotpot_unit_receipt_v1','index':index,'worker':unit['worker'],'plan_sha256':bindings['plan_sha256'],
            'status':'COMPLETE','matrix_protocol_hash':bindings['matrix_protocol_hash'],'matrix_sha256':a.sha(mat),
            'input_file_sha256':a.sha(inp),'terminal_keys':list(a.ORDER),'terminal_conditions':11,'scores_computed':False,
            'overflow_conditions':0,'total_physical_cost':matrix['total_physical_cost']}
        rows={'INPUT.json':inp,'MATRIX.json':mat,'UNIT_RECEIPT.json':encode(receipt)}
        files.update({directory+'/'+name:value for name,value in rows.items()})
        units.append({'index':index,'directory':directory,'sha256':{name:a.sha(value) for name,value in rows.items()}})
    manifest={'classification':'PRIVATE_NO_LABELS','pins':{'protocol_sha256':bindings['scientific_protocol_sha256'],'selector_sha256':a.SELECTOR_PIN},'questions':questions}
    manifest_bytes=encode(manifest);bindings['selection_manifest_sha256']=a.sha(manifest_bytes)
    raw={'status':'CLOSED_SNAPSHOT','plan_sha256':bindings['plan_sha256'],'scores_accessed':False,'files':{name:a.sha(v) for name,v in files.items()}}
    residence={'all_workers_closed':True,'unknown_unclosed_residence':False,'kill_integrity_errors':[],
               'closed_sum_seconds':123.,'cap_used_as_cost':False}
    files['RAW_OUTPUT_MANIFEST.json']=encode(raw);files['PARENT_RESIDENCE.json']=encode(residence)
    attempt={'status':'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT','all_worker_residences_closed':True,
             'completed_units':384,'terminal_conditions':4224,'overflow_conditions':0,'plan_sha256':bindings['plan_sha256'],
             'raw_manifest_sha256':a.sha(files['RAW_OUTPUT_MANIFEST.json']),'residence_sha256':a.sha(files['PARENT_RESIDENCE.json']),
             'scores_computed':False,'score_release_authorized':False,'automatic_retry':False}
    files['ATTEMPT_RESULT.json']=encode(attempt)
    review={'schema':'hotpot_closed_archive_score_release_v1','decision':'APPROVED_FOR_SCORING','bindings':bindings,
            'scientific_protocol_status':'FROZEN','reviewed_scopes':sorted(a.REQUIRED_REVIEW_SCOPES),'unique_questions':384,
            'terminal_conditions':4224,'runtime_failure_count':0,'all_attempts_closed':True,'physical_ledger_closed':True,
            'label_vault_sha256':a.sha(encode(labels)),'input_overflow_conditions':0,
            'archive_metadata_sha256':{name:a.sha(files[name]) for name in a.METADATA},'units':list(reversed(units)),
            'test_evidence':'ARTIFICIAL_FIXTURE_ONLY_NOT_REAL_AUTHORIZATION'}
    raw_review=encode(review)
    return {'selection_manifest_bytes':manifest_bytes,'files':files,'review_bytes':raw_review,
            'expected_review_sha256':a.sha(raw_review),'expected_bindings':bindings},encode(labels)


def resign(arguments,matrix_mutator=None,input_mutator=None,overflow=0):
    """Artificial reviewer fixture only: tests internal validation beyond file SHA."""
    result=dict(arguments);files=dict(arguments['files']);review=a.decode(arguments['review_bytes'])
    entry=next(e for e in review['units'] if e['index']==0);directory=entry['directory']
    matrix=a.decode(files[directory+'/MATRIX.json']);inp=a.decode(files[directory+'/INPUT.json'])
    receipt=a.decode(files[directory+'/UNIT_RECEIPT.json'])
    if matrix_mutator:matrix_mutator(matrix)
    if input_mutator:input_mutator(inp)
    files[directory+'/MATRIX.json']=encode(matrix);files[directory+'/INPUT.json']=encode(inp)
    receipt.update(matrix_sha256=a.sha(files[directory+'/MATRIX.json']),input_file_sha256=a.sha(files[directory+'/INPUT.json']),
        status=matrix['status'],overflow_conditions=overflow,total_physical_cost=matrix['total_physical_cost'])
    files[directory+'/UNIT_RECEIPT.json']=encode(receipt)
    for name in entry['sha256']:entry['sha256'][name]=a.sha(files[directory+'/'+name])
    raw=a.decode(files['RAW_OUTPUT_MANIFEST.json']);raw['files'].update({directory+'/'+n:h for n,h in entry['sha256'].items()})
    files['RAW_OUTPUT_MANIFEST.json']=encode(raw)
    attempt=a.decode(files['ATTEMPT_RESULT.json']);attempt.update(raw_manifest_sha256=a.sha(files['RAW_OUTPUT_MANIFEST.json']),overflow_conditions=overflow)
    if overflow:attempt['status']='COMPLETE_WITH_CAPACITY_FAILURES'
    files['ATTEMPT_RESULT.json']=encode(attempt)
    review['input_overflow_conditions']=overflow
    review['archive_metadata_sha256']={name:a.sha(files[name]) for name in a.METADATA}
    result.update(files=files,review_bytes=encode(review),expected_review_sha256=a.sha(encode(review)))
    return result


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arguments,cls.labels=synthetic_fixture();cls.bound=a.bind_archive(**cls.arguments)
    def test_named_axes_selection_order_and_true_zero_q_single_cost(self):
        result=a.score_bound_archive(self.bound,self.labels,run_statistics=False)
        expected=[[0.,.25,4/9,.6],[8/11,10/12,12/13,1.]]
        for i in (0,96,383):
            for seed in (0,1):
                for column in range(4):self.assertAlmostEqual(result['f1']['graphs'][i][seed][column],expected[seed][column])
        self.assertEqual(result['em']['graphs'][0],[[0.,0.,0.,0.],[0.,0.,0.,1.]])
        single=result['full_logical_costs_by_model']['SINGLE_M']
        self.assertEqual(single['qwen'],dict.fromkeys(a.FIELDS,0));self.assertEqual(single['mistral']['calls'],384)
        self.assertEqual(result['full_logical_costs_by_model']['B']['qwen']['calls'],2304)
        self.assertEqual(result['full_logical_costs_by_model']['B']['mistral']['calls'],2688)
        self.assertEqual(result['frontier']['status'],'DOMINATED')
    def test_existing_statistics_four_gates_called_without_modification(self):
        result=a.score_bound_archive(self.bound,self.labels)
        self.assertEqual(list(result['statistics']['primary_EM']),list(a._helper('applicability_statistics').NAMES))
        self.assertEqual(result['statistics']['bootstrap']['family_size'],4)
        self.assertEqual(result['statistics']['bootstrap']['seed'],261113103)
        self.assertFalse(result['statistics']['decision']['GPU_admission_granted'])
    def test_wrong_edge_seed_qid_protocol_and_node_hash_rejected(self):
        changes=[lambda m:m['conditions'][1]['mask'].__setitem__(1,0),
                 lambda m:m['conditions'][1].__setitem__('seed',42),
                 lambda m:m.__setitem__('question_id','hotpotqa_other'),
                 lambda m:m['conditions'][0]['execution'].__setitem__('protocol_hash','a'*64),
                 lambda m:m['conditions'][0]['execution']['nodes']['aggregate'].__setitem__('output_digest','b'*64)]
        for change in changes:
            with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=change))
    def test_missing_unit_duplicate_question_and_missing_terminal_rejected(self):
        missing=dict(self.arguments);missing['files']=dict(missing['files']);missing['files'].pop('worker_0/unit_000/MATRIX.json')
        with self.assertRaises((KeyError,a.BindingError)):a.bind_archive(**missing)
        with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=lambda m:m['conditions'].pop()))
        with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,input_mutator=lambda x:x['unit'].__setitem__('id','artificial_001')))
    def test_infrastructure_does_not_become_zero_or_success_subset(self):
        for state in ('INCOMPLETE','NOT_RUN','FAILED'):
            with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=lambda m:m['conditions'][0].__setitem__('status',state)))
    def test_cache_logical_cost_omission_rejected(self):
        def change(matrix):
            c=matrix['conditions'][1];self.assertGreater(c['execution']['cache_reuse_count'],0)
            c['logical_service_cost']=c['physical_cost'];c['execution']['logical_service_cost']=c['physical_cost']
        with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=change))
    def test_malformed_normalized_answer_rejected_without_raw_fallback(self):
        def change(matrix):
            r=matrix['conditions'][0]['execution']['nodes']['aggregate'];r['parsed']['answer']=123
            r['output_digest']=a.digest({k:r[k] for k in a.INTEGRITY})
        with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=change))
    def test_format_false_scores_zero_even_if_parsed_answer_matches_gold(self):
        def change(matrix):
            r=matrix['conditions'][6]['execution']['slots'][0]['record'];r['message_metadata']['format_ok']=False
            r['output_digest']=a.digest({k:r[k] for k in a.INTEGRITY})
        bound=a.bind_archive(**resign(self.arguments,matrix_mutator=change))
        result=a.score_bound_archive(bound,self.labels,run_statistics=False)
        self.assertEqual(result['em']['single'][0],0.);self.assertEqual(result['format_failure_conditions'],1)
    def test_real_preflight_terminal_zero_keeps384_and_blocks_frontier(self):
        def change(matrix):
            c=matrix['conditions'][6];e=c['execution']
            c.update(status='INPUT_REJECTED',execution=None,logical_service_cost=None,failure_category='INPUT_TOKEN_LIMIT',
                partial_execution={'protocol_hash':e['protocol_hash'],'question_fingerprint':e['question_fingerprint'],'seed':e['seed']},
                request_trace=[{'generation':None,'error_type':'ValueError'}])
            matrix.update(status='COMPLETE_WITH_INPUT_REJECTIONS',all_conditions_completed=False,total_logical_service_cost=None)
        bound=a.bind_archive(**resign(self.arguments,matrix_mutator=change,overflow=1))
        result=a.score_bound_archive(bound,self.labels,run_statistics=False)
        self.assertEqual(len(result['em']['single']),384);self.assertEqual(result['em']['single'][0],0.)
        self.assertEqual(result['frontier']['status'],'NOT_ASSESSED')
        self.assertIsNone(result['full_logical_costs_by_model']['SINGLE_M']['mistral']['calls'])
        self.assertEqual(result['full_logical_costs_by_model']['SINGLE_M']['qwen']['calls'],0)
    def test_labels_are_not_loaded_until_external_archive_review_succeeds(self):
        calls=[];bad=dict(self.arguments);bad['expected_review_sha256']='0'*64
        def loader(request):calls.append(request);return self.labels
        with self.assertRaises(a.BindingError):a.evaluate_after_release(label_loader=loader,**bad)
        self.assertEqual(calls,[])
        result=a.evaluate_after_release(label_loader=loader,run_statistics=False,**self.arguments)
        self.assertEqual(len(calls),1);self.assertEqual(result['audit_inputs']['complete_conditions'],4224)
    def test_label_vault_exact_bytes_and_ids_required(self):
        with self.assertRaises(a.BindingError):a.score_bound_archive(self.bound,b'{}',run_statistics=False)
    def test_duplicate_selection_ids_rejected_even_with_new_review_hash(self):
        args=dict(self.arguments);manifest=a.decode(args['selection_manifest_bytes'])
        manifest['questions'][1]['id']=manifest['questions'][0]['id']
        args['selection_manifest_bytes']=encode(manifest);args['expected_bindings']=dict(args['expected_bindings'])
        args['expected_bindings']['selection_manifest_sha256']=a.sha(args['selection_manifest_bytes'])
        review=a.decode(args['review_bytes']);review['bindings']=args['expected_bindings']
        args['review_bytes']=encode(review);args['expected_review_sha256']=a.sha(args['review_bytes'])
        with self.assertRaises(a.BindingError):a.bind_archive(**args)
    def test_aggregate_slot_is_found_by_explicit_alias_and_index_not_list_position(self):
        def change(matrix):
            slots=matrix['conditions'][5]['execution']['slots'];slots[0],slots[-1]=slots[-1],slots[0]
        bound=a.bind_archive(**resign(self.arguments,matrix_mutator=change))
        result=a.score_bound_archive(bound,self.labels,run_statistics=False)
        self.assertEqual(result['em']['independent12'][0],0.)
        self.assertEqual(result['full_logical_costs_by_model']['INDEPENDENT12']['qwen']['calls'],2304)
    def test_scoring_rejects_post_binding_mutation(self):
        old=self.bound.logical_costs[0]['S1/B']['qwen']['calls']
        try:
            self.bound.logical_costs[0]['S1/B']['qwen']['calls']=0
            with self.assertRaisesRegex(a.BindingError,'changed before scoring'):a.score_bound_archive(self.bound,self.labels,run_statistics=False)
        finally:self.bound.logical_costs[0]['S1/B']['qwen']['calls']=old
    def test_overflow_cannot_be_used_to_hide_infrastructure_failure(self):
        def change(matrix):
            c=matrix['conditions'][0];e=c['execution']
            c.update(status='INPUT_REJECTED',execution=None,logical_service_cost=None,failure_category='INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE',
                partial_execution={'protocol_hash':e['protocol_hash'],'question_fingerprint':e['question_fingerprint'],'seed':e['seed']},
                request_trace=[{'generation':None,'error_type':'RuntimeError'}])
            matrix.update(status='COMPLETE_WITH_INPUT_REJECTIONS',all_conditions_completed=False,total_logical_service_cost=None)
        with self.assertRaises(a.BindingError):a.bind_archive(**resign(self.arguments,matrix_mutator=change,overflow=1))

if __name__=='__main__':unittest.main(verbosity=2)
