import dataclasses,json,types,unittest
import exogenous_path_operator as m

@dataclasses.dataclass
class Generation:
    text:str; input_tokens:int; output_tokens:int; elapsed_seconds:float
    gpu_seconds:float; gpu_peak_allocated_bytes:int; finish_reason:str; call_id:str
class Tokenizer:
    def apply_chat_template(self,*a,**kw):return list(range(10))
class Backend:
    def __init__(self,name):
        self.identity={'model':name};self.physical_calls=[];self.tokenizer=Tokenizer()
        self.model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=16384))
    def encode(self,text):return list(text.encode())
    def generate(self,messages,seed,cap,config):
        text='  '+m.digest([messages,seed])[:12]+'  '
        g=Generation(text,10,12,.2,.1,100,'eos',str(self.identity['model'])+str(len(self.physical_calls)))
        self.physical_calls.append({**dataclasses.asdict(g),'status':'SUCCESS','seed':seed,'prompt_sha256':m.digest(messages)})
        return g
def payload():return {'id':'synthetic-only','question':'A fabricated question?',
    'context':{'title':[f'Document {i}' for i in range(10)],'sentences':[[f'Fabricated evidence {i}'] for i in range(10)]}}
class Tests(unittest.TestCase):
    def test_generator_cost_keeps_both_models(self):
        rs=[{'model':x,'generation':dataclasses.asdict(Generation('x',10,12,.2,.1,100,'eos',x)),
             'received_message_tokens':0} for x in ['qwen','mistral']]
        c=m._logical_cost(r for r in rs)
        self.assertEqual([c[x]['calls'] for x in ['qwen','mistral']],[1,1])
    def test_complete_matrix_cache_and_full_cost(self):
        calls=[];op=m.PathOperator(Backend('q'),Backend('m'),before_request=calls.append)
        r=op.run_matrix(payload(),[26111311,26111312],'U')
        self.assertEqual(r['complete_terminal_conditions'],18)
        self.assertEqual(sum(x['calls'] for x in r['logical_completed_node_cost_by_model'].values()),60)
        self.assertEqual(len(calls),32)
        self.assertEqual(sum(c['cache_reuse_count'] for c in r['conditions']),28)
        for c in r['conditions']:self.assertEqual(c['answer'],c['nodes']['M']['generation']['text'].strip())
    def test_information_permissions(self):
        r=m.PathOperator(Backend('q'),Backend('m')).run_matrix(payload(),[1,2],'V')
        for c in r['conditions']:
            for node,record in c['nodes'].items():
                data=json.loads(record['request']['messages'][1]['content'])
                if c['condition'] in m.GRAPH_CONDITIONS:
                    self.assertEqual(len(data['documents']),5 if node in ['U','V'] else 0)
                    self.assertEqual([x['source'] for x in data['evidence_notes']],
                        [x for x in m.NODES if [x,node] in c['edges']])
                else:self.assertEqual(len(data['documents']),10)
    def test_overflow_no_backend_and_no_retry(self):
        q=Backend('q');mm=Backend('m');op=m.PathOperator(q,mm,m.PathConfig(max_input_tokens=9))
        with self.assertRaises(m.PathExecutionError) as e:op.run_matrix(payload(),[1,2],'U')
        self.assertEqual(e.exception.partial_result['status'],'INPUT_REJECTED')
        self.assertEqual(len(q.physical_calls)+len(mm.physical_calls),0)
        with self.assertRaises(RuntimeError):op.run_matrix(payload(),[1,2],'U')
    def test_labels_and_partition(self):
        p=payload();parts=m.partition_document_indices(p)
        self.assertEqual(sorted(parts['U']+parts['V']),list(range(10)))
        self.assertEqual(len(parts['U']),5)
        p['answer']='forbidden'
        with self.assertRaises(ValueError):m.validate_payload(p)
    def test_exact_graph_changes(self):
        for affected in ['U','V']:
            self.assertEqual(set(m.G0_EDGES)-set(m.condition_edges('G1',affected)),{(affected,'R')})
            self.assertEqual(set(m.G0_EDGES)-set(m.condition_edges('G1_MINUS_EU',affected)),{(affected,'R'),('U','M')})
if __name__=='__main__':unittest.main()
