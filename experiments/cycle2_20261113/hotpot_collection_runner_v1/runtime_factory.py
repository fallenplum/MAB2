"""Deferred production imports. Merely importing this file never loads models."""
import importlib
import importlib.util
import dataclasses
import sys
import time
from pathlib import Path
from orchestration_core import digest,require
from storage import ensure_disk,file_sha,write_new


def build_runtime(plan,worker_id,output,context,before_request):
    runtime=plan['runtime'];minimum=plan['budget']['disk_min_free_bytes']
    for directory in reversed(runtime['python_import_paths']):sys.path.insert(0,directory)
    for row in plan['sources']:require(file_sha(row['path'])==row['sha256'],'Source changed before model imports')
    backend_module=importlib.import_module('graph_credit.backend')
    matrix_module=importlib.import_module('matrix_executor')
    durable_module=importlib.import_module('durable_backend')
    for name,module in [('backend',backend_module),('matrix',matrix_module),('durable',durable_module)]:
        spec=runtime['modules'][name]
        require(Path(module.__file__).resolve()==Path(spec['path']).resolve() and file_sha(module.__file__)==spec['sha256'],'Unexpected imported '+name+' module')
    graph_config=matrix_module.GraphConfig(rounds=3)
    require(digest(dataclasses.asdict(graph_config))==digest(runtime['graph_config']),
            'JSON configuration differs from unchanged qualified three-round default')
    collector_spec=runtime['collector']
    require(file_sha(collector_spec['path'])==collector_spec['sha256'],'Collector source changed')
    spec=importlib.util.spec_from_file_location('collection_frozen_collector',collector_spec['path'])
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    matrix_ref=[None];backends={};loads=[]
    def request_context():
        return {**context,'node':None if matrix_ref[0] is None else matrix_ref[0].router._selected.get()}
    for model in ('qwen','mistral'):
        ensure_disk(output,minimum);before=time.monotonic()
        write_new(output/('MODEL_LOAD_STARTED_'+model+'.json'),{'model':model,'worker':worker_id,'monotonic':before})
        native=backend_module.HFBackend(runtime['models'][model]['path'],'cuda:0',physical_log=None)
        require(digest(native.identity)==digest(runtime['models'][model]['identity']),'Loaded model/runtime identity differs')
        backends[model]=durable_module.DurableBackend(native,model,output/(model+'_raw_requests.jsonl'),
                  context_provider=request_context,before_request=before_request)
        elapsed=time.monotonic()-before
        loads.append({'model':model,'load_wall_seconds':elapsed})
        write_new(output/('MODEL_LOAD_COMPLETE_'+model+'.json'),{'model':model,'worker':worker_id,'wall_seconds':elapsed,
                   'identity':native.identity,'runtime_private':native.runtime})
    matrix=matrix_module.HotpotApplicabilityMatrixExecutor(backends,config=graph_config,route_log=None);matrix_ref[0]=matrix
    require(matrix.protocol_hash==runtime['expected_matrix_protocol_hash'],'Matrix protocol differs from engineering qualification')
    write_new(output/'RUNTIME_IDENTITY.json',{'matrix_protocol_hash':matrix.protocol_hash,'loads':loads,
              'durable_helper_sha256':file_sha(durable_module.__file__),'model_source_binding_checked':True})
    return {'matrix':matrix,'collector':collector.collector_payload,'backends':backends,
            'extract_error':durable_module.extract_durable_error_evidence,'loads':loads}
