"""CPU-only synthetic tests of durable boundaries and unchanged generation."""
import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from durable_backend import DurableBackend,DurableBackendError,read_journal

@dataclasses.dataclass
class Generation:
    text:str='private synthetic output'
    call_id:str='native-1'
    input_tokens:int=7
    output_tokens:int=3

@dataclasses.dataclass
class Config:
    temperature:float=.7

class Native:
    def __init__(self):
        self.identity={'model':'synthetic'};self.physical_calls=[];self.calls=0
        self.result=Generation();self.fail=None;self.received=None
    def encode(self,text):return list(text)
    def decode(self,tokens):return ''.join(tokens)
    def generate(self,messages,seed,max_new_tokens,config):
        self.calls+=1;self.received=(messages,seed,max_new_tokens,config)
        if self.fail:raise self.fail
        self.physical_calls.append({'call_id':'native-1','status':'SUCCESS','input_tokens':7,'output_tokens':3})
        return self.result

class DurableTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'raw.jsonl';self.native=Native()
        self.messages=[{'role':'user','content':'private synthetic prompt'}];self.config=Config()
    def tearDown(self):self.tmp.cleanup()
    def make(self,**kwargs):return DurableBackend(self.native,'qwen',self.path,**kwargs)
    def call(self,backend):return backend.generate(self.messages,17,128,self.config)
    def test_identity_arguments_result_rng_and_native_ledger_are_unchanged(self):
        backend=self.make(context_provider=lambda:{'unit':1});state=random.getstate()
        result=self.call(backend)
        self.assertIs(result,self.native.result);self.assertIs(backend.identity,self.native.identity)
        self.assertIs(backend.physical_calls,self.native.physical_calls)
        self.assertIs(self.native.received[0],self.messages);self.assertIs(self.native.received[3],self.config)
        self.assertEqual(self.native.received[1:3],(17,128));self.assertEqual(random.getstate(),state)
        self.assertEqual(backend.decode(backend.encode('abc')),'abc')
        parsed=read_journal(self.path);row=next(iter(parsed['requests'].values()))
        self.assertEqual(row['terminal']['generation'],dataclasses.asdict(result))
        self.assertEqual(row['terminal']['native_events'],self.native.physical_calls)
        self.assertEqual(len(self.native.physical_calls),1);self.assertEqual(parsed['pending_request_ids'],[])
    def test_start_is_fsynced_before_native_and_return_is_fsynced_before_return(self):
        backend=self.make();original=self.native.generate
        def generate(*args):
            journal=read_journal(self.path);self.assertEqual(len(journal['pending_request_ids']),1)
            return original(*args)
        self.native.generate=generate
        with patch('durable_backend.os.fsync',wraps=__import__('os').fsync) as sync:self.call(backend)
        self.assertEqual(sync.call_count,2)
    def test_start_write_failure_never_calls_native_and_poisons_wrapper(self):
        backend=self.make()
        with patch.object(backend,'_append',side_effect=OSError('synthetic disk')):
            with self.assertRaises(DurableBackendError):self.call(backend)
        with self.assertRaises(DurableBackendError):self.call(backend)
        self.assertEqual(self.native.calls,0)
    def test_return_write_failure_keeps_paid_generation_in_error_without_duplicate_fee(self):
        backend=self.make();append=backend._append
        def failing(row):
            if row['event']=='RETURNED':raise OSError('synthetic disk')
            append(row)
        with patch.object(backend,'_append',side_effect=failing):
            with self.assertRaises(DurableBackendError) as caught:self.call(backend)
        self.assertIs(caught.exception.returned_generation,self.native.result)
        self.assertEqual(caught.exception.native_events,self.native.physical_calls)
        self.assertEqual(len(self.native.physical_calls),1)
        self.assertEqual(len(read_journal(self.path)['pending_request_ids']),1)
    def test_original_preflight_error_is_preserved_when_error_journal_succeeds(self):
        backend=self.make();error=ValueError('Prompt has 8193 tokens; frozen cap is 8192');self.native.fail=error
        with self.assertRaises(ValueError) as caught:self.call(backend)
        self.assertIs(caught.exception,error)
        row=next(iter(read_journal(self.path)['requests'].values()))['terminal']
        self.assertIsNone(row['generation']);self.assertEqual(row['native_events'],[])
        self.assertEqual(row['event'],'ERROR')
    def test_error_write_failure_does_not_expose_overflow_as_cause(self):
        backend=self.make();append=backend._append;self.native.fail=ValueError('Prompt has 8193 tokens; frozen cap is 8192')
        def failing(row):
            if row['event']=='ERROR':raise OSError('synthetic disk')
            append(row)
        with patch.object(backend,'_append',side_effect=failing):
            with self.assertRaises(DurableBackendError) as caught:self.call(backend)
        self.assertIsNone(caught.exception.__cause__);self.assertIsNone(caught.exception.returned_generation)
    def test_callback_failure_stops_before_native_and_has_no_overflow_cause(self):
        def refuse(context):raise ValueError('Prompt has 8193 tokens; frozen cap is 8192')
        backend=self.make(before_request=refuse)
        with self.assertRaises(DurableBackendError) as caught:self.call(backend)
        self.assertEqual(self.native.calls,0);self.assertIsNone(caught.exception.__cause__)
    def test_same_call_id_native_multi_status_events_are_preserved_not_recounted(self):
        backend=self.make()
        def failing(*args):
            self.native.physical_calls.extend([{'call_id':'one','status':'SUCCESS','output_tokens':3},
                                               {'call_id':'one','status':'FAILED','output_tokens':None}])
            raise OSError('native pre-return failure')
        self.native.generate=failing
        with self.assertRaises(OSError):self.call(backend)
        terminal=next(iter(read_journal(self.path)['requests'].values()))['terminal']
        self.assertEqual(len(terminal['native_events']),2);self.assertIsNone(terminal['generation'])
        self.assertEqual(len(self.native.physical_calls),2)
    def test_fresh_journal_refuses_overwrite_and_reader_refuses_truncated_tail(self):
        backend=self.make();self.call(backend)
        with self.assertRaises(DurableBackendError):self.make()
        with self.path.open('ab') as stream:stream.write(b'{')
        with self.assertRaises(ValueError):read_journal(self.path)
    def test_pending_start_is_unknown_and_concurrency_is_rejected(self):
        backend=self.make();backend._lock.acquire()
        try:
            with self.assertRaises(DurableBackendError):self.call(backend)
        finally:backend._lock.release()
        backend._append({'event':'START','request_id':'kill','schema':'durable_backend_journal_v1'})
        self.assertEqual(read_journal(self.path)['pending_request_ids'],['kill'])
        self.assertEqual(self.native.calls,0)

if __name__=='__main__':unittest.main(verbosity=2)
