"""Caller-side append/fsync observation; no model, RNG or cost-ledger mutation.

Journals contain private inputs and raw generations. Use a fresh private journal
per backend and attempt. Configure the native HFBackend with physical_log=None:
its in-memory physical_calls remain the sole native cost evidence. A process
killed before a RETURNED/ERROR row has an unresolved START, never zero cost.
"""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

SCHEMA='durable_backend_journal_v1'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


class DurableBackendError(RuntimeError):
    def __init__(self,phase,request_id=None,returned_generation=None,native_events=None,context=None,error_type=None):
        super().__init__('Durable backend failed at '+phase+'; execution must stop')
        self.phase=phase;self.request_id=request_id;self.returned_generation=returned_generation
        self.native_events=native_events;self.context=context;self.error_type=error_type


class DurableBackend:
    def __init__(self,underlying,model_name,journal_path,context_provider=None,before_request=None,
                 require_fresh_journal=True):
        if require_fresh_journal is not True:
            raise ValueError('Journal resume is unsupported; use a new attempt directory')
        self.underlying=underlying;self.model_name=model_name;self.journal_path=Path(journal_path)
        self.context_provider=context_provider;self.before_request=before_request
        self._lock=threading.Lock();self._poisoned=False
        try:
            header={'event':'HEADER','schema':SCHEMA,'model':model_name,'backend_identity':copy.deepcopy(self.identity),
                'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            data=(canonical(header)+'\n').encode('utf-8')
            with self.journal_path.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
        except Exception as error:
            raise DurableBackendError('HEADER',error_type=type(error).__name__) from None

    @property
    def identity(self):return self.underlying.identity

    @property
    def physical_calls(self):return self.underlying.physical_calls

    def encode(self,text):return self.underlying.encode(text)

    def decode(self,tokens):return self.underlying.decode(tokens)

    def __getattr__(self,name):return getattr(self.underlying,name)

    def _append(self,event):
        data=(canonical(event)+'\n').encode('utf-8')
        with self.journal_path.open('ab') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())

    def generate(self,messages,seed,max_new_tokens,config):
        if not self._lock.acquire(blocking=False):
            raise DurableBackendError('CONCURRENT_REQUEST')
        request_id=uuid.uuid4().hex;context=None;start=None;generation=None
        try:
            if self._poisoned:raise DurableBackendError('POISONED_AFTER_FAILURE')
            try:
                context=copy.deepcopy(self.context_provider() if self.context_provider else {})
                canonical(context)
                if self.before_request is not None:self.before_request(copy.deepcopy(context))
                start=len(self.physical_calls)
                event={'schema':SCHEMA,'request_id':request_id,'model':self.model_name,'context':context,
                    'messages':copy.deepcopy(messages),'seed':seed,'max_new_tokens':max_new_tokens,
                    'config':dataclasses.asdict(config) if dataclasses.is_dataclass(config) else copy.deepcopy(config),
                    'native_event_start':start,'backend_identity_sha256':hashlib.sha256(canonical(self.identity).encode()).hexdigest()}
                self._append({**event,'event':'START'})
            except Exception as error:
                self._poisoned=True
                raise DurableBackendError('START_OR_ADMISSION',request_id,context=context,error_type=type(error).__name__) from None
            try:
                generation=self.underlying.generate(messages,seed,max_new_tokens,config)
            except Exception as error:
                try:
                    rows=copy.deepcopy(self.physical_calls[start:])
                    self._append({**event,'event':'ERROR','error_type':type(error).__name__,
                                  'generation':None,'native_event_end':len(self.physical_calls),'native_events':rows})
                except Exception as write_error:
                    self._poisoned=True
                    raise DurableBackendError('ERROR_JOURNAL',request_id,native_events=locals().get('rows'),
                        context=context,error_type=type(write_error).__name__) from None
                raise
            try:
                rows=copy.deepcopy(self.physical_calls[start:])
                self._append({**event,'event':'RETURNED','generation':dataclasses.asdict(generation),
                              'native_event_end':len(self.physical_calls),'native_events':rows})
            except Exception as error:
                self._poisoned=True
                raise DurableBackendError('RETURNED_JOURNAL',request_id,generation,locals().get('rows'),context,
                                          type(error).__name__) from None
            return generation
        finally:self._lock.release()


def read_journal(path):
    """Strict reader: an incomplete trailing line is corruption, not a missing call."""
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('Incomplete durable journal')
    events=[json.loads(line) for line in raw.splitlines()]
    if events[0].get('event')!='HEADER' or events[0].get('schema')!=SCHEMA:
        raise ValueError('Invalid durable journal header')
    requests={}
    for event in events[1:]:
        request=event['request_id'];kind=event['event']
        if kind=='START':
            if request in requests:raise ValueError('Duplicate durable request')
            requests[request]={'start':event,'terminal':None}
        elif kind in ('RETURNED','ERROR'):
            if request not in requests or requests[request]['terminal'] is not None:
                raise ValueError('Unpaired durable terminal')
            start=requests[request]['start']
            if any(event.get(k)!=v for k,v in start.items() if k!='event'):
                raise ValueError('Durable request changed between events')
            requests[request]['terminal']=event
        else:raise ValueError('Unknown durable event')
    return {'header':events[0],'requests':requests,
            'pending_request_ids':[key for key,row in requests.items() if row['terminal'] is None]}


def extract_durable_error_evidence(error):
    """Find returned-but-not-durable raw values behind frozen operator wrappers.

    Caller may try an emergency PRIVATE checkpoint; persistent disk failure or
    SIGKILL can still prevent recovery. This creates no new cost events.
    """
    found=[];seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current,DurableBackendError):
            found.append({'phase':current.phase,'request_id':current.request_id,'context':current.context,
                'error_type':current.error_type,'generation':dataclasses.asdict(current.returned_generation)
                if current.returned_generation is not None else None,'native_events':current.native_events,
                'emergency_evidence_only':True,'new_physical_cost_event':False})
        current=current.__cause__
    return found
