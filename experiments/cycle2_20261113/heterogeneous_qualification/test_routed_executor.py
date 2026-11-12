import dataclasses
import json
import unittest

from graph_credit.executor import Executor, Generation, GraphConfig, affected_closure
from routed_executor import RoutedBackend, RoutedExecutor


class ToyBackend:
    def __init__(self, name, repeat=1):
        self.identity = {'model': name, 'tokenizer_repeat': repeat}
        self.repeat = repeat
        self.calls = []
        self.fail = False

    def encode(self, text):
        return [ord(c) for c in text for _ in range(self.repeat)]

    def decode(self, tokens):
        return ''.join(chr(tokens[i]) for i in range(0, len(tokens), self.repeat))

    def generate(self, messages, seed, max_new_tokens, config):
        if self.fail:
            raise RuntimeError('Synthetic backend failure')
        self.calls.append((messages, seed))
        text = json.dumps({'answer': '2', 'evidence': self.identity['model'], 'uncertainty': ''})
        return Generation(text, len(self.encode(json.dumps(messages))), len(self.encode(text)), 0)


def make(rounds=2, order=('a','b','a','b')):
    a,b=ToyBackend('a'),ToyBackend('b',2)
    router=RoutedBackend({'a':a,'b':b},order,'a')
    config=GraphConfig(rounds=rounds,message_tokens=256,aggregate_tokens=256,max_input_tokens=20000)
    return RoutedExecutor(router,config),a,b


class RoutedExecutorTest(unittest.TestCase):
    def setUp(self):
        self.question={'id':'synthetic_only','question':'What is one plus one?'}

    def test_model_dispatch_and_receiver_tokenizer(self):
        exe,a,b=make()
        graph=(1,)*12
        result=exe.execute(self.question,graph,99173)
        self.assertEqual((len(a.calls),len(b.calls)),(5,4))
        self.assertEqual(len(exe.backend.route_events),9)
        for node,record in result.nodes.items():
            receiving=exe.backend.backends[exe.backend.model_for_node(node)]
            if node=='aggregate':
                sources=[result.nodes[f'r2:w{i}'].wire_message for i in range(4)]
            elif node.startswith('r2:'):
                w=int(node[-1]);sources=[result.nodes[f'r1:w{i}'].wire_message for i in range(4) if i!=w]
            else:sources=[]
            self.assertEqual(record.received_message_tokens,sum(len(receiving.encode(s)) for s in sources))
            self.assertEqual(record.message_metadata['message_tokens'],len(receiving.encode(record.wire_message)))
        with self.assertRaises(RuntimeError):exe.backend.encode('outside execution')

    def test_cache_replay_three_rounds(self):
        exe,a,b=make(rounds=3)
        old=(1,)*12;new=(0,)+old[1:]
        base=exe.execute(self.question,old,99173)
        before=len(a.calls)+len(b.calls)
        changed=exe.execute(self.question,new,99173,base)
        closure=affected_closure(old,new,3)
        self.assertEqual(len(a.calls)+len(b.calls)-before,len(closure))
        self.assertEqual({n for n,r in changed.nodes.items() if not r.reused},closure)
        fresh=exe.execute(self.question,new,99173)
        for n in fresh.nodes:self.assertEqual(fresh.nodes[n].wire_message,changed.nodes[n].wire_message)

    def test_route_and_model_change_invalidate_cache(self):
        first,a,b=make();base=first.execute(self.question,(1,)*12,99173)
        second,c,d=make(order=('b','a','b','a'))
        self.assertNotEqual(first.protocol_hash,second.protocol_hash)
        result=second.execute(self.question,(1,)*12,99173,base)
        self.assertEqual(sum(r.reused for r in result.nodes.values()),0)
        self.assertEqual(len(c.calls)+len(d.calls),9)
        second.backend.backends['a'].identity['model']='changed'
        with self.assertRaises(RuntimeError):second.execute(self.question,(1,)*12,99173)

    def test_homogeneous_adapter_preserves_prompt_and_wire(self):
        b=ToyBackend('a');router=RoutedBackend({'a':b},('a',)*4,'a')
        config=GraphConfig(message_tokens=256,aggregate_tokens=256,max_input_tokens=20000)
        routed=RoutedExecutor(router,config).execute(self.question,(1,)*12,99173)
        plain=Executor(ToyBackend('a'),config).execute(self.question,(1,)*12,99173)
        for node in plain.nodes:
            self.assertEqual(plain.nodes[node].messages,routed.nodes[node].messages)
            self.assertEqual(plain.nodes[node].wire_message,routed.nodes[node].wire_message)
            self.assertEqual(plain.nodes[node].received_message_tokens,routed.nodes[node].received_message_tokens)

    def test_failure_does_not_fallback_or_leave_active_node(self):
        exe,a,b=make();b.fail=True
        with self.assertRaises(RuntimeError):exe.execute(self.question,(1,)*12,99173)
        self.assertEqual((len(a.calls),len(b.calls)),(1,0))
        self.assertEqual(exe.backend.route_events[-1]['status'],'FAILED')
        with self.assertRaises(RuntimeError):exe.backend.encode('x')
        b.fail=False
        self.assertEqual(exe.execute(self.question,(0,)*12,99173).answer,'2')

    def test_route_validation_and_labels(self):
        with self.assertRaises(ValueError):RoutedBackend({'a':ToyBackend('a')},('a',)*3,'a')
        with self.assertRaises(ValueError):RoutedBackend({'a':ToyBackend('a')},('a',)*4,'b')
        exe,_,_=make()
        for node in ('r0:w0','r4:w0','r1:w4','x'):
            with self.assertRaises(ValueError):exe.backend.select_node(node)
        with self.assertRaises(ValueError):exe.execute({**self.question,'answer':'2'},(1,)*12,1)
        self.assertEqual(len(exe.backend.route_events),0)

    def test_concurrent_router_use_rejected(self):
        exe,_,_=make();exe.backend._execution_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):exe.execute(self.question,(1,)*12,1)
        finally:exe.backend._execution_lock.release()

    def test_route_mutation_after_freeze_rejected(self):
        exe,_,_=make()
        exe.backend.worker_models=('b','b','a','a')
        with self.assertRaises(RuntimeError):exe.execute(self.question,(1,)*12,1)
        self.assertEqual(len(exe.backend.route_events),0)


if __name__=='__main__':unittest.main()
