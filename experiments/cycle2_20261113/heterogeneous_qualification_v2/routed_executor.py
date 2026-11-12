"""Serial, node-routed adapter for the unchanged closed-input executor.

Engineering support only. Model selection is a frozen route, never an answer-
dependent policy. Wire normalization uses the sender tokenizer; incoming-token
accounting uses the receiving tokenizer. A route change invalidates the entire
execution cache conservatively. No hidden model loading or fallback is allowed.
"""
from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

from graph_credit.executor import Backend, Executor, GraphConfig, canonical, digest


class RoutedBackend:
    def __init__(self, backends: Mapping[str, Backend], worker_models: Sequence[str],
                 aggregator_model: str, route_log: str | Path | None = None):
        if len(worker_models) != 4 or not all(isinstance(x, str) for x in worker_models):
            raise ValueError('Exactly four explicit worker model names are required')
        names = set(worker_models) | {aggregator_model}
        if not isinstance(aggregator_model, str) or names != set(backends):
            raise ValueError('Backend names must exactly match the frozen route')
        self.backends = dict(backends)
        self.worker_models = tuple(worker_models)
        self.aggregator_model = aggregator_model
        self._selected = contextvars.ContextVar('mab2_routed_node', default=None)
        self._execution_lock = threading.Lock()
        self.route_log = Path(route_log) if route_log is not None else None
        self._pins = {name: digest(b.identity) for name, b in self.backends.items()}
        self.identity = {
            'backend': 'explicit_node_route_v1',
            'model_identities': {name: copy.deepcopy(b.identity) for name, b in self.backends.items()},
            'worker_models': list(self.worker_models), 'aggregator_model': self.aggregator_model,
            'sender_normalization': True, 'receiver_token_accounting': True,
            'route_change_cache_invalidation': 'whole_execution',
            'routing_adapter_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.identity_pin = digest(self.identity)
        self.route_events = []

    def model_for_node(self, node: str) -> str:
        if node == 'aggregate':
            return self.aggregator_model
        import re
        match = re.fullmatch(r'r([123]):w([0123])', node)
        if match is None:
            raise ValueError('Unknown execution node')
        return self.worker_models[int(match.group(2))]

    def select_node(self, node: str):
        self.model_for_node(node)
        self._selected.set(node)

    def clear_node(self):
        self._selected.set(None)

    def _current(self):
        node = self._selected.get()
        if node is None:
            raise RuntimeError('Node routing requires an active execution node')
        name = self.model_for_node(node)
        backend = self.backends[name]
        if (tuple(self.identity['worker_models']) != self.worker_models
                or self.identity['aggregator_model'] != self.aggregator_model
                or set(self.backends) != set(self._pins)
                or digest(backend.identity) != self._pins[name]
                or digest(self.identity) != self.identity_pin):
            raise RuntimeError('Frozen model or route identity was modified')
        return node, name, backend

    def encode(self, text):
        return self._current()[2].encode(text)

    def decode(self, tokens):
        return self._current()[2].decode(tokens)

    def _record(self, row):
        self.route_events.append(row)
        if self.route_log is not None:
            self.route_log.parent.mkdir(parents=True, exist_ok=True)
            with self.route_log.open('a', encoding='utf-8') as handle:
                handle.write(canonical(row) + '\n')
                handle.flush()
                os.fsync(handle.fileno())

    def generate(self, messages, seed, max_new_tokens, config):
        node, name, backend = self._current()
        row = {'node': node, 'model': name, 'model_identity_sha256': self._pins[name],
               'seed': seed, 'max_new_tokens': max_new_tokens,
               'prompt_sha256': digest(messages), 'started_unix': time.time()}
        try:
            result = backend.generate(messages, seed, max_new_tokens, config)
        except Exception as error:
            self._record({**row, 'status': 'FAILED', 'error_type': type(error).__name__})
            raise
        self._record({**row, 'status': 'SUCCESS', 'call_id': result.call_id,
                      'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens})
        return result


class RoutedExecutor(Executor):
    def __init__(self, backend: RoutedBackend, config: GraphConfig | None = None):
        if not isinstance(backend, RoutedBackend):
            raise TypeError('RoutedExecutor requires a RoutedBackend')
        super().__init__(backend, config)

    def _messages(self, question, node, graph, outputs):
        self.backend.select_node(node)
        return super()._messages(question, node, graph, outputs)

    def execute(self, question, mask, seed, base=None):
        if not self.backend._execution_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent use of a routed backend is forbidden')
        try:
            return super().execute(question, mask, seed, base)
        finally:
            self.backend.clear_node()
            self.backend._execution_lock.release()
