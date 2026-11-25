# P2 production path audit

## Finding

The frozen `experiments/p2_multihop_20261124/p2_service_runner.py::run_row`
cannot be passed directly to the P4 online broker without changing its
information boundary. The runner has the following fixed execution order.

1. It requires a `gold` argument and checks the gold map before calling
   `run_row`.
2. Inside `run_row`, it computes `new_cache` and `target` before constructing
   or running any online method.
3. It implements each method as a local branch over `Ledger` calls and has no
   `Policy.next_action(PublicState)` or broker callback interface.
4. Its RAG branch is a fixed prompt with a hard-coded `retrieval_calls = 2`
   counter, so it is not a production retriever adapter.

Passing this function directly to the P4 broker would therefore expose the
post-edit target before the policy acts and would invalidate T05/T14. It would
also make the online policy and the frozen runner use different control paths.

## Current admissible integration

`p4_p2_service_adapter.py` imports only the frozen runner's `encode`, `prompts`
and `summary` functions. It executes the actual Qwen forward path through an
independent hook backend and routes post-edit queries through
`p4_online_runtime.OnlineBroker`. The full new graph is audited only after the
online stop. This is the minimal adapter that preserves the frozen renderer
and tokenizer semantics while keeping the P4 public-state boundary.

## Gate consequence

The adapter passes bounded development checks on one exposed row, including
the second-parent invalidation and repeat semantic checks. It does not close
T10/T15 or G0. A clean production integration requires a separately frozen
runner entry point that accepts a broker and does not construct `target` or
load `gold` before online stopping. That entry point is not present in the
current frozen runner and should be registered as a new protocol artifact,
not silently substituted under the old runner hash.
