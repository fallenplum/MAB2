# Risk gate versus fair answer-action baseline

The fair answer-action baseline refreshes the graph-descendant relay rows and
stops after scoring the final answer. It does not purchase downstream cells
needed for a full edge certificate. This baseline is the correct comparator for
answer-region action maintenance.

Across the 21 local cells, risk-gated action exactness is 1.0 everywhere. In
the deduplicated Qwen-0.5B/Llama-1B rows4 cells, risk-gated relay tokens are
319.4, 262.2 and 307.1 versus fair-mask costs of 357.0, 392.3 and 352.3.
Rows1 and rows2 generally select the mask route and have equal cost.

The historical Qwen-14B panels expose the quality--cost tradeoff. Fair-mask
answer-action exactness is 0.875, 0.859 and 0.783 for CommonsenseQA, QASC and
SciQ. The risk gate reaches 1.0 at 343.5, 232.2 and 487.0 logical relay
tokens. The higher cost is the price of recovering actions that a mask-only
answer route misses.

On matched single-forward Qwen-14B worker outputs, the risk gate saves 82.6%
versus the fair mask at rows1 and has equal cost at rows2/rows4. Its larger
87.1%, 25.1% and 18.7% reductions are relative to the full edge-audit TCR,
which purchases downstream conditional cells and serves a different target.
The Qwen-0.5B service benchmark keeps action exactness at 1.0 for both routes
and shows equal actual relay-call cost.

