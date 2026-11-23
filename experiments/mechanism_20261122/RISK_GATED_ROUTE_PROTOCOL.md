# Risk-gated credit revalidation protocol

## Runtime inputs

The controller receives the edited source distribution, cached old relay
outputs, the graph-descendant uncertainty mask, per-row drift radii calibrated
on development data, and relay query prices. It does not inspect new relay
values before selecting a route.

## Gate

Compute the cached post-edit answer margin between the largest and second
largest answer score. Divide the calibrated uncertainty mass of the uncertain
descendant rows by this margin. A threshold selected on development data sends
low-risk cases through descendant-mask revalidation. High-risk cases use the
answer-region controller. Missing interface fingerprints force the conservative
all-row path.

## Acceptance criteria

- Development selection requires exact answer-action recovery and exact edge
  recovery for the selected route.
- The independent residue reports answer-action exactness, edge-decision
  exactness, answer accuracy, relay tokens, relay forwards and wall time.
- A model-side forward mismatch invalidates service action comparisons until the
  worker reference and service calls use the same execution protocol.

## Current evidence

The gate reaches exact action and edge recovery on 21 local historical cells.
Qwen-14B historical panels show lower action recovery for TCR and exact
recovery for the risk-gated offline replay. Qwen-0.5B single-forward service
results show exact action recovery for both TCR and answer-region q=1, with no
reduction in relay forwards. Qwen-14B single-forward cost measurements are
available for three densities, while its action comparisons remain diagnostic
because the custom model path produces replay differences.

