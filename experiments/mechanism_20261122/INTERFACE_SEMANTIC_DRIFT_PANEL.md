# Semantic relay-interface drift panel

This panel changes the relay interface while keeping the question text and
source distribution fixed. The scorer compares each controller with the full
new-graph decision set. It reports audit-set exact recovery and logical relay
token cost. It does not report user answer accuracy.

The nine completed cells cover QASC, SciQ and CommonsenseQA, three model
scales, and edit densities rows1, rows2 and rows4. Each cell contains 128
questions. The cells were selected to cover all three task families and model
scales under the same semantic interface change.

| cell | stale exact | warm exact | TCR exact | TCR tokens | cold exact | cold tokens |
|---|---:|---:|---:|---:|---:|---:|
| Llama-1B Commonsense rows4 | .7422 | .7422 | 1.0000 | 606.55 | 1.0000 | 1220.11 |
| Llama-1B QASC rows1 | .9531 | .9609 | 1.0000 | 185.57 | 1.0000 | 1297.84 |
| Llama-1B SciQ rows1 | .9219 | .9766 | 1.0000 | 221.74 | 1.0000 | 1606.13 |
| Qwen-0.5B Commonsense rows1 | .9766 | .9766 | 1.0000 | 144.46 | 1.0000 | 1382.95 |
| Qwen-0.5B QASC rows2 | .8203 | .8281 | 1.0000 | 334.34 | 1.0000 | 1645.83 |
| Qwen-0.5B SciQ rows2 | .9609 | .9766 | 1.0000 | 333.25 | 1.0000 | 1526.95 |
| Qwen-14B Commonsense rows2 | .9766 | .9766 | 1.0000 | 211.73 | 1.0000 | 744.61 |
| Qwen-14B QASC rows4 | .9688 | .9688 | 1.0000 | 403.55 | 1.0000 | 554.39 |
| Qwen-14B SciQ rows4 | .9531 | .9297 | 1.0000 | 388.30 | 1.0000 | 845.71 |

The completed semantic drift cells support the need to revalidate affected
relay rows. They do not isolate a new answer-region algorithm. The strongest
claim supported by this panel is that stale or warm transport can lose the
finite decision set under a changed relay interface, while TCR guard recovers
the set on the evaluated cells.

