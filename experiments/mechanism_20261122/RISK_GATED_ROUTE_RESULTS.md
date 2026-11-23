# Risk-gated revalidation route

The route uses a development-selected quantile and answer-margin risk threshold.
The test residue is reported with the existing TCR and cold-restart traces.

| cell | q | threshold | risk action | risk decision | risk tokens | TCR action | TCR tokens | cold tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m60_conditional_dedup_commonsense | 0.5 | inf | 1.000 | 1.000 | 90.8 | 1.000 | 98.3 | 1349.8 |
| m60_conditional_dedup_qasc | 0.5 | inf | 1.000 | 1.000 | 99.6 | 1.000 | 182.2 | 1655.3 |
| m60_conditional_dedup_rows2_commonsense | 0.5 | inf | 1.000 | 1.000 | 181.5 | 1.000 | 207.3 | 1328.3 |
| m60_conditional_dedup_rows2_qasc | 0.5 | inf | 1.000 | 1.000 | 199.1 | 1.000 | 283.2 | 1623.1 |
| m60_conditional_dedup_rows2_sciq | 0.5 | inf | 1.000 | 1.000 | 179.1 | 1.000 | 256.9 | 1465.6 |
| m60_conditional_dedup_rows4_commonsense | 0.5 | 0 | 1.000 | 1.000 | 319.4 | 1.000 | 453.6 | 1317.1 |
| m60_conditional_dedup_rows4_qasc | 0.75 | 0 | 1.000 | 1.000 | 262.2 | 1.000 | 508.9 | 1592.3 |
| m60_conditional_dedup_rows4_sciq | 0.5 | 0 | 1.000 | 1.000 | 307.1 | 1.000 | 456.8 | 1287.7 |
| m60_conditional_dedup_sciq | 0.5 | inf | 1.000 | 1.000 | 89.6 | 1.000 | 163.0 | 1548.5 |
| m60_conditional_llama3_commonsense | 0.5 | inf | 1.000 | 1.000 | 90.8 | 1.000 | 98.3 | 1349.8 |
| m60_conditional_llama3_qasc | 0.5 | inf | 1.000 | 1.000 | 99.6 | 1.000 | 182.2 | 1655.3 |
| m60_conditional_llama3_rows2_commonsense | 0.5 | inf | 1.000 | 1.000 | 181.5 | 1.000 | 207.3 | 1328.3 |
| m60_conditional_llama3_rows2_qasc | 0.5 | inf | 1.000 | 1.000 | 199.1 | 1.000 | 283.2 | 1623.1 |
| m60_conditional_llama3_rows2_sciq | 0.5 | inf | 1.000 | 1.000 | 179.1 | 1.000 | 256.9 | 1465.6 |
| m60_conditional_llama3_rows4_commonsense | 0.5 | 0 | 1.000 | 1.000 | 319.4 | 1.000 | 453.6 | 1317.1 |
| m60_conditional_llama3_rows4_qasc | 0.75 | 0 | 1.000 | 1.000 | 262.2 | 1.000 | 508.9 | 1592.3 |
| m60_conditional_llama3_rows4_sciq | 0.5 | 0 | 1.000 | 1.000 | 307.1 | 1.000 | 456.8 | 1287.7 |
| m60_conditional_llama3_sciq | 0.5 | inf | 1.000 | 1.000 | 89.6 | 1.000 | 163.0 | 1548.5 |
| m60_conditional_server_commonsense | 0.9 | 0.5 | 1.000 | 1.000 | 343.5 | 0.914 | 98.6 | 1350.0 |
| m60_conditional_server_qasc | 0.95 | 1 | 1.000 | 1.000 | 232.2 | 0.891 | 182.8 | 1645.2 |
| m60_conditional_server_sciq | 0.5 | 0.25 | 1.000 | 1.000 | 487.0 | 0.867 | 160.5 | 1540.9 |

The risk-gated route reaches exact answer-action and edge-decision recovery on every listed local cell. On the three Qwen-14B server panels, TCR action exactness is 0.914, 0.891 and 0.867, while risk-gated recovery is exact with 343.5, 232.2 and 487.0 logical relay tokens. These panels are historical model-generated traces and still require single-forward service confirmation.

