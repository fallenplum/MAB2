# D18 negative: realized-message embeddings do not establish useful decisions

Both fixed message predictors fail the primary screen and risk targeting on both tasks. No new generation or GPU cost. CPU feature extraction27.36s and model fitting/evaluation9.08s; independent scalar cost/choice audit and two held-out-fold refits passed1922 checks. All192 question groups remain development data.

|Task|Fixed policy U|Question preselected U|Message Ridge U|Message ExtraTrees U|Matched single U|
|---|---:|---:|---:|---:|---:|
|Hotpot|.636202|.600560|.576157|.598359|.599417|
|2Wiki|.595661|.606432|.599839|.601036|.567813|

Message-conditioned policies pay all observed prefixes; static/preselected alternatives pay only generated prefixes. The top-quarter low-margin risk enrichment is below1 for both message predictors on both tasks. Thus the frozen uncertainty proxy does not target larger decision regret. Do not replace this negative outcome with a larger network or tune the same folds until positive.

D17's oracle headroom remains real within its limited diagnostic, but D18 has not converted it into a cross-question predictor. D19 next tests actual gold-free counterfactual consistency on the same exposed records, with every probe charged. This is a baseline/foundation diagnostic, not a new-method claim. Separate fresh confirmation and complete acquisition comparisons remain required.
