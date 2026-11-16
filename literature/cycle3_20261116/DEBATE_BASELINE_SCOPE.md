# Strong noncommunication controls and limits of transfer

Scoped primary-source reading during blinded Formal7/Formal8 collection. It changes neither running operator nor gate. Metadata and reading depth are recorded in DEBATE_BASELINE_PRIMARY_RECORDS.json; no author-code reproduction or exhaustive survey is claimed.

| Related line | Required interpretation for MAB2 |
|---|---|
| [Smit et al., ICML2024](https://proceedings.mlr.press/v235/smit24a.html) | Retain independent sampling/aggregation controls; beating a weak communication graph is insufficient. Protocol tuning can change outcomes, so a negative result is conditional. |
| [Choi et al., v2](https://arxiv.org/html/2508.17536v2) | The DCM martingale result has explicit neighbor-belief conditions. Our heterogeneous, evidence-asymmetric edge intervention does not automatically meet them. |
| [Zhu et al., v3](https://arxiv.org/html/2601.19921v3) | Diversity pooling and trained confidence are existing mechanisms with extra costs. Renaming either as credit maintenance would not establish novelty. |
| [Bertalanič and Fortuna, v1](https://arxiv.org/abs/2605.00914v1) | Existing work already studies communication-induced degradation. A generic claim that communication can hurt is not an adequate new contribution. Only abstract-level conclusions are used here. |

The unresolved MAB2 question remains whether graph changes invalidate useful decision information, and whether paying to refresh that information improves a future service decision more than free structural rules, random/periodic rechecks and strong fixed controls. Evidence of variable credit, or an unavailable perfect-information oracle, is insufficient.

Advantages: the same independent/vote controls already frozen in the two pilots address part of this evaluation concern. Risks: the current menu may have too little decision headroom; practical confidence can require extra training; a generic negative communication study overlaps heavily with prior work. These are limitations to test, not new empirical findings. Task-specific pilot failures cannot be rescued by pooling tasks or substituting retrospective winners.
