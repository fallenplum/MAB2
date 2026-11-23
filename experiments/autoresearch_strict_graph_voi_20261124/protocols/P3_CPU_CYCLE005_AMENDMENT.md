# P3 CPU cycle005 amendment

This amendment freezes the bounded-horizon-2 candidate before the confirmatory CPU run. The policy uses a two-step receding-horizon posterior rollout. At the rollout leaf it uses a declared deterministic completion heuristic. It may read only the declared source prior, current observations, true parent dependencies, legal cached transcript entries, query costs, and the strict weighted certificate. Target labels and unqueried outputs are forbidden for policy selection.

The gate is strict action exactness of 1.0 and at least 10 percent mean cost reduction against masked Pairwise on the five-seed matrix. This is a CPU candidate gate. It does not authorize GPU or NLP service execution.

