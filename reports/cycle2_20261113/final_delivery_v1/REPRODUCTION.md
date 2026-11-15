# Reproduction and artifact map

Repository: `fallenplum/MAB2`; branch: `research/cycle2-closure-revalidation-20261115`.

1. Frozen scientific design and input hashes: `protocol/cycle2_closure_revalidation_20261115/` and the Formal5 experiment README.
2. Implementation, portable CPU test bundle builder, raw scorer, policy auditor, and report renderer: `experiments/cycle2_20261113/closure_revalidation_v1/README.md`. Follow its exact commands; private replay requires the owner's hash-pinned archive and approvals. No private answer file is distributed in Git.
3. Public numerical reproduction: run `python paper/closure_study_v1/generate_results.py --repo .` and `python paper/closure_study_v1/generate_history.py --repo .`. These read aggregate evidence only and generate LaTeX tables; they make no model calls and compute no intervals.
4. Compile in a clean local build directory with the supplied `main.tex`, sections, generated tables, bibliography, official ACL style, and figures: `pdflatex main`, `bibtex main`, then `pdflatex main` twice. Use an installed LaTeX distribution. The verified build disables shell escape and automatic package installation.
5. Full paper evidence bindings and final numerical checks are in this delivery directory. Historical reports preserve their own frozen protocols; newer no-interval rules do not rewrite archived reports.

The generator and aggregate artifacts allow numerical/table reproduction. CPU request replay uses saved stochastic generations. Neither substitutes for independent GPU regeneration, human review, new-task confirmation, or access to private source questions. Environment versions and tokenizer/model identity are pinned in the experiment's operational artifacts. Server addresses and credentials are intentionally absent.

Full method-stage-model token/call/kernel costs: `../closure_formal5_results_v1/SUMMARY.json` and `EPISODE_SEED_AGGREGATES.csv`. GPU residence: `COST_LEDGER.json`. Do not add logical policy kernel charges across competing methods as if they were actual collection GPU consumption.
