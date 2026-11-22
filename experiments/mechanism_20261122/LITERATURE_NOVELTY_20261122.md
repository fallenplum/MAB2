# Literature Tree and Novelty Matrix for CTDR-M

**Search date**: 2026-11-22  
**Scope**: papers on decision-region active testing, LLM multi-agent communication efficiency, graph-conditioned adaptation, and certificate reuse.  
**Sources queried**: web search with primary publisher pages, ACL Anthology, PMLR, and the AISTATS author-hosted proceedings PDF. The review is a targeted scoping review, not an exhaustive systematic review.

## Literature tree

### A. Decision-region active testing

- **Javdani et al. 2014, Near Optimal Bayesian Active Learning for Decision Making, AISTATS 17.** The paper formalizes tests, overlapping decision regions, and stopping when all consistent hypotheses belong to one decision region. Its Hyperedge Cutting objective is a generic active-testing baseline with adaptive-submodularity guarantees. CTDR-M inherits the decision-region framing and must not present decision-region stopping or cost-aware test choice as new by themselves. Source: https://publications.icl.cmu.edu/storage/publications/pub_files/2014/4/javdani14hec_extended.pdf
- The relevant transfer gap is that CTDR-M conditions certificate reuse on a graph edit and a relay-interface signature. The prior active-testing framework does not model reuse of a previously paid communication transcript after graph mutation.

### B. Sparse and adaptive multi-agent communication

- **Li et al. 2024, Improving Multi-Agent Debate with Sparse Communication Topology, Findings of EMNLP 2024.** Sparse communication topologies can preserve or improve debate quality while reducing computation. This is topology selection for a debate run, rather than post-edit certificate maintenance. Source: https://aclanthology.org/2024.findings-emnlp.427/
- **Zhang et al. 2025, Cut the Crap: An Economical Communication Pipeline for LLM-Based Multi-Agent Systems, ICLR 2025.** AgentPrune defines communication redundancy and prunes spatial-temporal message-passing graphs, reporting large token and cost reductions. It performs one-shot pruning, whereas CTDR-M decides whether old credit evidence remains valid after a graph edit and buys targeted revalidation. Source: https://proceedings.iclr.cc/paper_files/paper/2025/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf
- **Wang et al. 2025, AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration, ACL 2025.** AgentDropout dynamically removes agents or communication participants to reduce token use. The method operates on participation and interaction optimization, not formal finite-audit certificates or transport of historical relay observations. Source: https://aclanthology.org/2025.acl-long.1170.pdf
- **You et al. 2026, MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based Multi-Agent Communication, EMNLP 2026.** MessageSieve prunes message instances at runtime with local state cards and comparisons. It explicitly distinguishes message-level pruning from agent-, edge-, and topology-level decisions. The closest systems baseline for runtime communication efficiency, but it does not address graph edits, old transcript reweighting, or validity certificates. Source: https://aclanthology.org/2026.emnlp-main.2179/
- **AnyMAC, Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction, EMNLP 2025.** AnyMAC predicts the next agent in a flexible cascade and studies input-conditioned communication structures. It optimizes execution routing for a task, not revalidation of previously paid communication evidence after graph changes. Source: https://aclanthology.org/2025.emnlp-main.584.pdf

### C. Graph shift and graph-conditioned adaptation

- **Qin et al. 2022, Graph Neural Architecture Search Under Distribution Shifts, ICML 2022.** GRACES customizes graph architectures per instance under graph distribution shift. This establishes graph-shift adaptation as a recognized problem, but it does not provide communication-credit certificates or active transcript repair. Source: https://proceedings.mlr.press/v162/qin22b.html
- The CTDR-M gap is narrower and operational. A graph edit changes which relay interfaces may be affected. A safe method must infer affected descendants from a frozen graph manifest and fall back conservatively when signatures are missing.

### D. Certificate sharing and robustness

- **Shared Certificates for Neural Network Verification, CAV 2022.** Shared certificates reuse proof effort across multiple inputs. This is a useful conceptual neighbor for reusing paid evidence, but its verification objects and threat model differ from communication-credit audits. Source: https://github.com/eth-sri/proof-sharing
- **Gosch et al. 2026, Certifying Graph Neural Networks Against Label and Structure Poisoning, ICML 2026.** This line studies certificates under graph perturbation and poisoning. It addresses robustness of a predictor under graph changes, while CTDR-M addresses validity and cost of reusing communication observations after an edit. Source: https://proceedings.mlr.press/v306/gosch26a.html

## Novelty matrix

| Work or family | Graph edit after paid transcript | Reuses old communication observations | Decision-region stopping | Interface or signature based affected-row mask | Budgeted targeted revalidation | Task-level quality and token cost |
|---|---:|---:|---:|---:|---:|---:|
| Javdani et al. HEC | No | No | Yes | No | Yes | Application-dependent |
| Sparse Debate Topology | Topology chosen before run | No | No | No | No | Yes |
| AgentPrune | Graph pruned during run | No | No | No | Indirect | Yes |
| AgentDropout | Dynamic participant removal | No | No | No | Indirect | Yes |
| MessageSieve | Fixed graph, runtime message pruning | No | No | No | Local message policy | Yes |
| AnyMAC | Input-conditioned cascade | No | No | No | Route prediction | Yes |
| GRACES | Graph distribution shift | Model adaptation | No | No | No | Graph classification |
| Shared proof certificates | Perturbation-dependent proofs | Proof reuse | No | Certificate identity | Verification scheduling | Verification cost |
| **CTDR-M target** | **Yes** | **Yes** | **Yes, finite audit** | **Yes** | **Yes** | **Required in confirmation** |

## Novelty boundary

The following components are established ideas and cannot carry the main novelty claim alone.

1. Decision-region stopping.
2. Source-distribution or total-variation bounds.
3. Maximum-uncertainty or generic value-of-information scheduling.
4. Sparse communication topology selection.
5. Runtime message pruning.
6. Generic certificate or proof reuse.

The defensible candidate contribution is the combination of a **graph-edit descendant mask**, **relay-interface fingerprints**, and **fail-closed certificate transport**. The method must prove that graph metadata can identify rows that require revalidation, that missing metadata causes conservative fallback, and that the resulting audit decisions remain exact under conditional-channel drift.

## Evidence requirements implied by the tree

- Compare against warm restart, reweighted audit, no-radius replay, maximum uncertainty, random, periodic, and cold restart under the same old transcript and incremental cost accounting.
- Include source-only edits, conditional-channel drift, formatting or prompt-signature drift, and degraded-manifest controls.
- Report finite-audit exact-set recovery and action error separately from normal task accuracy.
- Add an independent task family and multiple checkpoint families before making a broad ACL Main claim.
- Treat the current two-checkpoint, three-task density audit as mechanism development evidence until an independent graph-manifest panel and task-level service-quality evaluation are complete.

## Search limitations

The query was targeted and primary-source oriented. It did not enumerate all papers in active testing, formal verification, or multi-agent communication. EMNLP 2026 and ACL 2025 records were checked through ACL Anthology pages or proceedings PDFs. Search results from aggregators were used only to discover candidate titles, then replaced by primary records where available.


