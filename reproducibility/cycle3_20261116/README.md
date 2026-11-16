# Exact Formal7 execution sources

The source ZIP contains only29 source files and a SHA256 manifest. All22 execution pins from the frozen retry plan match the bundled bytes. It contains no question inputs, answer references, model weights, runtime outputs, credentials or host addresses. The ZIP is90,914bytes; its SHA256 is inFORMAL7_SOURCE_BUNDLE_RECEIPT.json.

Some historical source files useCRLF while newer ones useLF. Git text normalization can change raw-byte pins without changing Python semantics. Extract the ZIP unchanged when reproducing a frozen plan; verify every member againstSOURCE_SHA256.json before use. Do not edit a live run or alter old pins to conceal line-ending mismatches.

Exact source integrity is not experimental reproduction. A complete rerun also requires the privately retained question/selection manifests, source descriptor, closed plan, model identities and weights, seeds, environment and explicit bounded output path. Those private owner artifacts remain under the experiment root. Reconstruct plan paths for the new machine explicitly and retain old/new manifests; do not publish private questions or credentials. The model pool and fixed graph/operator/gate definitions are described inprotocol/cycle3_20261116/FORMAL7_EVIDENCE_PRESERVING_ANNOTATIONS.md.

Analysis sources include the frozen evaluator, independent score and execution checks, label-release helper, and existing Hotpot answer metric. They must run only after complete archive and actual-cost closure. The archive's analysis folder is not a claim of a single-command public-data reproduction. Formal7 is still running; no success claim follows from this source package.


Formal8 second-task transfer also has formal8_all_attempts_sources.zip and its own manifest/receipt. It preserves31 source files including both residency variants and the frozen answer-alias adapters. All23 execution pins from the first attempt and all current-retry pins match. No input, answer, alias table, output, model weight, host address, credential or private plan is included. Git normalization may change loose-file byte hashes; use the source ZIP when replaying a frozen plan. Both real collections remain incomplete at this source-package checkpoint; CRC/SHA checks establish file integrity, not a successful method or reproduced experiment.
