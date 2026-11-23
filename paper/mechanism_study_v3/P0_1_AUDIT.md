# P0-1 provenance and source/PDF audit

## Placeholder scan

The branch tree was scanned through the GitHub Git Data API and 93 small text
blobs were inspected. Forty-eight files contained PowerShell `Get-Content` /
`Cannot find path` error output instead of their intended artifact. The
affected paths were all under `experiments/mechanism_20261122`. They were
replaced with the corresponding local artifact bytes. One placeholder had no
valid local counterpart, `M60_INTERFACE_DRIFT_PANEL.md`, and was removed from
the branch tree. Historical research files were left untouched.

The replacement files now retain their real source sizes and parse as their
declared formats. The scan found no remaining PowerShell error placeholder in
the repaired small-blob set.

## Source/PDF pair

The current source is `main.tex`, SHA-256
`1a1a8420b84ef4e5a46a1b27a48a093bbd3210392168f186b1c3ff716c979623`.
The matching server-built PDF is six pages, 140227 bytes, SHA-256
`d93703d902072d0542d40f9b6d293aba7d1baa2b3a03914177be4497aaef4cb5`.
The PDF was rendered locally and pages 1--6 were visually checked after the
M60 table and theory additions.

## Reproduction status

A fresh shallow checkout of release commit `da1360fbbbedb29f57098fffacafe38f2ca8780d` was verified on the server with `PYTHONPATH=.`. Mechanism `compileall`, `test_none_semantics.py` (2/2), `test_p0_permissions_oracle.py` (4/4), and `repro/tests/test_clean_replay.py` (2/2) all passed. The explicit `prepare_inputs.py` -> `run_replay.py` fixture pipeline also completed. The server build uses the vendored ACL style and pdfTeX. Local MiKTeX and offline Tectonic are environment-limited in this Windows sandbox, so the server build is the authoritative compilation check for this release.

