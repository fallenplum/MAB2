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
`2ff366e955bc5ab4ba79631db53f5c801f7a61c6001fc816dfda22f3371d10ec`.
The matching server-built PDF is six pages, 140218 bytes, SHA-256
`01d0b20ba790ee68159db430d2dd71c3198985650162cbd6d6195ea916fdac21`.
The PDF was rendered locally and pages 1--6 were visually checked after the
M60 table and theory additions.

## Reproduction status

The CPU clean-replay fixture runs its two unit tests and covers all nine
controller labels. Mechanism Python files pass `compileall`. The server build
uses the vendored ACL style and pdfTeX. Local MiKTeX and offline Tectonic are
environment-limited in this Windows sandbox, so the server build is the
authoritative compilation check for this release.

