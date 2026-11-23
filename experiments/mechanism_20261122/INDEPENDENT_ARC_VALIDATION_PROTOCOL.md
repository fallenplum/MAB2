# Independent ARC-Challenge validation protocol

The independent cohort is built from the ARC-Challenge validation split of
`allenai/ai2_arc`. The generator keeps the first 128 records with exactly four
choices and answer keys A--D or 1--4. It derives a SHA-256 question key from
the question and option texts, writes the answer map separately, and creates
one-, two- and four-row graph-edit variants by changing the designated relay
facts.

The 128-question cohort is split by the existing key residue rule into 73
calibration, 31 selection and 24 held-out test questions per density. The
Pairwise Gap quantile is selected on the first two stages. The test answer map
is read only after route selection and replay.

The generator does not contain model weights or private answer data. A local
cached Arrow/JSON export can be supplied with `--source` when the Hugging Face
endpoint is unavailable.

## Frozen input provenance

The server cache was produced from Hugging Face dataset revision
`210d026faf9955653af8916fad021475a3f00453` for `ai2_arc`, configuration
`ARC-Challenge`, split `validation`. The cached Arrow file is
`data-00000-of-00001.arrow`, 98,080 bytes, SHA-256
`4cf59b321066b627168d8a7b4c0ddeb5c29b5db6a8083056871baa943d5b0c90`, with
dataset fingerprint `5336412f86bd5bc2`. The exported cohort contains 128
records and 128 unique question keys. These identifiers fix the input used by
the independent worker and service replays without publishing answer labels.

