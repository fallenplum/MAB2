# Fingerprint-enriched ARC replay

The ARC worker traces were postprocessed with tokenized old and new relay
interface fingerprints. The Pairwise route resolved the revalidation mask from
these fingerprints rather than from legacy `changed_rows` metadata.

| density | action exact | Pairwise tokens | fair fingerprint mask tokens |
|---:|---:|---:|---:|
| rows1 | 1.000 | 130.04 | 151.38 |
| rows2 | 1.000 | 273.33 | 302.75 |
| rows4 | 1.000 | 530.54 | 599.50 |

The fingerprint-enriched replay preserves the independent ARC action result.
The original ARC service trace used the same worker outputs before this
metadata enrichment. The enrichment is a worker-level metadata robustness
check and does not change the service measurements.

