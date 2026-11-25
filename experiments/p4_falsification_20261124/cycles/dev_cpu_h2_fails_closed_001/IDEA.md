# H2 fail-closed development diagnostic

This cycle copies the public P3 H2 implementation and changes only the two unresolved exits from target fallback to `action=None, status=UNRESOLVED`. It uses the previously exposed public generator and five exposed seeds. It is a diagnostic for the target fallback defect, not a new confirmation split.
