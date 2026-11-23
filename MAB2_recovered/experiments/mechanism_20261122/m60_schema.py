"""M60 worker output schema checks.

The seven probability rows are three source rows followed by four relay rows.
Relay rows are ordered [no advice, A, B, C]. The four source hypotheses A-D
are the probability columns in every row. Terminal rows are a 4x4 grid of
first/second researcher hypotheses and contain one option-label integer each.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SOURCE_ROWS = (0, 1, 2)
RELAY_ROWS = (3, 4, 5, 6)
RELAY_STATE_NAMES = ("none", "A", "B", "C")
TERMINAL_START = 7
N_HYPOTHESES = 4
N_LABELS = 8


def validate_full(full: Mapping[str, object]) -> dict[str, object]:
    keys = {str(k) for k in full}
    expected = {str(i) for i in range(23)}
    if keys != expected:
        raise ValueError(f"M60 full keys must be 0..22, got {sorted(keys)}")
    probabilities = []
    for index in range(7):
        row = np.asarray(full[str(index)], dtype=float)
        if row.shape != (N_HYPOTHESES,):
            raise ValueError(f"probability row {index} shape {row.shape}")
        if not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError(f"probability row {index} is invalid")
        if not np.isclose(float(row.sum()), 1.0, atol=1e-5):
            raise ValueError(f"probability row {index} does not sum to one")
        probabilities.append(row)
    labels = np.asarray([int(full[str(i)]) for i in range(TERMINAL_START, 23)])
    if labels.shape != (16,) or np.any(labels < 0) or np.any(labels >= N_LABELS):
        raise ValueError("terminal labels must contain 16 values in [0, 7]")
    return {
        "source_rows": probabilities[:3],
        "relay_rows": probabilities[3:],
        "relay_state_names": list(RELAY_STATE_NAMES),
        "terminal_labels": labels.reshape(N_HYPOTHESES, N_HYPOTHESES),
    }


def load_and_validate(row: Mapping[str, object]) -> dict[str, object]:
    return validate_full(row["full"])

