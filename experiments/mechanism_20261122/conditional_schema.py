from __future__ import annotations

from typing import Mapping

import numpy as np


def validate_conditional_full(full: Mapping[str, object]) -> dict[str, object]:
    expected = {str(i) for i in range(27)}
    if {str(k) for k in full} != expected:
        raise ValueError("conditional worker expects 27 fields")
    probs = []
    for i in range(11):
        row = np.asarray(full[str(i)], dtype=float)
        if row.shape != (4,) or not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError(f"invalid probability row {i}")
        if not np.isclose(float(row.sum()), 1.0, atol=1e-5):
            raise ValueError(f"probability row {i} is not normalized")
        probs.append(row)
    labels = np.asarray([int(full[str(i)]) for i in range(11, 27)], dtype=int)
    if labels.shape != (16,) or np.any(labels < 0) or np.any(labels >= 8):
        raise ValueError("invalid terminal labels")
    return {"source": np.asarray(probs[:3]), "old_relay": np.asarray(probs[3:7]),
            "new_relay": np.asarray(probs[7:11]), "labels": labels.reshape(4, 4)}
