"""Pure decision logic for P2 DAG aggregation."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def terminal_answer(source: np.ndarray, relay: Sequence[np.ndarray], labels: Mapping[tuple[int, int], int]) -> int:
    """Aggregate terminal labels with the DAG weight P(S0=i)*P(R_i=j)."""
    scores = np.zeros(8, dtype=float)
    for (i, j), label in labels.items():
        scores[int(label)] += float(source[i]) * float(relay[i][j])
    return int(np.argmax(scores))
