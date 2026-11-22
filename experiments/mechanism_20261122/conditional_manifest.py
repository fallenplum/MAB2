"""Conservative relay-row localization from frozen graph metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_changed_rows(old_manifest: Mapping, new_manifest: Mapping,
                        row_count: int = 4) -> tuple[int, ...]:
    """Return a conservative row mask without labels or a fresh audit table.

    Missing or malformed metadata returns every relay row. This permits extra
    revalidation but never silently skips a row whose interface is unknown.
    """
    all_rows = tuple(range(row_count))
    try:
        old_signatures = old_manifest["relay_signatures"]
        new_signatures = new_manifest["relay_signatures"]
        if not isinstance(old_signatures, Mapping) or not isinstance(new_signatures, Mapping):
            return all_rows
        changed = set()
        for index in all_rows:
            key = str(index)
            if key not in old_signatures or key not in new_signatures:
                return all_rows
            if canonical_hash(old_signatures[key]) != canonical_hash(new_signatures[key]):
                changed.add(index)
        for value in new_manifest.get("graph_affected_rows", ()):
            index = int(value)
            if index < 0 or index >= row_count:
                return all_rows
            changed.add(index)
        return tuple(sorted(changed))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return all_rows
