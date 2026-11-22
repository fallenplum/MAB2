from conditional_manifest import derive_changed_rows


def signatures(**updates):
    base = {str(i): {"prompt": f"p{i}", "model": "m", "upstream": [i]}
            for i in range(4)}
    base.update(updates)
    return base


def test_source_only_edit_keeps_rows_reusable():
    old = {"relay_signatures": signatures()}
    new = {"relay_signatures": signatures()}
    assert derive_changed_rows(old, new) == ()


def test_signature_change_marks_local_row():
    old = {"relay_signatures": signatures()}
    new = {"relay_signatures": signatures(**{"1": {"prompt": "changed"}})}
    assert derive_changed_rows(old, new) == (1,)


def test_graph_editor_mask_is_unioned():
    old = {"relay_signatures": signatures()}
    new = {"relay_signatures": signatures(), "graph_affected_rows": [2]}
    assert derive_changed_rows(old, new) == (2,)


def test_missing_metadata_falls_back_to_all_rows():
    assert derive_changed_rows({}, {"relay_signatures": signatures()}) == (0, 1, 2, 3)
