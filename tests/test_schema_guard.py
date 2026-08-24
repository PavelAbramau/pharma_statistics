import pytest

from pharma_stats.history import schema_guard as sg


def _history_list(changes=None, **extra_top):
    base = {
        "changes": changes if changes is not None else [],
        "lastUpdateVersions": {},
        "originalData": {},
        "outcomesUpdateCount": 0,
    }
    base.update(extra_top)
    return base


def _entry(**overrides):
    e = {
        "version": 0, "date": "2020-01-01", "status": "RECRUITING",
        "studyType": "INTERVENTIONAL", "moduleLabels": [],
        "lastUpdateSubmitQcDate": "2020-01-01",
    }
    e.update(overrides)
    return e


def test_history_list_accepts_valid_shape():
    obj = _history_list(changes=[_entry(), _entry(version=1, moduleLabels=["Study Status"])])
    h = sg.check_history_list(obj, nct_id="NCT001")
    assert isinstance(h, str) and len(h) == 16


def test_history_list_tolerates_extra_top_level_keys():
    obj = _history_list(changes=[_entry()], someNewField=42)
    sg.check_history_list(obj, nct_id="NCT001")  # should not raise


def test_history_list_tolerates_variable_shaped_sibling_fields():
    obj = _history_list(
        changes=[_entry()],
        lastUpdateVersions={"outcomes": 3, "enrollmentInfo": 1},
        originalData={"enrollmentCountSame": True, "leadSponsorSame": False},
    )
    sg.check_history_list(obj, nct_id="NCT001")  # should not raise despite variable keys


def test_history_list_rejects_missing_top_level_key():
    obj = _history_list(changes=[_entry()])
    del obj["originalData"]
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_list(obj, nct_id="NCT001")


def test_history_list_rejects_missing_entry_key():
    entry = _entry()
    del entry["moduleLabels"]
    obj = _history_list(changes=[entry])
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_list(obj, nct_id="NCT001")


def test_history_list_rejects_non_int_version():
    obj = _history_list(changes=[_entry(version="0")])
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_list(obj, nct_id="NCT001")


def test_history_list_rejects_non_list_changes():
    obj = _history_list()
    obj["changes"] = {"not": "a list"}
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_list(obj, nct_id="NCT001")


def test_history_version_accepts_valid_shape():
    obj = {"studyVersion": 3, "study": {"protocolSection": {"identificationModule": {}}}}
    h = sg.check_history_version(obj, nct_id="NCT001", version=3)
    assert isinstance(h, str) and len(h) == 16


def test_history_version_tolerates_extra_study_keys():
    obj = {
        "studyVersion": 3,
        "study": {"protocolSection": {}, "hasResults": True, "derivedSection": {}},
    }
    sg.check_history_version(obj, nct_id="NCT001", version=3)  # should not raise


def test_history_version_rejects_missing_protocol_section():
    obj = {"studyVersion": 3, "study": {"hasResults": True}}
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_version(obj, nct_id="NCT001", version=3)


def test_history_version_rejects_non_int_study_version():
    obj = {"studyVersion": "3", "study": {"protocolSection": {}}}
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_version(obj, nct_id="NCT001", version=3)


def test_history_version_rejects_missing_study():
    obj = {"studyVersion": 3}
    with pytest.raises(sg.SchemaGuardError):
        sg.check_history_version(obj, nct_id="NCT001", version=3)


def test_hash_stable_across_variable_sibling_field_content():
    """The whole point of the targeted-signature design: two responses
    that differ only in lastUpdateVersions/originalData content (normal
    per-trial variation) must hash identically."""
    obj_a = _history_list(
        changes=[_entry()], lastUpdateVersions={"outcomes": 1}, originalData={"x": True},
    )
    obj_b = _history_list(
        changes=[_entry()], lastUpdateVersions={"enrollmentInfo": 9}, originalData={"y": 1, "z": 2},
    )
    assert sg.check_history_list(obj_a, nct_id="A") == sg.check_history_list(obj_b, nct_id="B")
