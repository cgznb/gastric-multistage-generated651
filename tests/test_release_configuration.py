import pytest

from stageworld.regularization_spec import specification


def test_selected_protocol_limits_both_training_stages(monkeypatch):
    monkeypatch.setenv("GENERATED651_ARM", "bs4_baseline")
    spec = specification()
    assert spec["batch_sizes"] == [4]
    assert len(spec["arms"]) == 1
    assert spec["arms"][0]["name"] == "bs4_baseline"
    assert spec["paired_baseline"] == "bs4_baseline"
    assert len(spec["seeds"]) == 20
    assert spec["phase_epochs"] == 400


def test_full_study_remains_available(monkeypatch):
    monkeypatch.setenv("GENERATED651_ARM", "all")
    assert len(specification()["arms"]) == 15
    monkeypatch.setenv("GENERATED651_ARM", "unknown")
    with pytest.raises(ValueError, match="Unknown"):
        specification()
