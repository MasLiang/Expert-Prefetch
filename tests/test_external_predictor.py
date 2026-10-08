"""External predictor trace validation tests."""

import pytest

from edge_moe_sim.predictor import ExternalTracePredictor
from edge_moe_sim.schema import ValidationError
from edge_moe_sim.synthetic import make_decode_event


def test_external_predictor_loads_unique_record(tmp_path, small_model) -> None:
    """A valid JSONL record is returned by its three-field key."""
    path = tmp_path / "predictor.jsonl"
    path.write_text(
        '{"request_id":0,"token_position":0,"layer_id":0,'
        '"candidate_ids":[1,2],"candidate_scores":[0.9,0.4],"ready_offset_cycles":3}\n',
        encoding="utf-8",
    )
    predictor = ExternalTracePredictor(path, small_model)
    output = predictor.predict(make_decode_event((0, 1)), 0)
    assert output is not None
    assert output.candidate_ids == (1, 2)
    assert output.ready_offset_cycles == 3


def test_external_predictor_rejects_duplicate_key(tmp_path, small_model) -> None:
    """Duplicate request/token/layer keys fail loudly."""
    line = (
        '{"request_id":0,"token_position":0,"layer_id":0,'
        '"candidate_ids":[1],"candidate_scores":[0.9],"ready_offset_cycles":0}\n'
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate predictor trace key"):
        ExternalTracePredictor(path, small_model)


def test_external_predictor_missing_record_is_explicit(tmp_path, small_model) -> None:
    """Missing keys error unless allow_missing selects NoPredictor fallback."""
    path = tmp_path / "predictor.jsonl"
    path.write_text(
        '{"request_id":9,"token_position":0,"layer_id":0,'
        '"candidate_ids":[1],"candidate_scores":[0.9],"ready_offset_cycles":0}\n',
        encoding="utf-8",
    )
    event = make_decode_event((0, 1))
    with pytest.raises(ValidationError, match="no record"):
        ExternalTracePredictor(path, small_model).predict(event, 0)
    assert ExternalTracePredictor(path, small_model, allow_missing=True).predict(event, 0) is None
