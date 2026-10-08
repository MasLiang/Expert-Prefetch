"""Control-only optimization must preserve compute service and register style."""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_handoff_service_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling timestamps may change; executed tile set and MAC duration may not."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    module = importlib.import_module("validate_gu_early_start")
    row = dict(expert_id=3,tile_id=0,gu_start_cycle=10,gu_done_cycle=266,
               down_start_cycle=300,down_done_cycle=428)
    original: dict[str, Any] = dict(tile_records=[row])
    shifted = {**row,"gu_start_cycle":9,"gu_done_cycle":265,
               "down_start_cycle":298,"down_done_cycle":426}
    module.verify_compute_services(original,dict(tile_records=[shifted]))
    with pytest.raises(ValueError,match="compute service"):
        module.verify_compute_services(original,dict(tile_records=[{**shifted,"gu_done_cycle":264}]))
    with pytest.raises(ValueError,match="compute service"):
        module.verify_compute_services(original,dict(tile_records=[]))
    with pytest.raises(ValueError,match="duplicate"):
        module.verify_compute_services(original,dict(tile_records=[shifted,shifted]))


def test_compute_controller_one_register_per_always() -> None:
    """The rewritten controller follows the requested single-register block style."""
    source = (ROOT/"hardware/m10_u200/rtl/tile_compute_controller.sv").read_text()
    blocks = re.split(r"\balways_ff\b",source)[1:]
    assert blocks
    for block in blocks:
        registers = set(re.findall(r"\b([A-Za-z_]\w*)\s*<=",block))
        assert len(registers) == 1, registers


def test_stage_release_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject missing, repeated, premature or delayed GU resource retirement."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    module = importlib.import_module("validate_gu_early_start")
    timing = dict(tile_records=[dict(expert_id=3,tile_id=0,gu_done_cycle=10,down_done_cycle=30)])
    gu = dict(event="GU_SLOT_RELEASE",expert_id=3,tile_id=0,cycle=11)
    down = dict(event="DOWN_SLOT_RELEASE",expert_id=3,tile_id=0,cycle=31)
    assert module.verify_stage_releases([gu,down],timing) == dict(gu_release_count=1,down_release_count=1)
    with pytest.raises(ValueError,match="tile set"):
        module.verify_stage_releases([gu],timing)
    with pytest.raises(ValueError,match="duplicate"):
        module.verify_stage_releases([gu,gu,down],timing)
    for cycle in (9,30,31):
        with pytest.raises(ValueError,match="causality"):
            module.verify_stage_releases([{**gu,"cycle":cycle},down],timing)
    with pytest.raises(ValueError,match="causality"):
        module.verify_stage_releases([gu,{**down,"cycle":29}],timing)


def test_execution_scheduler_one_register_per_always() -> None:
    """Lookahead uses one registered field per block, with no multi-variable always_comb."""
    source = (ROOT/"hardware/m10_u200/rtl/tile_execution_scheduler.sv").read_text()
    assert "always_comb" not in source
    blocks = re.split(r"\balways_ff\b",source)[1:]
    assert blocks
    for block in blocks:
        registers = set(re.findall(r"\b([A-Za-z_]\w*)(?:\[[^\n;]*?\])?\s*<=",block))
        assert len(registers) == 1, registers


def test_mac_backend_one_register_per_always() -> None:
    """Chunk pipeline and independent reset blocks retain single-field ownership."""
    source = (ROOT/"hardware/m10_u200/rtl/int8_swiglu_tile_backend.sv").read_text()
    assert "always_comb" not in source
    for block in re.split(r"\balways_ff\b",source)[1:]:
        registers = set(re.findall(r"\b([A-Za-z_]\w*)(?:\[[^\n;]*?\])?\s*<=",block))
        assert len(registers) == 1, registers


def test_chunk_pipeline_service_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compute optimization must satisfy a formula, not bypass duration checks."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    module = importlib.import_module("validate_gu_early_start")
    row = dict(expert_id=3,tile_id=0,gu_start_cycle=10,gu_done_cycle=50,
               down_start_cycle=70,down_done_cycle=94)
    timing = dict(tile_records=[row])
    module.verify_pipeline_services(timing,timing,16,16)
    for stage in ("gu_done_cycle","down_done_cycle"):
        with pytest.raises(ValueError,match="service mismatch"):
            module.verify_pipeline_services(timing,dict(tile_records=[{**row,stage:row[stage]-1}]),16,16)
    with pytest.raises(ValueError,match="tile set"):
        module.verify_pipeline_services(timing,dict(tile_records=[]),16,16)
    with pytest.raises(ValueError,match="tile set"):
        module.verify_pipeline_services(timing,dict(tile_records=[row,row]),16,16)
