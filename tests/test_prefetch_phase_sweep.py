"""合成收益区域扫描的激励公平性、PE映射与确定性验证。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def experiment() -> tuple[Any, dict[str, Any]]:
    """加载独立实验驱动与配置，不执行完整扫描。"""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase_experiment", root / "examples/prefetch_phase_sweep.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, yaml.safe_load((root / "configs/experiments/prefetch_phase.yaml").read_text())


def test_nested_quality_and_oracle(experiment: tuple[Any, dict[str, Any]]) -> None:
    """质量等级嵌套且精确；全部硬件/策略共享truth。"""
    module, settings = experiment
    records = module.make_inputs(settings)
    assert records == module.make_inputs(settings)
    decisions = settings["samples"] * (settings["layers"] - 1)
    previous: set[tuple[int, int]] = set()
    for quality in settings["qualities"]:
        hits = set()
        for record in records:
            for target in record["targets"]:
                ids = module.candidates(record, target, quality, decisions)
                true = record["truth"][target["target"]]
                assert len(set(ids)) == 2
                assert module.candidates(record, target, quality, decisions, True) == (true,)
                if true in ids:
                    assert ids[0] == true
                    hits.add((record["sample"], target["target"]))
        assert len(hits) == int(quality * decisions + 0.5)
        assert previous <= hits
        previous = hits


@pytest.mark.parametrize("pe", [1, 4, 16])
@pytest.mark.parametrize("policy", ["no_prediction", "uniform_prefix", "oracle_all_gu"])
def test_deterministic_capacity_and_byte_accounting(
        experiment: tuple[Any, dict[str, Any]], pe: int, policy: str) -> None:
    """PE全分配给当前GEMM，低容量下仍能完成并守恒。"""
    module, settings = experiment
    record = module.make_inputs(settings)[0]
    point = dict(tile_groups=4, pe_count=pe, bytes_per_cycle_per_channel=4,
                 ocm_expert_equivalents=1, policy=policy, quality=.9, prefix_depth=4)
    row = module.simulate_job((settings, point, record))
    assert row == module.simulate_job((settings, point, record))
    assert row["peak_ocm_bytes"] <= row["ocm_bytes"]
    assert row["speculative_bytes"] == row["useful_speculative_bytes"] + row["wasted_speculative_bytes"]
    assert row["compute_to_memory_payload_ratio"] == 1 / pe
    if policy == "no_prediction":
        assert row["speculative_bytes"] == 0
    if policy == "oracle_all_gu":
        assert row["wasted_speculative_bytes"] == 0
