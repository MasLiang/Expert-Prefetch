"""历史收益复测的激励嵌套、分母、容量、去重和实际模型检查。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """读取真实保存激励，不执行完整矩阵。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    spec = importlib.util.spec_from_file_location("benefit_retest", root / "examples/prefetch_benefit_retest.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((root / "configs/experiments/prefetch_benefit_retest.yaml").read_text())
    base = yaml.safe_load((root / config["base_config"]).read_text())
    traces = json.loads((root / config["historical_directory"] / "inputs.json").read_text())
    return module, config, base, traces["20261005"][0]


def test_nested_truth_and_unique_candidates(experiment: tuple[Any, ...]) -> None:
    """K增加只扩展truth；候选合法且hit/miss决策不变。"""
    module, _, base, record = experiment
    expanded = [module.extend_record(record, 20261005, k, 16) for k in (1, 2, 4)]
    for layer in range(base["layers"]):
        assert expanded[2]["truth"][layer][:2] == expanded[1]["truth"][layer]
        assert expanded[1]["truth"][layer][:1] == expanded[0]["truth"][layer]
    for k, trace in zip((1, 2, 4), expanded):
        for target in trace["targets"]:
            ids = module.prediction_ids(trace, target, .9, 20)
            assert len(ids) == len(set(ids)) == 2 * k
            assert all(0 <= e < 16 for e in ids)
            truth = set(trace["truth"][target["target"]])
            assert len(truth.intersection(ids)) == (k if target["quality_order"] < 18 else 0)


def test_top1_candidates_match_history(experiment: tuple[Any, ...]) -> None:
    """扩展驱动在K=1时保留原始候选及rank。"""
    module, _, _, record = experiment
    extended = module.extend_record(record, 20261005, 1, 16)
    for before, after in zip(record["targets"], extended["targets"]):
        assert module.prediction_ids(extended, after, .9, 20) == module.candidates(record, before, .9, 20)


def test_fixed_capacity_and_no_duplicate_baseline(experiment: tuple[Any, ...]) -> None:
    """容量两条曲线明确区分；每硬件点baseline唯一。"""
    module, config, base, _ = experiment
    points = module.build_points(config, base)
    assert len(points) == 186
    fixed = [p for p in points if p["capacity_mode"] == "fixed_bytes"]
    assert {p["ocm_bytes"] for p in fixed} == {196608}
    identities = [(p["stage"], p["tile_groups"], p["ratio"], p["top_k"], p["capacity_mode"], p["policy"], p["prefix_depth"]) for p in points]
    assert len(set(identities)) == len(points)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_actual_worker_deterministic_and_conservation(experiment: tuple[Any, ...], k: int) -> None:
    """实际运行主模型，检查单token多K前进性/容量/字节与确定性。"""
    module, config, base, record = experiment
    point = next(p for p in module.build_points(config, base) if p["stage"] == "growth" and p["top_k"] == k
        and p["tile_groups"] == 2 and p["ratio"] == .9 and p["policy"] == "uniform_prefix")
    result = module.worker((base, point, 20261005, record))
    assert result == module.worker((base, point, 20261005, record))
    assert result["peak_ocm_bytes"] <= result["ocm_bytes"]
    assert result["speculative_bytes"] == result["useful_speculative_bytes"] + result["wasted_speculative_bytes"]


def test_top1_growth_matches_historical_worker(experiment: tuple[Any, ...]) -> None:
    """K=1原始执行函数与新扩展执行构造完全相同，防止重建配置漂移。"""
    module, config, base, record = experiment
    points = module.build_points(config, base)
    old = next(p for p in points if p["stage"] == "historical" and p["tile_groups"] == 2
               and p["ratio"] == .9 and p["policy"] == "uniform_prefix")
    new = next(p for p in points if p["stage"] == "growth" and p["tile_groups"] == 2
        and p["ratio"] == .9 and p["policy"] == "uniform_prefix" and p["top_k"] == 1
        and p["capacity_mode"] == "scaled_expert")
    a, b = module.worker((base, old, 20261005, record)), module.worker((base, new, 20261005, record))
    for key in ("total_cycles", "demand_bytes", "speculative_bytes", "peak_ocm_bytes", "first_start_sum_cycles"):
        assert a[key] == b[key]


def test_pairing_rejects_missing_samples(experiment: tuple[Any, ...]) -> None:
    """分母不能静默使用不匹配样本。"""
    module, config, base, record = experiment
    points = [p for p in module.build_points(config, base) if p["stage"] == "historical"
              and p["tile_groups"] == 2 and p["ratio"] == .9]
    raw = [module.worker((base, p, 20261005, record)) for p in points]
    assert module.aggregate(raw)[1]["pooled_speedup"] > 1
    with pytest.raises(ValueError, match="paired samples"):
        module.aggregate([raw[0], {**raw[1], "sample": 99}])
