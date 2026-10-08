"""多seed加密扫描的去重、配对汇总和采样段语义。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


@pytest.fixture
def dense(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """加载扫描驱动，不执行完整矩阵。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "path", [str(root / "examples"), *sys.path])
    spec = importlib.util.spec_from_file_location("dense_experiment", root / "examples/prefetch_dense_sweep.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, yaml.safe_load((root / "configs/experiments/prefetch_dense.yaml").read_text())


def test_point_count_and_effective_depth(dense: tuple[Any, dict[str, Any]]) -> None:
    """局部消融共享baseline；超过Expert深度的配置不重复运行。"""
    module, config = dense
    points = module.build_points(config)
    assert len(points) == 264
    assert sum(map(len, points.values())) == 922
    for (groups, pe, ratio, capacity), policies in points.items():
        assert len(policies) == len(set(policies))
        assert ("no_prediction", 0., 0) in policies
        assert ("oracle_all_gu", 1., groups) in policies
        assert all(depth <= groups for _, _, depth in policies)


def test_repeated_seeds_rejected(dense: tuple[Any, dict[str, Any]]) -> None:
    """重复seed不能伪装成独立重复实验。"""
    module, config = dense
    with pytest.raises(ValueError, match="distinct seeds"):
        module.build_points({**config, "seeds": [1, 1]})


def test_seed_aggregation(dense: tuple[Any, dict[str, Any]]) -> None:
    """同时报告平均配对速度比和合并cycles比，避免混淆二者。"""
    module, config = dense
    rows = []
    for seed, baseline, current in [(1, 100, 50), (2, 300, 300)]:
        rows.append(dict(tile_groups=4, pe_count=1, ratio=2., ocm_expert_equivalents=4.,
            policy="uniform_prefix", quality=.9, prefix_depth=1, seed=seed,
            speedup=baseline/current, baseline_cycles=baseline, total_cycles=current,
            net_saved_cycles=baseline-current, first_start_saved_cycles=0,
            future_speculation_blocking_cycles=0, useful_speculative_bytes=0,
            wasted_speculative_bytes=0, extra_transfer_bytes=0))
    result = module.aggregate_seeds(rows, 2, [1.05])[0]
    assert result["speedup_mean"] == 1.5
    assert result["pooled_speedup"] == 400/350
    assert result["speedup_min"] == 1.
    assert result["seeds_ge_1.05"] == 1
    with pytest.raises(ValueError, match="duplicate seed"):
        module.aggregate_seeds([rows[0], rows[0]], 2, [1.05])


def test_regions_do_not_bridge_failed_point(dense: tuple[Any, dict[str, Any]]) -> None:
    """通过的两侧采样点不能跨过失败谷值合并成一个保证区间。"""
    module, config = dense
    tiny = {**config, "tile_groups": [4], "pe_counts": [1], "ocm_expert_equivalents": [4],
            "compute_to_memory_ratios": [.5, 1., 2.], "thresholds": [1.05]}
    rows = [dict(tile_groups=4, pe_count=1, ratio=r, ocm_expert_equivalents=4,
        policy="uniform_prefix", quality=.9, prefix_depth=1, speedup_min=v)
        for r,v in [(.5,1.1),(1.,.9),(2.,1.1)]]
    result = module.sampled_regions(rows, tiny)[0]
    assert result["all_seeds_passing_points"] == 2
    assert result["consecutive_passing_sample_segments"] == '[[0.5], [2.0]]'


@pytest.mark.parametrize("ratio", [.9, 1.])
def test_diagnostic_trace_equivalence(dense: tuple[Any, dict[str, Any]], ratio: float) -> None:
    """开启timeline的诊断重建必须与扫描makespan一致。"""
    module, config = dense
    import prefetch_dense_diagnostics as diagnostics
    settings = yaml.safe_load((module.ROOT / config["base_config"]).read_text())
    record = module.make_inputs(settings)[0]
    point = dict(tile_groups=4,pe_count=1,bytes_per_cycle_per_channel=4*ratio,
        ocm_expert_equivalents=4,policy="uniform_prefix",quality=.9,prefix_depth=1)
    result = diagnostics.diagnose(settings, record, point)
    assert result["timeline"]
    assert len(result["layers"]) == settings["layers"]
    if ratio == 1.:
        first = result["blocking_records"][0]
        assert first["block_start_cycle"] == 2721
        assert first["blocked_cycles"] == 1031
