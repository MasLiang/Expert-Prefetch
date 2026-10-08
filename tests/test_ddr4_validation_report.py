"""DDR 仿真报告的验收门：缺失 PASS/校准/合法传输不能被报告成成功。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def reporting_module() -> Any:
    """加载独立 FPGA 验证脚本，无需启动 Vivado。"""
    path = Path(__file__).resolve().parents[1] / "hardware/m10_u200/scripts/run_ddr4_validation.py"
    spec = importlib.util.spec_from_file_location("ddr4_validation_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(tmp_path: Path) -> Path:
    """生成仅供 parser 单元测试的日志；不作为实际 DDR 验证证据。"""
    case = tmp_path / "hit"
    sim = case / "xsim"
    sim.mkdir(parents=True)
    lines = [f"DDR_CALIBRATED bank={b} cycle=100" for b in range(4)]
    lines += [f"DDR_INITIALIZED bank={b} cycle=200" for b in range(4)]
    for layer, value in enumerate((24, 60, 254)):
        lines += [f"LAYER_START layer={layer} cycle={210+layer*100}",
                  f"LAYER_RESULT layer={layer} cycle={290+layer*100} y={value}"]
    lines += [f"DDR_DRAM_SUMMARY bank={b} reads=1 activates=1 refreshes=0 precharges=0" for b in range(4)]
    lines += ["PASS: real 3-layer MoE chain", "$finish called"]
    (sim / "simulate.log").write_text("\n".join(lines))
    (sim / "compile.log").write_text("compiled")
    (sim / "elaborate.log").write_text("elaborated")
    (sim / "tb_u200_moe.sh").write_text('xelab -generic_top "BUDGET=0"')
    for b in range(4):
        (sim / f"ddr_bank_{b}.csv").write_text(
            "bank,address_bytes,beats,ar_cycle,first_r_cycle,last_r_cycle\n"
            f"{b},16384,4,220,260,263\n")
    return case


def test_ddr_report_excludes_initialization(tmp_path: Path) -> None:
    """报告以开始/最终结果的差值计时，不包括 DDR warm-up。"""
    result = reporting_module().parse_case(fake_run(tmp_path))
    assert result["total_cycles"] == 280
    assert result["transferred_bytes"] == 256
    assert result["first_latency_cycles"] == {"min": 40, "max": 40, "mean": 40.0}


@pytest.mark.parametrize("marker", ["PASS: real 3-layer MoE chain", "$finish called", "DDR_CALIBRATED bank=3 cycle=100"])
def test_ddr_report_rejects_incomplete_run(tmp_path: Path, marker: str) -> None:
    """即使工具退出码为零，也必须有全部功能/校准证据。"""
    case = fake_run(tmp_path)
    log = case / "xsim/simulate.log"
    log.write_text(log.read_text().replace(marker, ""))
    with pytest.raises(ValueError):
        reporting_module().parse_case(case)


def test_ddr_report_rejects_preload_samples(tmp_path: Path) -> None:
    """预加载测量不能混进 workload 服务统计。"""
    case = fake_run(tmp_path)
    sample = case / "xsim/ddr_bank_0.csv"
    sample.write_text(sample.read_text().replace("220,260,263", "150,190,193"))
    with pytest.raises(ValueError, match="Initialization"):
        reporting_module().parse_case(case)


def test_ddr_report_rejects_fatal_even_with_pass(tmp_path: Path) -> None:
    """禁止因已有 PASS 字符串而忽略 fatal。"""
    case = fake_run(tmp_path)
    log = case / "xsim/simulate.log"
    log.write_text(log.read_text() + "\nFatal: DDR payload mismatch\n")
    with pytest.raises(ValueError, match="Simulation error"):
        reporting_module().parse_case(case)


def test_ddr_report_rejects_default_current_prefetch(tmp_path: Path) -> None:
    """缺少 BUDGET=0 的基线配置不能被误标为跨层对照实验。"""
    case = fake_run(tmp_path)
    (case / "xsim/tb_u200_moe.sh").write_text("xelab -generic_top LAYERS=3")
    with pytest.raises(ValueError, match="Current-layer"):
        reporting_module().parse_case(case)


def test_ddr_report_rejects_wrong_baseline_bytes(tmp_path: Path) -> None:
    """纯基线不能出现少传/多传 payload 后仍报告正确。"""
    case = fake_run(tmp_path)
    baseline = case.with_name("no_prefetch")
    case.rename(baseline)
    with pytest.raises(ValueError, match="exactly three layers"):
        reporting_module().parse_case(baseline)
