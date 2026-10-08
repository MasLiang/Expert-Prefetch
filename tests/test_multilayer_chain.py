"""Verify the sequential RTL layer-chain contract and requested register style."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "hardware/m10_u200"


def test_chain_numeric_retirement_and_abort(tmp_path: Path) -> None:
    """Run signed saturation, accepted config, drain delay and abort/restart checks."""
    executable = tmp_path / "chain"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_moe_layer_chain", "-o", str(executable),
         str(ROOT / "rtl/moe_layer_chain.sv"), str(ROOT / "tb/tb_moe_layer_chain.sv")],
        check=True, capture_output=True, text=True, timeout=30,
    )
    result = subprocess.run(["vvp", str(executable)], check=True, capture_output=True,
                            text=True, timeout=30)
    assert result.stdout.count("PASS:") == 1
    assert "FATAL" not in result.stdout


@pytest.mark.parametrize("filename", ["moe_layer_chain.sv", "u200_moe_top.sv",
                                     "cross_layer_prefetch.sv", "layer_weight_barrier.sv",
                                     "multilayer_tile_demander.sv"])
def test_chain_register_style(filename: str) -> None:
    """Layer context and activation each retain one registered field per always."""
    source = (ROOT / "rtl" / filename).read_text()
    for block in re.split(r"\balways_ff\b", source)[1:]:
        registers = set(re.findall(r"\b([A-Za-z_]\w*)(?:\[[^\n;]*?\])?\s*<=", block))
        assert len(registers) == 1, registers


def test_retirement_uses_real_resources() -> None:
    """The context barrier includes FIFO/adapter/descriptors, not just result_valid."""
    dma = (ROOT / "rtl/burst_dma_channel.sv").read_text()
    core = (ROOT / "rtl/tile_prefetch_dataflow_top.sv").read_text()
    adapter = (ROOT / "rtl/tile_prefetch_axi_top.sv").read_text()
    assert "assign drained" in dma
    assert "demand_count == 0" in dma and "spec_count == 0" in dma
    assert "32'(table_free_entries) == DESCRIPTOR_TABLE_ENTRIES" in core
    assert "core_memory_drained && !adapter_busy" in adapter


def test_cross_layer_resource_release_credit_and_abort(tmp_path: Path) -> None:
    """Execute barrier-free future release, credit and abort without model input."""
    executable = tmp_path / "cross_layer"
    files = [ROOT / "rtl" / name for name in (
        "package.sv", "hardware_greedy_v1.sv", "candidate_prefetch_scheduler.sv",
        "cross_layer_prefetch.sv", "layer_weight_barrier.sv")]
    files.append(ROOT / "tb/tb_cross_layer_prefetch.sv")
    subprocess.run(["iverilog", "-g2012", "-s", "tb_cross_layer_prefetch", "-o",
                    str(executable), *map(str, files)], check=True, capture_output=True,
                   text=True, timeout=30)
    result = subprocess.run(["vvp", str(executable)], check=True, capture_output=True,
                            text=True, timeout=30)
    assert result.stdout.count("PASS:") == 1
    assert "FATAL" not in result.stdout


def test_no_previous_weight_gate_in_rtl() -> None:
    """Completion telemetry must never gate the cross-layer allocator."""
    source = (ROOT / "rtl/cross_layer_prefetch.sv").read_text()
    assert "weights_done" not in source
    assert "received[layer]==ctx.count && layer>int'(current_layer)" in source
