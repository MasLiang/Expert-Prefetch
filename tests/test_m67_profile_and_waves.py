"""M6.7 physical-profile and distinct-wave accounting checks."""
from __future__ import annotations
from pathlib import Path
from edge_moe_sim.config import load_hardware_profile

ROOT=Path(__file__).resolve().parents[1]
def test_vek280_theoretical_mapping_and_ridge_are_physical() -> None:
    synthetic=load_hardware_profile(ROOT/'configs/hardware_profile_synthetic_balanced.yaml')
    vek=load_hardware_profile(ROOT/'configs/hardware_profile_vek280_theoretical.yaml')
    assert vek.total_aie_units * vek.physical_aie_tiles_per_compute_group <= vek.physical_aie_tiles == 304
    assert vek.theoretical_compute_peak_ops_per_sec > 100 * synthetic.theoretical_compute_peak_ops_per_sec
    assert vek.effective_compute_peak_ops_per_sec / vek.effective_memory_bandwidth_bytes_per_sec > 10 * (synthetic.effective_compute_peak_ops_per_sec / synthetic.effective_memory_bandwidth_bytes_per_sec)
