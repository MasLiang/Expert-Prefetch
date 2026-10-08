"""Run real sequential multilayer RTL on XSim; no simulator semantic changes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from compare_u200_default_alignment import PROJECT, ROOT, command_run
from compare_u200_alignment import write_json


def export_events(text: str, directory: Path) -> dict[str, int]:
    """Export absolute RTL observation cycles and physical-byte accounting."""
    rows = []
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"(FUTURE_ADMIT|FUTURE_DMA|WEIGHTS_READY|LAYER_START|LAYER_RESULT) (.*)", line)
        if match:
            event, fields = match.groups()
            values = dict(re.findall(r"(\w+)=(-?\d+)", fields))
            rows.append(dict(event=event, layer_id=values["layer"], cycle=values["cycle"],
                             current_layer=values.get("current", ""), epoch=values.get("epoch", ""),
                             result=values.get("y", "")))
            counts[event] = counts.get(event, 0) + 1
        if line.startswith("CROSS_BYTES "):
            counts.update({key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)})
    with (directory / "cross_layer_events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("event", "layer_id", "cycle", "current_layer", "epoch", "result"))
        writer.writeheader()
        writer.writerows(rows)
    completed_layers: set[int] = set()
    counts['future_admissions_before_previous_weights'] = 0
    for row in rows:
        layer_id = int(row['layer_id'])
        if row['event'] == 'LAYER_START' and layer_id == 0:
            completed_layers.clear()  # 多次 sequence 重启不能沿用上一条完成历史。
        elif row['event'] == 'WEIGHTS_READY':
            completed_layers.add(layer_id)
        elif row['event'] == 'FUTURE_ADMIT' and layer_id - 1 not in completed_layers:
            counts['future_admissions_before_previous_weights'] += 1
    return counts


def main() -> None:
    """Require numerical, retirement and restart PASS on real layer chains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"results/m10_multilayer_current")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"summary.json",dict(passed=False,status="running",simulator_run=False))
    build = output/"rtl_build"
    build.mkdir(exist_ok=True)
    package = ROOT/"rtl/package.sv"
    benches = [ROOT/"tb/tb_u200_moe.sv",ROOT/"tb/tb_moe_layer_chain.sv",ROOT/"tb/tb_cross_layer_prefetch.sv"]
    sources = [package,*sorted(p for p in (ROOT/"rtl").glob("*.sv") if p != package),*benches]
    write_json(output/"input_manifest.json",dict(backend="XSim",mode="behavioral",simulator_run=False,
        hardware=dict(H=16,F_TILE=16,E=6,K=2,tiles_per_expert=2,ocm_banks=2,slots_per_bank=2),
        limitation="sequential single-token MoE-only 3-bit layer ABI; unique epoch per layer",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    command_run(["xvlog","--sv",*map(str,sources)],build,build/"compile.log")
    rows = []
    for name,top,generics in (
        ("controller","tb_moe_layer_chain",{}),
        ("cross_layer_controller","tb_cross_layer_prefetch",{}),
        ("single_layer","tb_u200_moe",{}),
        ("three_layers","tb_u200_moe",dict(LAYERS=3)),
        ("three_layers_requant","tb_u200_moe",dict(LAYERS=3,QUANT_SHIFT=1)),
        ("eight_layers_restart","tb_u200_moe",dict(LAYERS=8,REPEATS=2)),
        ("future_hit","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=1)),
        ("future_wrong","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=2)),
        ("future_fast","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=1,DDR_GAPS=0)),
        ("future_abort","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=1,ABORT_FUTURE=1)),
        ("future_eight_restart","tb_u200_moe",dict(LAYERS=8,FUTURE_MODE=1,REPEATS=2)),
        ("future_larger_ocm","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=1,OCM_SLOTS_PER_BANK=4)),
        ("future_minimal_table","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=2,DESCRIPTOR_ENTRIES=2,FIFO_DEPTH=2,EXPECT_EARLY_OVERLAP=0)),
        ("future_table_pressure","tb_u200_moe",dict(LAYERS=3,FUTURE_MODE=2,DESCRIPTOR_ENTRIES=4,FIFO_DEPTH=2)),
    ):
        directory = output/name
        directory.mkdir(exist_ok=True)
        snapshot = "chain_"+name
        command = ["xelab","--debug","off","--mt","4","--snapshot",snapshot]
        for key,value in generics.items():
            command += ["--generic_top",f"{key}={value}"]
        command += ["work."+top]
        command_run(command,build,directory/"elaborate.log")
        batch = directory/"run.tcl"
        batch.write_text("run 1 ms\nquit\n",encoding="utf-8")
        timing = command_run(["xsim",snapshot,"--tclbatch",str(batch),"--onerror","quit",
            "--onfinish","quit","--log",str(directory/"simulation.log")],build,directory/"console.log")
        text = (directory/"simulation.log").read_text(encoding="utf-8")
        if text.count("PASS:") != 1 or "$finish called" not in text or re.search(r"Fatal:|FATAL:|ERROR:|Error:",text):
            raise ValueError(f"multilayer acceptance failed: {name}")
        observations = export_events(text,directory)
        if name in ('future_hit', 'future_larger_ocm') and not observations['future_admissions_before_previous_weights']:
            raise ValueError(f'旧前层权重屏障仍在控制未来层 admission: {name}')
        if name == 'future_minimal_table' and observations.get('FUTURE_ADMIT', 0):
            raise ValueError('最小 table 被未来层 speculation 占用，破坏 demand 保留')
        rows.append(dict(name=name,passed=True,generics=generics,timing=timing,
                         ocm_slots=2*generics.get("OCM_SLOTS_PER_BANK",2) if top=="tb_u200_moe" else None,
                         observations=observations))
        print(name+": PASS",flush=True)
    write_json(output/"summary.json",dict(passed=True,cases=rows,simulator_run=False,
        limitations=["Sequential single token, fixed geometry", "No attention/residual", "No synthesis/STA claim"]))


if __name__ == "__main__":
    main()
