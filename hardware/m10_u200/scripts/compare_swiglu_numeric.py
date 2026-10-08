"""Execute RTL against independent dense int8 SwiGLU reference matrices."""
from __future__ import annotations

import json
import math
import random
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def saturate(value: int) -> int:
    """Saturate to a signed int8 value."""
    return max(-128, min(127, value))


def run_case(hidden: int, tile: int, seed: int) -> dict[str, int | bool]:
    """Check every vector element for a seeded dense-matrix RTL execution."""
    rng = random.Random(seed)
    activation = [rng.randrange(-16, 17) for _ in range(hidden)]
    gate = [[rng.randrange(-3, 4) for _ in range(hidden)] for _ in range(tile)]
    up = [[rng.randrange(-3, 4) for _ in range(hidden)] for _ in range(tile)]
    down = [[rng.randrange(-3, 4) for _ in range(tile)] for _ in range(hidden)]
    nonlinear = []
    for gate_row, up_row in zip(gate, up):
        g = saturate(sum(x * w for x, w in zip(activation, gate_row)) >> 1)
        u = saturate(sum(x * w for x, w in zip(activation, up_row)) >> 1)
        silu_q4 = saturate(math.floor(g / (1.0 + math.exp(-g / 16.0)) + 0.5))
        nonlinear.append(saturate((silu_q4 * u) >> 4))
    expected = [sum(a * w for a, w in zip(nonlinear, row)) & 0xFFFFFFFF for row in down]
    payload = [value & 255 for matrix in (gate, up, down) for row in matrix for value in row]
    words = [sum(byte << (8 * lane) for lane, byte in enumerate(payload[offset:offset + 16]))
             for offset in range(0, len(payload), 16)]
    activation_hex = sum((value & 255) << (8 * lane) for lane, value in enumerate(activation))
    with tempfile.TemporaryDirectory(prefix="m101-numeric-") as directory:
        path = Path(directory)
        memory = path / "payload.hex"
        output = path / "expected.hex"
        executable = path / "test.out"
        memory.write_text("".join(f"{word:032x}\n" for word in words), encoding="utf-8")
        output.write_text("".join(f"{word:08x}\n" for word in expected), encoding="utf-8")
        subprocess.run(["iverilog", "-g2012", "-s", "tb_swiglu_dense_golden",
                        f"-Ptb_swiglu_dense_golden.H={hidden}",
                        f"-Ptb_swiglu_dense_golden.F_TILE={tile}", "-o", str(executable),
                        str(ROOT / "rtl/int8_swiglu_tile_backend.sv"),
                        str(ROOT / "tb/tb_swiglu_dense_golden.sv")],
                       check=True, capture_output=True, text=True)
        run = subprocess.run(["vvp", str(executable), f"+MEM={memory}",
                              f"+EXPECTED={output}", f"+ACT={activation_hex:x}"],
                             check=True, capture_output=True, text=True, timeout=30)
        print(run.stdout.strip())
    return {"hidden": hidden, "tile": tile, "seed": seed, "passed": True}


def main() -> None:
    """Run all deterministic dense configurations and save a comparison report."""
    results = [run_case(h, f, seed) for h, f in ((16, 16), (32, 16), (16, 32))
               for seed in (17, 29)]
    report = ROOT / "reports/swiglu_numeric_comparison.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"passed": True, "cases": results, "mismatches": []}, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
