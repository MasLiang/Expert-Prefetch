"""Generate deterministic HardwareGreedy vectors and check frozen RTL expectations."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from edge_moe_sim.hardware_greedy_v1 import (
    HardwareGreedyV1Candidate,
    HardwareGreedyV1Config,
    hardware_greedy_v1,
)

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "traces" / "deterministic" / "hardware_greedy_vectors.json"
REPORT = ROOT / "reports" / "rtl_python_comparison.json"


def python_order(case: dict[str, object]) -> list[dict[str, int]]:
    logical_bytes = int(case["gu_bytes"]) + int(case["down_bytes"])
    candidates = tuple(
        HardwareGreedyV1Candidate(
            int(row["expert_id"]), rank, round(float(row["confidence"]) * 4095),
            (logical_bytes,) * int(case["tiles_per_expert"]),
        )
        for rank, row in enumerate(case["candidates"])
    )
    plan = hardware_greedy_v1(
        candidates,
        HardwareGreedyV1Config(
            int(case["channels"]), int(case["budget_bytes"]),
            (int(case["budget_bytes"]),) * int(case["channels"]),
        ),
    )
    return [{"expert_id": item.expert_id, "tile_id": item.tile_id,
             "channel": item.memory_channel, "logical_bytes": item.logical_bytes}
            for item in plan]


def rtl_order(case: dict[str, object]) -> list[dict[str, int]]:
    """Compile and execute the actual RTL selector for a complete allocation trace."""
    rows = case["candidates"]
    count = len(rows)
    channels = int(case["channels"])
    tiles = int(case["tiles_per_expert"])
    budget = int(case["budget_bytes"])
    tile_bytes = int(case["gu_bytes"]) + int(case["down_bytes"])
    loads_width = max(1, (count * tiles).bit_length())
    expert = sum(int(row["expert_id"]) << (10 * i) for i, row in enumerate(rows))
    rank = sum(i << (5 * i) for i in range(count))
    confidence = sum(round(float(row["confidence"]) * 4095) << (12 * i) for i, row in enumerate(rows))
    byte_vector = sum(tile_bytes << (32 * i) for i in range(count))
    credit = sum(budget << (32 * i) for i in range(channels))
    source = f'''module golden_trace;
  localparam integer C={count}, CH={channels}, LW={loads_width};
  logic [C-1:0] valid_rows = '1;
  logic [C*10-1:0] experts = {count*10}'h{expert:x};
  logic [C*5-1:0] ranks = {count*5}'h{rank:x};
  logic [C*12-1:0] confidences = {count*12}'h{confidence:x};
  logic [C*8-1:0] depths = '0;
  logic [C*32-1:0] byte_sizes = {count*32}'h{byte_vector:x};
  logic [31:0] global_credit = 32'd{budget};
  logic [CH*32-1:0] channel_credit = {channels*32}'h{credit:x};
  logic [CH*LW-1:0] loads = '0;
  logic winner_valid;
  logic [{max(1,(count-1).bit_length())-1}:0] winner_index;
  logic [{max(1,(channels-1).bit_length())-1}:0] winner_channel;
  integer selected_index;
  integer selected_channel;
  hardware_greedy_v1 #(
    .CANDIDATE_COUNT(C),
    .CHANNEL_COUNT(CH),
    .MAX_ALLOCATIONS(C*{tiles})
  ) dut (
    .candidate_valid(valid_rows),
    .candidate_expert_id(experts),
    .candidate_rank(ranks),
    .candidate_confidence_q12(confidences),
    .candidate_next_depth(depths),
    .candidate_logical_bytes(byte_sizes),
    .global_credit_bytes(global_credit),
    .channel_credit_bytes(channel_credit),
    .channel_load_allocations(loads),
    .winner_valid(winner_valid),
    .winner_index(winner_index),
    .winner_channel(winner_channel)
  );
  initial begin
    #1;
    while (winner_valid) begin
      selected_index = winner_index;
      selected_channel = winner_channel;
      $display("ALLOC %0d %0d %0d %0d", experts[selected_index*10 +: 10],
        depths[selected_index*8 +: 8], selected_channel, {tile_bytes});
      global_credit = global_credit - {tile_bytes};
      channel_credit[selected_channel*32 +: 32] = channel_credit[selected_channel*32 +: 32] - {tile_bytes};
      loads[selected_channel*LW +: LW] = loads[selected_channel*LW +: LW] + 1;
      depths[selected_index*8 +: 8] = depths[selected_index*8 +: 8] + 1;
      if (depths[selected_index*8 +: 8] == {tiles}) valid_rows[selected_index] = 0;
      #1;
    end
    $finish;
  end
endmodule
'''
    with tempfile.TemporaryDirectory(prefix="m101-golden-") as directory:
        bench = Path(directory) / "golden_trace.sv"
        executable = Path(directory) / "golden_trace.out"
        bench.write_text(source, encoding="utf-8")
        subprocess.run(["iverilog", "-g2012", "-s", "golden_trace", "-o", str(executable),
                        str(ROOT / "rtl/hardware_greedy_v1.sv"), str(bench)],
                       check=True, capture_output=True, text=True)
        output = subprocess.run(["vvp", str(executable)], check=True,
                                capture_output=True, text=True, timeout=30).stdout
    result = []
    for line in output.splitlines():
        if line.startswith("ALLOC "):
            values = [int(value) for value in line.split()[1:]]
            result.append(dict(zip(("expert_id", "tile_id", "channel", "logical_bytes"), values)))
    return result


def main() -> None:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    mismatches: list[dict[str, object]] = []
    results = []
    for case in vectors["cases"]:
        actual = python_order(case)
        expected = case["expected_logical_allocation"]
        rtl_actual = rtl_order(case)
        passed = actual == expected == rtl_actual
        results.append({"name": case["name"], "passed": passed, "python": actual, "rtl": rtl_actual})
        if not passed:
            mismatches.append({"name": case["name"], "expected": expected, "python": actual, "rtl": rtl_actual})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"passed": not mismatches, "cases": results,
                                  "mismatches": mismatches,
                                  "rtl_contract": "M10 HardwareGreedy-v1 integer policy; historical M7-M9.3 policy is not used"}, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise SystemExit(f"{len(mismatches)} Python/RTL vector mismatches; see {REPORT}")
    print(f"PASS: {len(results)} executed RTL/Python allocation traces, zero mismatch")


if __name__ == "__main__":
    main()
