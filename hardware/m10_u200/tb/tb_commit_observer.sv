`timescale 1ns/1ps
/** 独立被动 wrapper：不改变基础激励，观察 payload 与 committed 的区别。 */
module tb_commit_observer #(
  parameter integer FUTURE_MODE=1
);
  tb_u200_moe #(
    .LAYERS(3),
    .FUTURE_MODE(FUTURE_MODE),
    .DDR_GAPS(0),
    .BUDGET(0)
  ) bench ();
  always @(posedge bench.clk) begin
    if(bench.rst_n && (bench.dut.core.auto_req_valid || bench.dut.core.memory_dataflow.core.compute.phase==1))
      $display("FOCUS_GU cycle=%0d layer=%0d phase=%0d ready=%0d state=%0d payload=%0d",
        bench.cycle,bench.layer,bench.dut.core.memory_dataflow.core.compute.phase,
        bench.dut.core.memory_dataflow.core.compute_gu_ready_state,
        bench.dut.core.memory_dataflow.core.compute_gu_state,
        bench.dut.core.memory_dataflow.core.compute_gu_payload_complete);
    if(bench.rst_n && bench.dut.core.memory_dataflow.core.table_mark_promoted)
      $display("FOCUS_COMMIT cycle=%0d layer=%0d expert=%0d tile=%0d matrix=%0d",
        bench.cycle,bench.dut.core.memory_dataflow.core.route_query_descriptor.layer_id,
        bench.dut.core.memory_dataflow.core.route_query_descriptor.expert_id,
        bench.dut.core.memory_dataflow.core.route_query_descriptor.tile_id,
        bench.dut.core.memory_dataflow.core.route_query_descriptor.matrix_kind);
  end
endmodule
