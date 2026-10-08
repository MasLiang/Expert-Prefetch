`timescale 1ns/1ps
/** 检查选择流水级在 backpressure、credit 更新和 stop 下的 descriptor 稳定性。 */
module tb_candidate_scheduler_backpressure;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n = 0;
  logic transaction_begin = 0;
  logic stop_allocation = 0;
  logic candidate_accept = 0;
  logic logical_ready = 0;
  logic logical_valid;
  logic allocation_active;
  logic allocation_done;
  logic [31:0] remaining_credit_bytes;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  tile_descriptor_t saved_gu;
  tile_descriptor_t saved_down;

  candidate_prefetch_scheduler #(
    .MAX_CANDIDATES(1),
    .TILES_PER_EXPERT(2),
    .GU_BYTE_RATIO(2)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .transaction_begin(transaction_begin),
    .stop_allocation(stop_allocation),
    .transaction_epoch(16'd1),
    .transaction_sequence(16'd2),
    .transaction_layer(3'd0),
    .speculative_budget_bytes(32'd1536),
    .channel_credit_bytes({4{32'd1536}}),
    .candidate_accept(candidate_accept),
    .candidate_epoch(16'd1),
    .candidate_layer(3'd0),
    .candidate_expert(10'd3),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd4095),
    .candidate_last(1'b1),
    .gu_base_addr(64'h1000),
    .down_base_addr(64'h9000),
    .expert_stride_bytes(32'd4096),
    .physical_tile_bytes(32'd256),
    .logical_valid(logical_valid),
    .logical_ready(logical_ready),
    .logical_gu_descriptor(gu),
    .logical_down_descriptor(down),
    .allocation_active(allocation_active),
    .allocation_done(allocation_done),
    .remaining_credit_bytes(remaining_credit_bytes)
  );

  task automatic begin_prediction;
    @(negedge clk);
    transaction_begin = 1;
    @(negedge clk);
    transaction_begin = 0;
    candidate_accept = 1;
    @(negedge clk);
    candidate_accept = 0;
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1;
    begin_prediction();
    wait (logical_valid);
    @(negedge clk);
    saved_gu = gu;
    saved_down = down;
    if (gu.tile_id != 0 || gu.external_addr != 64'h4000 ||
        down.external_addr != 64'hc000) $fatal(1, "initial geometry mismatch");
    repeat (10) begin
      @(negedge clk);
      if (!logical_valid || gu !== saved_gu || down !== saved_down ||
          remaining_credit_bytes != 1536 || dut.channel_remaining[3] != 1536)
        $fatal(1, "backpressure changed request or prematurely consumed credit");
    end
    logical_ready = 1;
    @(negedge clk);
    logical_ready = 0;
    if (remaining_credit_bytes != 768 || dut.channel_remaining[3] != 768 ||
        dut.channel_remaining[0] != 1536) $fatal(1, "first admission credit mismatch");
    wait (logical_valid);
    @(negedge clk);
    if (gu.tile_id != 1 || gu.external_addr != 64'h4200 ||
        down.external_addr != 64'hc100) $fatal(1, "stale winner/depth reused");
    saved_gu = gu;
    saved_down = down;
    repeat (5) begin
      @(negedge clk);
      if (!logical_valid || gu !== saved_gu || down !== saved_down ||
          remaining_credit_bytes != 768 || dut.channel_remaining[3] != 768)
        $fatal(1, "second issue stall consumed or reused stale credit");
    end
    logical_ready = 1;
    @(negedge clk);
    logical_ready = 0;
    wait (allocation_done);
    if (remaining_credit_bytes != 0 || dut.channel_remaining[3] != 0 ||
        dut.channel_remaining[0] != 1536) $fatal(1, "second admission credit mismatch");
    begin_prediction();
    wait (logical_valid);
    @(negedge clk);
    stop_allocation = 1;
    logical_ready = 1;
    repeat (4) begin
      @(negedge clk);
      if (logical_valid || remaining_credit_bytes != 1536)
        $fatal(1, "stop admitted a pending descriptor");
    end
    $display("PASS: candidate scheduler backpressure/pipeline/stop");
    $finish;
  end

  initial begin
    #10000;
    $fatal(1, "candidate scheduler timeout");
  end
endmodule
