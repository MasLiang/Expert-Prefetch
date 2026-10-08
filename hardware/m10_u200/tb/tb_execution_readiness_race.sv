`timescale 1ns/1ps
/** E1 snapshot ready while E0 becomes ready: preserve live Router-order priority. */
module tb_execution_readiness_race;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic capture_valid = 0;
  logic dma_done = 0;
  logic route_accept = 0;
  logic route_last = 0;
  logic [9:0] route_expert = 0;
  logic compute_req_valid;
  tile_descriptor_t incoming;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  tile_execution_scheduler #(
    .SLOT_COUNT(4),
    .TOP_K_MAX(2),
    .TILES_PER_EXPERT(1),
    .H(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .capture_valid(capture_valid),
    .capture_descriptor(incoming),
    .capture_generation(8'd1),
    .dma_done(dma_done),
    .dma_descriptor(incoming),
    .route_accept(route_accept),
    .route_expert(route_expert),
    .route_weight_q15(16'd32768),
    .route_last(route_last),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(1'b0),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .compute_done(1'b0),
    .gu_release_valid(1'b0),
    .partial_y(32'd0)
  );

  function automatic tile_descriptor_t descriptor(input integer slot);
    tile_descriptor_t value;
    value = '0;
    value.ocm_slot = 16'(slot);
    value.expert_id = slot < 2 ? 10'd3 : 10'd7;
    if (slot % 2 == 0) value.matrix_kind = MATRIX_GU;
    else value.matrix_kind = MATRIX_DOWN;
    return value;
  endfunction

  initial begin
    incoming = '0;
    repeat (2) @(negedge clk);
    rst_n = 1;
    for (integer slot = 0; slot < 4; slot = slot + 1) begin
      @(negedge clk);
      incoming = descriptor(slot);
      capture_valid = 1;
    end
    @(negedge clk);
    capture_valid = 0;
    incoming = descriptor(0);
    dma_done = 1;
    @(negedge clk);
    incoming = descriptor(2);
    @(negedge clk);
    dma_done = 0;
    route_accept = 1;
    route_expert = 3;
    route_last = 0;
    @(negedge clk);
    route_expert = 7;
    route_last = 1;
    dma_done = 1;
    incoming = descriptor(3);
    @(negedge clk);
    route_accept = 0;
    route_last = 0;
    incoming = descriptor(1);
    @(negedge clk);
    dma_done = 0;
    wait (compute_req_valid);
    #0.1;
    if (gu.expert_id != 3 || gu.ocm_slot != 0 || down.ocm_slot != 1)
      $fatal(1, "stale E1 readiness snapshot defeated currently ready earlier E0");
    $display("PASS: live ready-first priority across adjacent DMA completion edges");
    $finish;
  end
  initial begin
    #1000;
    $fatal(1, "readiness race watchdog");
  end
endmodule
