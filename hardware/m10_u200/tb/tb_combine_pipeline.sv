`timescale 1ns/1ps
/** Verify Q15 writeback latency and flush in each new combine pipeline stage. */
module tb_combine_pipeline;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic capture_valid = 0;
  logic dma_done = 0;
  logic route_accept = 0;
  logic compute_req_ready = 0;
  logic compute_done = 0;
  logic compute_req_valid;
  logic layer_done;
  logic [511:0] layer_y;
  tile_descriptor_t descriptor;
  tile_descriptor_t unused_gu;
  tile_descriptor_t unused_down;
  logic [7:0] unused_gu_gen;
  logic [7:0] unused_down_gen;

  tile_execution_scheduler #(
    .SLOT_COUNT(2),
    .TOP_K_MAX(1),
    .TILES_PER_EXPERT(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .capture_valid(capture_valid),
    .capture_descriptor(descriptor),
    .capture_generation(8'd1),
    .dma_done(dma_done),
    .dma_descriptor(descriptor),
    .route_accept(route_accept),
    .route_expert(10'd3),
    .route_weight_q15(16'd16384),
    .route_last(1'b1),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(compute_req_ready),
    .gu_descriptor(unused_gu),
    .down_descriptor(unused_down),
    .gu_generation(unused_gu_gen),
    .down_generation(unused_down_gen),
    .compute_done(compute_done),
    .partial_y({16{32'd16}}),
    .layer_done(layer_done),
    .layer_y(layer_y)
  );

  task automatic execute_tile;
    @(negedge clk);
    clear = 1;
    @(negedge clk);
    clear = 0;
    descriptor = '0;
    descriptor.expert_id = 3;
    descriptor.matrix_kind = MATRIX_GU;
    capture_valid = 1;
    @(negedge clk);
    capture_valid = 0;
    dma_done = 1;
    @(negedge clk);
    dma_done = 0;
    descriptor.ocm_slot = 1;
    descriptor.matrix_kind = MATRIX_DOWN;
    capture_valid = 1;
    @(negedge clk);
    capture_valid = 0;
    dma_done = 1;
    route_accept = 1;
    @(negedge clk);
    dma_done = 0;
    route_accept = 0;
    wait (compute_req_valid);
    @(negedge clk);
    compute_req_ready = 1;
    @(negedge clk);
    compute_req_ready = 0;
    compute_done = 1;
    @(negedge clk);
    compute_done = 0;
    if (layer_done || layer_y != 0 || !dut.combine_valid1)
      $fatal(1, "combine must defer publication until writeback");
  endtask

  initial begin
    descriptor = '0;
    repeat (2) @(negedge clk);
    rst_n = 1;
    for (integer stage = 1; stage <= 2; stage = stage + 1) begin
      execute_tile();
      if (stage == 2) @(negedge clk);
      clear = 1;
      @(negedge clk);
      clear = 0;
      repeat (5) begin
        @(negedge clk);
        if (layer_done || layer_y != 0) $fatal(1, "flush leaked combine stage %0d", stage);
      end
    end
    execute_tile();
    @(negedge clk);
    if (layer_done || layer_y != 0) $fatal(1, "product stage published early");
    @(negedge clk);
    if (!layer_done || layer_y != {16{32'd8}}) $fatal(1, "combine final writeback mismatch");
    @(negedge clk);
    if (layer_done) $fatal(1, "duplicate completion");
    $display("PASS: pipelined Q15 combine writeback and stage-wise flush");
    $finish;
  end
  initial begin
    #10000;
    $fatal(1, "combine pipeline timeout");
  end
endmodule
