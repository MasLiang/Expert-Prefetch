`timescale 1ns/1ps
/** A previous-cycle registered route accept must not resurrect flushed demand. */
module tb_demander_flush_priority;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic route_accept = 0;
  logic [9:0] route_expert = 3;
  logic start = 0;
  logic logical_ready = 0;
  logic logical_valid;
  logic active;
  logic done;
  tile_descriptor_t empty_descriptor;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  true_tile_demander #(
    .MAX_CANDIDATES(2),
    .TOP_K_MAX(2),
    .TILES_PER_EXPERT(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .speculative_accept(1'b0),
    .speculative_descriptor(empty_descriptor),
    .route_accept(route_accept),
    .route_epoch(16'd1),
    .route_layer(3'd0),
    .route_expert(route_expert),
    .start(start),
    .sequence_id(16'd0),
    .gu_base_addr(64'd0),
    .down_base_addr(64'd4096),
    .expert_stride_bytes(32'd512),
    .physical_tile_bytes(32'd128),
    .logical_valid(logical_valid),
    .logical_ready(logical_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .active(active),
    .done(done)
  );
  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask
  initial begin
    empty_descriptor = '0;
    repeat (2) tick();
    @(negedge clk);
    rst_n = 1;
    route_accept = 1;
    tick();
    @(negedge clk);
    // All stale producer signals coincide with clear. Clear must dominate.
    clear = 1;
    start = 1;
    logical_ready = 1;
    tick();
    if (active || done) $fatal(1, "flush overridden by stale resolution start");
    @(negedge clk);
    clear = 0;
    route_accept = 0;
    start = 0;
    logical_ready = 0;
    // Start without any fresh truth proves the old accept was not retained.
    tick();
    @(negedge clk);
    start = 1;
    tick();
    if (active || logical_valid) $fatal(1, "flush retained a stale Router accept");
    @(negedge clk);
    start = 0;
    route_accept = 1;
    route_expert = 7;
    tick();
    @(negedge clk);
    route_accept = 0;
    start = 1;
    tick();
    if (!active || !logical_valid || gu.expert_id != 7)
      $fatal(1, "fresh demand truth failed after flush");
    @(negedge clk);
    start = 0;
    logical_ready = 1;
    tick();
    if (!done || active) $fatal(1, "stale truth appended a second expert after restart");
    @(negedge clk);
    logical_ready = 0;
    tick();
    if (logical_valid || active) $fatal(1, "completed demand repeated after flush restart");
    $display("PASS: demander flush dominates accept/start/advance and clean restart");
    $finish;
  end
  initial begin
    #1000;
    $fatal(1, "demander flush watchdog");
  end
endmodule
