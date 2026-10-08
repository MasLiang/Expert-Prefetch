`timescale 1ns/1ps
/** Balanced ready-pair selection: non-power-of-two padding, tie-breaks,
 * backpressure, next-tile snapshots, and clear followed by slot reuse. */
module tb_execution_selection #(
  parameter integer SLOTS = 5,
  parameter integer EXPERTS = 3
);
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic capture_valid = 0;
  logic dma_done = 0;
  logic route_accept = 0;
  logic route_last = 0;
  logic [9:0] route_expert = 0;
  logic compute_req_ready = 0;
  logic compute_done = 0;
  logic compute_req_valid;
  logic layer_done;
  logic [31:0] layer_y;
  logic [7:0] gu_generation;
  logic [7:0] down_generation;
  tile_descriptor_t incoming;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  tile_descriptor_t held_gu;
  tile_descriptor_t held_down;

  tile_execution_scheduler #(
    .SLOT_COUNT(SLOTS),
    .TOP_K_MAX(EXPERTS),
    .TILES_PER_EXPERT(2),
    .H(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .capture_valid(capture_valid),
    .capture_descriptor(incoming),
    .capture_generation(8'd5),
    .dma_done(dma_done),
    .dma_descriptor(incoming),
    .route_accept(route_accept),
    .route_expert(route_expert),
    .route_weight_q15(16'd32768),
    .route_last(route_last),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(compute_req_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .gu_generation(gu_generation),
    .down_generation(down_generation),
    .compute_done(compute_done),
    .gu_release_valid(1'b0),
    .partial_y(32'd6),
    .layer_done(layer_done),
    .layer_y(layer_y)
  );

  task automatic load_slot(input integer slot, input integer expert, input matrix_kind_t kind);
    @(negedge clk);
    incoming = '0;
    incoming.ocm_slot = 16'(slot);
    incoming.expert_id = 10'(expert);
    incoming.matrix_kind = kind;
    capture_valid = 1;
    @(negedge clk);
    capture_valid = 0;
    dma_done = 1;
    @(negedge clk);
    dma_done = 0;
  endtask

  task automatic send_truth(input integer expert, input logic last);
    @(negedge clk);
    route_expert = 10'(expert);
    route_last = last;
    route_accept = 1;
    @(negedge clk);
    route_accept = 0;
  endtask

  task automatic finish_compute;
    @(negedge clk);
    compute_req_ready = 1;
    @(negedge clk);
    compute_req_ready = 0;
    compute_done = 1;
    @(negedge clk);
    compute_done = 0;
  endtask

  initial begin
    incoming = '0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    load_slot(0, 3, MATRIX_GU);
    load_slot(1, 3, MATRIX_DOWN);
    load_slot(2, 7, MATRIX_GU);
    load_slot(3, 7, MATRIX_DOWN);
    load_slot(4, 7, MATRIX_GU);
    send_truth(7, 0);
    send_truth(3, 0);
    send_truth(9, 1);
    wait (compute_req_valid);
    if (gu.expert_id != 7 || gu.ocm_slot != 4 || down.ocm_slot != 3 ||
        gu_generation != 5 || down_generation != 5)
      $fatal(1, "Router-order/highest-slot tie break or generation changed");
    held_gu = gu;
    held_down = down;
    repeat (5) begin
      @(negedge clk);
      if (!compute_req_valid || gu !== held_gu || down !== held_down)
        $fatal(1, "request changed under backpressure");
    end
    finish_compute();
    wait (compute_req_valid);
    if (gu.expert_id != 3 || gu.tile_id != 0)
      $fatal(1, "stale snapshot repeated consumed tile or ready expert lost");
    @(negedge clk);
    clear = 1;
    #0.1;
    if (compute_req_valid) $fatal(1, "clear did not mask pending request");
    @(negedge clk);
    clear = 0;
    repeat (4) begin
      @(negedge clk);
      if (compute_req_valid || layer_done || layer_y != 0)
        $fatal(1, "clear leaked a stale selection/output");
    end
    load_slot(0, 9, MATRIX_GU);
    load_slot(1, 9, MATRIX_DOWN);
    send_truth(9, 1);
    wait (compute_req_valid);
    if (gu.expert_id != 9 || gu.ocm_slot != 0 || down.ocm_slot != 1)
      $fatal(1, "slot reuse after clear selected stale metadata");
    $display("PASS: execution balanced selection slots=%0d experts=%0d", SLOTS, EXPERTS);
    $finish;
  end
  initial begin
    #10000;
    $fatal(1, "execution selection watchdog");
  end
endmodule
