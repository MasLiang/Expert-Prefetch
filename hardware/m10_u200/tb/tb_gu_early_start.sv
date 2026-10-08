`timescale 1ns/1ps
/** GU可先运行；DOWN长期背压、descriptor身份、flush与完整退休。 */
module tb_gu_early_start;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic capture_valid = 0;
  logic dma_done = 0;
  logic route_accept = 0;
  logic request_valid, request_ready, compute_done;
  logic down_resident = 0;
  logic gu_start, down_start, release_pair, busy, waiting_down;
  logic release_gu;
  logic layer_done;
  logic [31:0] layer_y;
  logic [7:0] gu_generation, down_generation;
  tile_descriptor_t incoming, gu, down;

  tile_execution_scheduler #(
    .SLOT_COUNT(2),
    .TOP_K_MAX(1),
    .TILES_PER_EXPERT(1),
    .H(1)
  ) scheduler (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .capture_valid(capture_valid),
    .capture_descriptor(incoming),
    .capture_generation(8'd5),
    .dma_done(dma_done),
    .dma_descriptor(incoming),
    .route_accept(route_accept),
    .route_expert(10'd3),
    .route_weight_q15(16'd32768),
    .route_last(1'b1),
    .compute_req_valid(request_valid),
    .compute_req_ready(request_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .gu_generation(gu_generation),
    .down_generation(down_generation),
    .compute_done(compute_done),
    .gu_release_valid(release_gu),
    .partial_y(32'd6),
    .layer_done(layer_done),
    .layer_y(layer_y)
  );
  tile_compute_controller controller (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .compute_req_valid(request_valid),
    .compute_req_ready(request_ready),
    .compute_gu_descriptor(gu),
    .compute_gu_slot_generation(gu_generation),
    .compute_down_descriptor(down),
    .compute_down_slot_generation(down_generation),
    .gu_resident(1'b1),
    .down_resident(down_resident),
    .gu_cycles(32'd2),
    .nonlinear_cycles(32'd1),
    .down_cycles(32'd2),
    .backend_gu_done(1'b0),
    .backend_nonlinear_done(1'b0),
    .backend_down_done(1'b0),
    .busy(busy),
    .done(compute_done),
    .waiting_gu(),
    .waiting_down(waiting_down),
    .gu_compute_start(gu_start),
    .down_compute_start(down_start),
    .consumer_gu_release_valid(release_gu),
    .consumer_release_valid(release_pair),
    .consumer_release_gu_slot(),
    .consumer_release_down_slot(),
    .consumer_release_epoch(),
    .consumer_release_gu_generation(),
    .consumer_release_down_generation()
  );

  task automatic capture(input integer slot, input integer expert, input matrix_kind_t matrix);
    @(negedge clk);
    incoming = '0;
    incoming.epoch_id = 1;
    incoming.ocm_slot = 16'(slot);
    incoming.expert_id = 10'(expert);
    incoming.matrix_kind = matrix;
    capture_valid = 1;
    @(negedge clk);
    capture_valid = 0;
  endtask

  initial begin
    incoming = '0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    for (integer round = 0; round < 2; round = round + 1) begin
      @(negedge clk);
      route_accept = 1;
      @(negedge clk);
      route_accept = 0;
      capture(0,3,MATRIX_GU);
      dma_done = 1;
      @(negedge clk);
      dma_done = 0;
      repeat (5) begin
        @(negedge clk);
        if (request_valid || busy) $fatal(1,"missing DOWN descriptor must block issue");
      end
      capture(1,7,MATRIX_DOWN);
      repeat (5) begin
        @(negedge clk);
        if (request_valid || busy) $fatal(1,"wrong Expert DOWN must not match GU");
      end
      capture(1,3,MATRIX_DOWN); // descriptor present, no DOWN DMA completion yet
      wait (gu_start);
      if (down_resident || scheduler.ready_mem[1] || gu.expert_id != 3 || down.expert_id != 3 ||
          gu_generation != 5 || down_generation != 5)
        $fatal(1,"GU did not start early with stable tagged pair");
      wait (waiting_down);
      repeat (20) begin
        @(negedge clk);
        if (down_start || compute_done || release_pair || layer_done || !busy ||
            scheduler.valid_mem[0] || !scheduler.valid_mem[1])
          $fatal(1,"GU was not retired or DOWN wait prematurely released DOWN ownership");
      end
      if (round == 0) begin
        clear = 1;
        @(negedge clk);
        clear = 0;
        repeat (5) begin
          @(negedge clk);
          if (busy || compute_done || release_pair || layer_done) $fatal(1,"flush leaked late result");
        end
      end else begin
        down_resident = 1;
        incoming = down;
        dma_done = 1;
        @(negedge clk);
        dma_done = 0;
        wait (down_start);
        wait (layer_done);
        if (layer_y != 6 || scheduler.valid_mem[0] || scheduler.valid_mem[1])
          $fatal(1,"final tile result or pair retirement failed");
      end
    end
    $display("PASS: GU early issue, delayed DOWN, pair identity/retirement and flush recovery");
    $finish;
  end
  initial begin
    #10000;
    $fatal(1,"GU early test watchdog");
  end
endmodule
