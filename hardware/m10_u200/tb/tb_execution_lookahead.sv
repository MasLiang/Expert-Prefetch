`timescale 1ns/1ps
/** Next-prefix search, retirement mutations, live priority and flush. */
module tb_execution_lookahead #(
  parameter integer SLOTS = 8
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
  logic [7:0] capture_generation = 5;
  logic request_ready = 0;
  logic request_valid;
  logic compute_done = 0;
  logic layer_done;
  logic [31:0] layer_y;
  logic [7:0] gu_generation, down_generation;
  tile_descriptor_t incoming, gu, down, held_gu, held_down;

  tile_execution_scheduler #(
    .SLOT_COUNT(SLOTS),
    .TOP_K_MAX(2),
    .TILES_PER_EXPERT(2),
    .H(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .capture_valid(capture_valid),
    .capture_descriptor(incoming),
    .capture_generation(capture_generation),
    .dma_done(dma_done),
    .dma_descriptor(incoming),
    .route_accept(route_accept),
    .route_expert(route_expert),
    .route_weight_q15(16'd32768),
    .route_last(route_last),
    .compute_req_valid(request_valid),
    .compute_req_ready(request_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .gu_generation(gu_generation),
    .down_generation(down_generation),
    .compute_done(compute_done),
    .gu_release_valid(1'b0),
    .partial_y(32'd1),
    .layer_done(layer_done),
    .layer_y(layer_y)
  );

  function automatic tile_descriptor_t descriptor(input integer slot);
    tile_descriptor_t value;
    value = '0;
    value.epoch_id = 1;
    value.ocm_slot = 16'(slot);
    value.expert_id = slot < 4 ? 10'd3 : 10'd7;
    value.tile_id = 16'((slot % 4)/2);
    if (slot % 2 == 0) value.matrix_kind = MATRIX_GU;
    else value.matrix_kind = MATRIX_DOWN;
    return value;
  endfunction

  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask

  task automatic load_slot(input integer slot, input bit ready);
    @(negedge clk);
    incoming = descriptor(slot);
    capture_valid = 1;
    tick();
    @(negedge clk);
    capture_valid = 0;
    dma_done = ready;
    tick();
    @(negedge clk);
    dma_done = 0;
  endtask

  initial begin
    incoming = '0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    for (integer mode = 0; mode < 5; mode = mode + 1) begin
      capture_generation = 5;
      for (integer slot = 0; slot < 8; slot = slot + 1)
        load_slot(slot, !((mode == 1 && slot == 2) || (mode == 2 && slot == 4) ||
                         (mode == 4 && slot == 0)));
      @(negedge clk);
      route_accept = 1;
      route_expert = 3;
      route_last = 0;
      @(negedge clk);
      route_expert = 7;
      route_last = 1;
      @(negedge clk);
      route_accept = 0;
      route_last = 0;
      wait (request_valid);
      if (gu.expert_id != (mode == 4 ? 7 : 3) || gu.tile_id != 0)
        $fatal(1,"unexpected initial ready expert mode=%0d",mode);
      held_gu = gu;
      held_down = down;
      @(negedge clk);
      request_ready = 1;
      tick();
      if (!dut.running || request_valid) $fatal(1,"request was not accepted once");
      @(negedge clk);
      request_ready = 0;
      repeat (4) begin
        tick();
        if (request_valid || gu !== held_gu || down !== held_down)
          $fatal(1,"lookahead overwrote active descriptor or launched concurrently");
      end
      @(negedge clk);
      compute_done = 1;
      if (mode == 1 || mode == 4) begin
        incoming = descriptor(mode == 1 ? 2 : 0);
        dma_done = 1;
      end
      if (mode == 2) begin
        incoming = descriptor(2);
        incoming.expert_id = 9;
        capture_generation = 7;
        capture_valid = 1;
      end
      if (mode == 3) clear = 1;
      tick();
      if (mode == 0 && (!dut.select_for_issue || dut.search_stale))
        $fatal(1,"resident successor required a post-retirement search refresh");
      if ((mode == 1 || mode == 2 || mode == 4) && dut.select_for_issue)
        $fatal(1,"retirement mutation used stale lookup");
      @(negedge clk);
      compute_done = 0;
      capture_valid = 0;
      dma_done = 0;
      if (mode == 3) begin
        clear = 0;
        repeat (5) begin
          tick();
          if (request_valid || layer_done || layer_y != 0 || dut.running)
            $fatal(1,"retirement/clear leaked successor or result");
        end
      end else begin
        if (mode == 2) begin
          repeat (5) begin
            tick();
            if (request_valid) $fatal(1,"reused wrong-expert slot inherited old readiness");
          end
          capture_generation = 9;
          load_slot(2,1);
        end
        if (mode == 0) begin
          tick();
          if (!request_valid) $fatal(1,"lookahead did not fill stable request on next edge");
        end else wait (request_valid);
        if (gu.expert_id != 3 || gu.tile_id != (mode == 4 ? 0 : 1) ||
            gu_generation != (mode == 2 ? 9 : 5) || down_generation != 5)
          $fatal(1,"prefix/live priority/generation revalidation failed mode=%0d",mode);
        held_gu = gu;
        held_down = down;
        // Unrelated completion cannot perturb an already valid stalled request.
        @(negedge clk);
        incoming = descriptor(7);
        dma_done = 1;
        repeat (5) begin
          tick();
          if (!request_valid || gu !== held_gu || down !== held_down)
            $fatal(1,"pending request changed under backpressure");
        end
        @(negedge clk);
        dma_done = 0;
      end
      @(negedge clk);
      clear = 1;
      tick();
      if (request_valid || layer_done || layer_y != 0) $fatal(1,"clear failed to discard lookahead");
      @(negedge clk);
      clear = 0;
    end
    $display("PASS: successor lookahead, retirement mutation/priority, generation reuse and clear slots=%0d",SLOTS);
    $finish;
  end
  initial begin
    #20000;
    $fatal(1,"lookahead watchdog");
  end
endmodule
