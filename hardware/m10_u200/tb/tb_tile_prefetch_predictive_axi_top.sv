`timescale 1ns/1ps

/** End-to-end parser -> scheduler -> resolver -> AXI/OCM smoke test. */
module tb_tile_prefetch_predictive_axi_top;
  logic clk;
  logic rst_n;
  logic prediction_begin_valid;
  logic prediction_begin_ready;
  logic candidate_valid;
  logic candidate_ready;
  logic candidate_last;
  logic route_begin_valid;
  logic route_begin_ready;
  logic route_expert_valid;
  logic route_expert_ready;
  logic route_last;
  logic parser_error;
  logic [31:0] parser_stale_transaction_count;
  logic [31:0] demand_stall_cycles;
  logic [31:0] stale_event_count;
  logic [31:0] discarded_dma_write_count;
  logic [31:0] payload_discarded_write_count;
  logic [31:0] dma_response_error_count;
  logic [31:0] tombstone_skip_count;
  logic [63:0] m_axi_araddr;
  logic [7:0] m_axi_arlen;
  logic [2:0] m_axi_arsize;
  logic [1:0] m_axi_arburst;
  logic m_axi_arvalid;
  logic m_axi_arready;
  logic [127:0] m_axi_rdata;
  logic [1:0] m_axi_rresp;
  logic m_axi_rlast;
  logic m_axi_rvalid;
  logic m_axi_rready;
  integer accepted_count;
  integer response_beats;
  logic response_active;

  tile_prefetch_predictive_axi_top #(
    .MAX_CANDIDATES(2),
    .TILES_PER_EXPERT(1),
    .CHANNEL_COUNT(2),
    .DESCRIPTOR_TABLE_ENTRIES(8),
    .NUM_OCM_BANKS(2),
    .SLOTS_PER_BANK(8),
    .OCM_SLOT_BYTES(512),
    .DMA_FIFO_DEPTH(4),
    .MAX_BURST_BYTES(64),
    .USE_REAL_COMPUTE(1'b0),
    .AUTO_COMPUTE(1'b0),
    .GU_BYTE_RATIO(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd0),
    .sequence_active(1'b1),
    .layer_done(1'b0),
    .prediction_begin_valid(prediction_begin_valid),
    .prediction_begin_ready(prediction_begin_ready),
    .prediction_epoch(16'd21),
    .prediction_sequence(16'd4),
    .prediction_layer(3'd2),
    .prediction_count(5'd1),
    .speculative_budget_bytes(32'd128),
    .channel_credit_bytes({32'd128, 32'd128}),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(10'd3),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd3000),
    .candidate_last(candidate_last),
    .route_begin_valid(route_begin_valid),
    .route_begin_ready(route_begin_ready),
    .route_epoch(16'd21),
    .route_layer(3'd2),
    .route_top_k(5'd1),
    .route_expert_valid(route_expert_valid),
    .route_expert_ready(route_expert_ready),
    .route_expert(10'd3),
    .route_last(route_last),
    .route_weight_q15(16'd32768),
    .gu_base_addr(64'h0000_1000),
    .down_base_addr(64'h0000_2000),
    .expert_stride_bytes(32'd64),
    .physical_tile_bytes(32'd64),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch('0),
    .compute_req_valid(1'b0),
    .compute_req_ready(),
    .compute_gu_descriptor('0),
    .compute_gu_slot_generation('0),
    .compute_down_descriptor('0),
    .compute_down_slot_generation('0),
    .compute_gu_cycles('0),
    .compute_nonlinear_cycles('0),
    .compute_down_cycles('0),
    .compute_busy(),
    .compute_done(),
    .compute_waiting_gu(),
    .compute_waiting_down(),
    .parser_error(parser_error),
    .parser_stale_transaction_count(parser_stale_transaction_count),
    .allocation_active(),
    .allocation_done(),
    .remaining_credit_bytes(),
    .resolver_full(),
    .resolver_dropped_capture_count(),
    .demand_stall_cycles(demand_stall_cycles),
    .stale_event_count(stale_event_count),
    .discarded_dma_write_count(discarded_dma_write_count),
    .payload_discarded_write_count(payload_discarded_write_count),
    .dma_response_error_count(dma_response_error_count),
    .tombstone_skip_count(tombstone_skip_count),
    .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),
    .m_axi_rlast(m_axi_rlast),
    .m_axi_rvalid(m_axi_rvalid),
    .m_axi_rready(m_axi_rready)
  );

  always #5 clk = ~clk;
  assign m_axi_arready = !response_active;
  assign m_axi_rvalid = response_active;
  assign m_axi_rresp = 2'b00;
  assign m_axi_rdata = {112'd0, response_beats[15:0]};
  assign m_axi_rlast = response_beats == 3;

  always @(posedge clk) begin
    if (!rst_n) begin
      accepted_count <= 0;
      response_beats <= 0;
      response_active <= 1'b0;
    end else begin
      if (m_axi_arvalid && m_axi_arready) begin
        if (accepted_count == 0 && m_axi_araddr != 64'h0000_10c0) begin
          $fatal(1, "unexpected GU address %h", m_axi_araddr);
        end
        if (accepted_count == 1 && m_axi_araddr != 64'h0000_20c0) begin
          $fatal(1, "unexpected DOWN address %h", m_axi_araddr);
        end
        if (m_axi_arsize != 3'd4 || m_axi_arburst != 2'b01) begin
          $fatal(1, "AXI burst shape is not 128-bit INCR");
        end
        accepted_count <= accepted_count + 1;
        response_beats <= 0;
        response_active <= 1'b1;
      end
      if (m_axi_rvalid && m_axi_rready) begin
        if (m_axi_rlast) begin
          response_active <= 1'b0;
        end else begin
          response_beats <= response_beats + 1;
        end
      end
    end
  end

  task automatic send_prediction;
    begin
      @(negedge clk);
      prediction_begin_valid = 1'b1;
      while (!prediction_begin_ready) @(negedge clk);
      @(negedge clk);
      prediction_begin_valid = 1'b0;
      candidate_valid = 1'b1;
      candidate_last = 1'b1;
      while (!candidate_ready) @(negedge clk);
      @(negedge clk);
      candidate_valid = 1'b0;
      candidate_last = 1'b0;
    end
  endtask

  task automatic send_route;
    begin
      @(negedge clk);
      route_begin_valid = 1'b1;
      while (!route_begin_ready) @(negedge clk);
      @(negedge clk);
      route_begin_valid = 1'b0;
      route_expert_valid = 1'b1;
      route_last = 1'b1;
      while (!route_expert_ready) @(negedge clk);
      @(negedge clk);
      route_expert_valid = 1'b0;
      route_last = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    prediction_begin_valid = 1'b0;
    candidate_valid = 1'b0;
    candidate_last = 1'b0;
    route_begin_valid = 1'b0;
    route_expert_valid = 1'b0;
    route_last = 1'b0;
    accepted_count = 0;
    response_beats = 0;
    response_active = 1'b0;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    send_prediction();
    repeat (12) @(negedge clk);
    send_route();
    repeat (80) @(negedge clk);
    if (accepted_count != 2 || parser_error || parser_stale_transaction_count != 0) begin
      $fatal(1, "predictive control path failed: accepted=%0d parser_error=%0b parser_stale=%0d active=%0b logical_valid=%0b winner=%0b remaining=%0d channel=%0d candidate_mask=%b", accepted_count, parser_error, parser_stale_transaction_count, dut.scheduler.allocation_active, dut.logical_valid, dut.scheduler.winner_valid, dut.scheduler.remaining_credit_bytes, dut.scheduler.channel_remaining[1], dut.scheduler.candidate_valid_mem);
    end
    if (stale_event_count != 0 || discarded_dma_write_count != 0 ||
        payload_discarded_write_count != 0 || dma_response_error_count != 0) begin
      $fatal(1, "predictive data path reported an ownership or AXI error");
    end
    $display("PASS: parser, tile scheduler, Router resolver and AXI data plane are connected");
    $finish;
  end
endmodule
