`timescale 1ns/1ps
/** Four real F-tile computations with partial prefetch, true miss, late Router,
 * finite 4-slot OCM, AXI response gaps, identity matrices and Q15 Top-2 combine. */
module tb_moe_end_to_end #(
  parameter integer PREDICTED_EXPERT = 3,
  parameter integer ROUTER_DELAY = 150,
  parameter integer BUDGET_BYTES = 768,
  parameter integer FLUSH_DELAY = 0,
  parameter integer AXI_ERROR = 0
);
  logic clk = 0;
  logic rst_n = 0;
  logic pred_begin = 0;
  logic pred_ready;
  logic candidate_valid = 0;
  logic candidate_ready;
  logic route_begin = 0;
  logic route_begin_ready;
  logic route_valid = 0;
  logic route_ready;
  logic [9:0] route_expert;
  logic route_last;
  logic result_valid;
  logic [511:0] result_y;
  logic [63:0] araddr;
  logic [7:0] arlen;
  logic arvalid;
  logic arready;
  logic [127:0] rdata;
  logic rvalid;
  logic rready;
  logic rlast;
  logic response_active = 0;
  integer beats_left = 0;
  integer beat_index = 0;
  integer address_count = 0;
  integer cycle_count = 0;
  logic [63:0] response_address;
  logic parser_error;
  logic [31:0] stale_count;
  logic [31:0] payload_error;
  logic [31:0] axi_error;
  logic [63:0] spec_bytes;
  logic [63:0] demand_bytes;
  logic [63:0] useful_bytes;
  logic [63:0] wasted_bytes;
  logic [63:0] cancelled_bytes;
  logic flush_valid = 0;
  logic [15:0] epoch = 1;
  logic transaction_error;
  integer allocated_spec_bytes = 0;
  integer compute_count = 0;
  always #5 clk = ~clk;

  tile_prefetch_predictive_axi_top #(
    .MAX_CANDIDATES(2),
    .TILES_PER_EXPERT(2),
    .CHANNEL_COUNT(2),
    .DESCRIPTOR_TABLE_ENTRIES(8),
    .NUM_OCM_BANKS(2),
    .SLOTS_PER_BANK(2),
    .OCM_SLOT_BYTES(512),
    .DMA_FIFO_DEPTH(4),
    .MAX_BURST_BYTES(256)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd0),
    .sequence_active(1'b1),
    .layer_done(1'b0),
    .prediction_begin_valid(pred_begin),
    .prediction_begin_ready(pred_ready),
    .prediction_epoch(epoch),
    .prediction_sequence(16'd1),
    .prediction_layer(3'd0),
    .prediction_count(5'd1),
    .speculative_budget_bytes(BUDGET_BYTES),
    .channel_credit_bytes({32'd768, 32'd768}),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(PREDICTED_EXPERT[9:0]),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd3000),
    .candidate_last(1'b1),
    .route_begin_valid(route_begin),
    .route_begin_ready(route_begin_ready),
    .route_epoch(epoch),
    .route_layer(3'd0),
    .route_top_k(5'd2),
    .route_expert_valid(route_valid),
    .route_expert_ready(route_ready),
    .route_expert(route_expert),
    .route_last(route_last),
    .route_weight_q15(16'd16384),
    .gu_base_addr(64'h10000),
    .down_base_addr(64'h20000),
    .expert_stride_bytes(32'd4096),
    .physical_tile_bytes(32'd256),
    .soft_flush_valid(flush_valid),
    .soft_flush_epoch(epoch),
    .compute_req_valid(1'b0),
    .compute_gu_descriptor('0),
    .compute_gu_slot_generation(8'd0),
    .compute_down_descriptor('0),
    .compute_down_slot_generation(8'd0),
    .compute_gu_cycles(32'd0),
    .compute_nonlinear_cycles(32'd0),
    .compute_down_cycles(32'd0),
    .compute_activation({16{8'd16}}),
    .gu_requant_shift(5'd0),
    .result_valid(result_valid),
    .result_y(result_y),
    .parser_error(parser_error),
    .stale_event_count(stale_count),
    .payload_discarded_write_count(payload_error),
    .dma_response_error_count(axi_error),
    .transaction_error(transaction_error),
    .telemetry_speculative_issued_bytes(spec_bytes),
    .telemetry_demand_issued_bytes(demand_bytes),
    .telemetry_useful_speculative_bytes(useful_bytes),
    .telemetry_wasted_speculative_bytes(wasted_bytes),
    .telemetry_cancelled_speculative_bytes(cancelled_bytes),
    .m_axi_araddr(araddr),
    .m_axi_arlen(arlen),
    .m_axi_arvalid(arvalid),
    .m_axi_arready(arready),
    .m_axi_rdata(rdata),
    .m_axi_rresp(AXI_ERROR != 0 && address_count == 1 && beat_index == 0 ? 2'b10 : 2'b00),
    .m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),
    .m_axi_rready(rready)
  );

  assign arready = !response_active && cycle_count % 3 != 0;
  assign rvalid = response_active && cycle_count % 3 != 0;
  assign rlast = beats_left == 1;
  always_comb begin
    rdata = 0;
    rdata[(((response_address/16)+beat_index)%16)*8 +: 8] = 1;
  end
  always @(posedge clk) begin
    if (!rst_n) begin
      response_active <= 0;
      address_count <= 0;
      cycle_count <= 0;
    end else begin
      cycle_count <= cycle_count + 1;
      if (arvalid && arready) begin
        response_active <= 1;
        response_address <= araddr;
        beats_left <= arlen + 1;
        beat_index <= 0;
        address_count <= address_count + 1;
      end
      if (rvalid && rready) begin
        beats_left <= beats_left - 1;
        beat_index <= beat_index + 1;
        if (rlast) response_active <= 0;
      end
      if (parser_error) $fatal(1, "parser error");
      if (dut.compute_done) compute_count <= compute_count + 1;
      if (PREDICTED_EXPERT == 4 && compute_count == 0 && dut.memory_dataflow.backend_gu_start &&
          dut.auto_gu_descriptor.expert_id != 4) $fatal(1, "ready-first must execute the prefetched second Expert first");
      if (dut.physical_accept_valid && !dut.physical_accept_descriptor.request_class)
        allocated_spec_bytes <= allocated_spec_bytes + dut.physical_accept_descriptor.byte_count;
    end
  end

  initial begin
    route_expert = 0;
    route_last = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    pred_begin = 1;
    wait (pred_ready);
    @(negedge clk);
    pred_begin = 0;
    candidate_valid = 1;
    wait (candidate_ready);
    @(negedge clk);
    candidate_valid = 0;
    if (FLUSH_DELAY != 0) begin
      repeat (FLUSH_DELAY) @(negedge clk);
      flush_valid = 1;
      @(negedge clk);
      flush_valid = 0;
      repeat (200) @(negedge clk);
      if (dut.memory_dataflow.core.ocm_occupancy != 0 || dut.memory_dataflow.core.table_free_entries != 8 || result_valid)
        $fatal(1, "flush did not retire old ownership: occupancy=%0d table_free=%0d", dut.memory_dataflow.core.ocm_occupancy, dut.memory_dataflow.core.table_free_entries);
      epoch = 2;
      pred_begin = 1;
      wait (pred_ready);
      @(negedge clk);
      pred_begin = 0;
      candidate_valid = 1;
      wait (candidate_ready);
      @(negedge clk);
      candidate_valid = 0;
    end
    repeat (ROUTER_DELAY) @(negedge clk);
    if (AXI_ERROR != 0) begin
      if (!transaction_error || result_valid || axi_error != 1)
        $fatal(1, "AXI response failure must suppress numerical result and signal transaction_error");
      flush_valid = 1;
      @(negedge clk);
      flush_valid = 0;
      repeat (200) @(negedge clk);
      if (dut.memory_dataflow.core.ocm_occupancy != 0 || transaction_error)
        $fatal(1, "AXI error flush failed to reclaim resources");
      $display("PASS: AXI response error suppresses output and flush recovers ownership");
      $finish;
    end
    if (FLUSH_DELAY == 0 && ROUTER_DELAY >= 150 && BUDGET_BYTES != 0 && address_count != 3)
      $fatal(1, "prefetch should transfer one GU/DOWN pair before truth");
    if (FLUSH_DELAY == 0 && ROUTER_DELAY >= 150 && BUDGET_BYTES != 0 && (useful_bytes != 0 || wasted_bytes != 0))
      $fatal(1, "DMA-before-truth bytes must remain unclassified");
    route_begin = 1;
    wait (route_begin_ready);
    @(negedge clk);
    route_begin = 0;
    route_valid = 1;
    route_expert = 3;
    route_last = 0;
    wait (route_ready);
    @(negedge clk);
    route_expert = 4;
    route_last = 1;
    @(negedge clk);
    route_valid = 0;
    route_last = 0;
    wait (result_valid);
    @(negedge clk);
    for (integer row = 0; row < 16; row = row + 1)
      if ($signed(result_y[row*32 +: 32]) != 24)
        $fatal(1, "Top-2 combine row=%0d expected=24 actual=%0d", row, $signed(result_y[row*32 +: 32]));
    // Faster critical work can finish before wrong speculative DMA retires.
    // Numerical latency is measured at result_valid; byte conservation at drain.
    wait(dut.memory_dataflow.memory_drained);
    repeat (10) @(negedge clk);
    if (demand_bytes != ((BUDGET_BYTES != 0 && (PREDICTED_EXPERT == 3 || PREDICTED_EXPERT == 4)) ? 2304 : 3072))
      $fatal(1, "demand byte conservation failed: %0d", demand_bytes);
    if (useful_bytes + wasted_bytes != spec_bytes ||
        useful_bytes + wasted_bytes + cancelled_bytes != allocated_spec_bytes)
      $fatal(1, "speculative classification/conservation failed issued=%0d useful=%0d wasted=%0d cancelled=%0d",
        spec_bytes, useful_bytes, wasted_bytes, cancelled_bytes);
    if (compute_count != 4 || stale_count != 0 || (FLUSH_DELAY == 0 && payload_error != 0) || axi_error != 0)
      $fatal(1, "conservation/ownership failure bursts=%0d stale=%0d payload=%0d axi=%0d", address_count, stale_count, payload_error, axi_error);
    $display("PASS: full AXI/OCM/int8 SwiGLU/demand fallback/Top-2 combine numerical path");
    $finish;
  end
  initial begin
    #50000;
    $fatal(1, "end-to-end timeout bursts=%0d compute_phase=%0d backend_phase=%0d gu_state=%0d down_state=%0d expert=%0d tile=%0d running=%0b", address_count,
      dut.memory_dataflow.core.compute.phase, dut.memory_dataflow.real_compute.backend.phase,
      dut.memory_dataflow.core.compute_gu_state, dut.memory_dataflow.core.compute_down_state,
      dut.automatic_execution.execution.expert_index, dut.automatic_execution.execution.tile_index,
      dut.automatic_execution.execution.running);
  end
endmodule
