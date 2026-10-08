`timescale 1ns/1ps
/** Self-checking AXI read/payload integration test for the hardware top. */
module tb_tile_prefetch_axi_top;
  import tile_prefetch_package::*;
  logic clk = 0;
  logic rst_n = 0;
  logic bandwidth_enable = 1;
  logic logical_valid;
  logic logical_ready;
  tile_descriptor_t logical_gu_descriptor;
  tile_descriptor_t logical_down_descriptor;
  logic physical_accept_valid;
  descriptor_token_t physical_accept_token;
  tile_descriptor_t physical_accept_descriptor;
  logic [7:0] physical_accept_slot_generation;
  logic m_axi_arvalid;
  logic m_axi_arready = 1;
  logic [63:0] m_axi_araddr;
  logic [7:0] m_axi_arlen;
  logic [2:0] m_axi_arsize;
  logic [1:0] m_axi_arburst;
  logic [127:0] m_axi_rdata;
  logic [1:0] m_axi_rresp;
  logic m_axi_rlast;
  logic m_axi_rvalid;
  logic m_axi_rready;
  logic read_active;
  logic [8:0] read_beats_left;
  logic [63:0] issued_address [0:3];
  integer issued_count;
  integer accepted_count;
  logic [31:0] payload_discarded;
  logic [31:0] dma_response_errors;
  logic [31:0] stale_events;
  logic [31:0] dma_discards;

  always #5 clk = ~clk;

  always_comb begin
    m_axi_rvalid = read_active;
    m_axi_rlast = read_active && read_beats_left == 1;
    m_axi_rresp = 2'b00;
    m_axi_rdata = {64'h0123_4567_89ab_cdef, 64'hfedc_ba98_7654_3210};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      read_active <= 1'b0;
      read_beats_left <= '0;
      issued_count <= 0;
      accepted_count <= 0;
    end else begin
      if (physical_accept_valid) begin
        accepted_count <= accepted_count + 1;
      end
      if (m_axi_arvalid && m_axi_arready) begin
        read_active <= 1'b1;
        read_beats_left <= m_axi_arlen + 1'b1;
        issued_address[issued_count] <= m_axi_araddr;
        issued_count <= issued_count + 1;
      end
      if (m_axi_rvalid && m_axi_rready) begin
        if (read_beats_left == 1) begin
          read_active <= 1'b0;
        end else begin
          read_beats_left <= read_beats_left - 1'b1;
        end
      end
    end
  end

  tile_prefetch_axi_top #(
    .DESCRIPTOR_TABLE_ENTRIES(8),
    .NUM_OCM_BANKS(4),
    .SLOTS_PER_BANK(4),
    .OCM_SLOT_BYTES(256),
    .DMA_FIFO_DEPTH(4),
    .MAX_BURST_BYTES(64)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
    .setup_cycles(0),
    .sequence_active(1'b1),
    .layer_done(1'b0),
    .scheduler_active(1'b0),
    .logical_valid(logical_valid),
    .logical_gu_descriptor(logical_gu_descriptor),
    .logical_down_descriptor(logical_down_descriptor),
    .existing_pair_live(1'b0),
    .existing_pair_is_speculative(1'b0),
    .logical_ready(logical_ready),
    .physical_accept_valid(physical_accept_valid),
    .physical_accept_token(physical_accept_token),
    .physical_accept_descriptor(physical_accept_descriptor),
    .physical_accept_slot_generation(physical_accept_slot_generation),
    .route_valid(1'b0),
    .route_token('0),
    .route_truth(1'b0),
    .route_ready(),
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
    .demand_stall_cycles(),
    .stale_event_count(stale_events),
    .discarded_dma_write_count(dma_discards),
    .payload_discarded_write_count(payload_discarded),
    .dma_response_error_count(dma_response_errors),
    .tombstone_skip_count(),
    .telemetry_total_cycles(),
    .telemetry_speculative_allocated_bytes(),
    .telemetry_speculative_issued_bytes(),
    .telemetry_demand_issued_bytes(),
    .telemetry_compute_wait_gu_cycles(),
    .telemetry_compute_wait_down_cycles(),
    .telemetry_ocm_peak_occupancy(),
    .telemetry_commit_count(),
    .telemetry_cancel_count(),
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

  function automatic tile_descriptor_t descriptor(
    input matrix_kind_t matrix_kind,
    input logic [63:0] external_addr
  );
    tile_descriptor_t value;
    begin
      value = '0;
      value.epoch_id = 1;
      value.layer_id = 1;
      value.expert_id = 3;
      value.tile_id = 0;
      value.matrix_kind = matrix_kind;
      value.external_addr = external_addr;
      value.byte_count = 64;
      value.request_class = REQUEST_DEMAND;
      descriptor = value;
    end
  endfunction

  initial begin
    logical_valid = 1'b0;
    logical_gu_descriptor = '0;
    logical_down_descriptor = '0;
    #17;
    rst_n = 1'b1;
    @(negedge clk);
    while (!logical_ready) begin
      @(negedge clk);
    end
    logical_gu_descriptor = descriptor(MATRIX_GU, 64'h0000_0000_1000_0000);
    logical_down_descriptor = descriptor(MATRIX_DOWN, 64'h0000_0000_2000_0000);
    logical_valid = 1'b1;
    @(negedge clk);
    logical_valid = 1'b0;
    wait (issued_count == 2);
    repeat (6) @(negedge clk);
    if (issued_address[0] != 64'h0000_0000_1000_0000) begin
      $fatal(1, "GU AXI address mismatch");
    end
    if (issued_address[1] != 64'h0000_0000_2000_0000) begin
      $fatal(1, "DOWN AXI address mismatch");
    end
    if (m_axi_arsize != 3'd4 || m_axi_arburst != 2'b01) begin
      $fatal(1, "AXI payload ABI is not 128-bit INCR");
    end
    if (payload_discarded != 0 || dma_response_errors != 0 ||
        stale_events != 0 || dma_discards != 0) begin
      $fatal(1, "AXI payload path rejected a valid transaction");
    end
    $display("PASS: AXI read adapter and OCM payload data path");
    $finish;
  end

  initial begin
    #5000;
    $fatal(1, "AXI integration timeout: accepted=%0d issued=%0d",
      accepted_count, issued_count);
  end
endmodule
