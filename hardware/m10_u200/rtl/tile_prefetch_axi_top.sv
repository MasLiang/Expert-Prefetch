`timescale 1ns/1ps
/**
 * 连接真实 AXI4 读数据面和 OCM payload RAM 的 platform-neutral 顶层。
 *
 * 该模块不绑定 U200 XPFM：板级 wrapper 只需把 m_axi_* 接至已验证的 DDR
 * interconnect。仲裁、descriptor token、OCM ownership 与 AXI response 的
 * completion path 都在此顶层闭合。
 */
module tile_prefetch_axi_top #(
  parameter integer DESCRIPTOR_TABLE_ENTRIES = 32,
  parameter integer NUM_OCM_BANKS = 4,
  parameter integer SLOTS_PER_BANK = 32,
  parameter integer OCM_SLOT_BYTES = 4096,
  parameter integer DMA_FIFO_DEPTH = 8,
  parameter integer MAX_BURST_BYTES = 256,
  parameter integer DEMAND_RESERVE_SLOTS = 2,
  parameter integer GENERATION_W = 8,
  parameter integer AXI_ADDR_W = 64,
  parameter integer AXI_DATA_W = 128,
  parameter bit USE_REAL_COMPUTE = 1'b0,
  parameter bit SWITCH_RELU = 1'b0,
  parameter integer H = 16,
  parameter integer F_TILE = 16,
  parameter integer NUM_ROUTER_EXPERTS = 64,
  parameter bit STRIPED_DDR = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic bandwidth_enable,
  input logic [31:0] setup_cycles,
  input logic sequence_active,
  input logic layer_done,
  input logic scheduler_active,
  input logic logical_valid,
  input tile_prefetch_package::tile_descriptor_t logical_gu_descriptor,
  input tile_prefetch_package::tile_descriptor_t logical_down_descriptor,
  input logic existing_pair_live,
  input logic existing_pair_is_speculative,
  output logic logical_ready,
  output logic physical_accept_valid,
  output tile_prefetch_package::descriptor_token_t physical_accept_token,
  output tile_prefetch_package::tile_descriptor_t physical_accept_descriptor,
  output logic [GENERATION_W-1:0] physical_accept_slot_generation,
  input logic route_valid,
  input tile_prefetch_package::descriptor_token_t route_token,
  input logic route_truth,
  output logic route_ready,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic compute_req_valid,
  output logic compute_req_ready,
  input tile_prefetch_package::tile_descriptor_t compute_gu_descriptor,
  input logic [GENERATION_W-1:0] compute_gu_slot_generation,
  input tile_prefetch_package::tile_descriptor_t compute_down_descriptor,
  input logic [GENERATION_W-1:0] compute_down_slot_generation,
  input logic [31:0] compute_gu_cycles,
  input logic [31:0] compute_nonlinear_cycles,
  input logic [31:0] compute_down_cycles,
  input logic [H*8-1:0] compute_activation,
  input logic [4:0] gu_requant_shift,
  input logic router_start,
  output logic router_done,
  output logic router_read_valid,
  output logic [31:0] router_read_offset,
  input logic [127:0] router_read_data,
  output logic [NUM_ROUTER_EXPERTS*32-1:0] router_logits,
  output logic [H*32-1:0] compute_partial_y,
  output logic dma_completion_valid,
  output tile_prefetch_package::tile_descriptor_t dma_completion_descriptor,
  output logic compute_busy,
  output logic memory_drained,
  output logic memory_ingress_idle,
  output logic compute_done,
  output logic compute_gu_release_valid,
  output logic compute_waiting_gu,
  output logic compute_waiting_down,
  output logic [31:0] demand_stall_cycles,
  output logic [31:0] stale_event_count,
  output logic [31:0] discarded_dma_write_count,
  output logic [31:0] payload_discarded_write_count,
  output logic [31:0] dma_response_error_count,
  output logic [31:0] tombstone_skip_count,
  output logic [63:0] telemetry_total_cycles,
  output logic [63:0] telemetry_speculative_allocated_bytes,
  output logic [63:0] telemetry_speculative_issued_bytes,
  output logic [63:0] telemetry_demand_issued_bytes,
  output logic [63:0] telemetry_compute_wait_gu_cycles,
  output logic [63:0] telemetry_compute_wait_down_cycles,
  output logic [63:0] telemetry_ocm_peak_occupancy,
  output logic [63:0] telemetry_commit_count,
  output logic [63:0] telemetry_cancel_count,
  output logic [63:0] telemetry_useful_speculative_bytes,
  output logic [63:0] telemetry_wasted_speculative_bytes,
  output logic [63:0] telemetry_cancelled_speculative_bytes,
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0] m_axi_arlen,
  output logic [2:0] m_axi_arsize,
  output logic [1:0] m_axi_arburst,
  output logic m_axi_arvalid,
  input logic m_axi_arready,
  input logic [AXI_DATA_W-1:0] m_axi_rdata,
  input logic [1:0] m_axi_rresp,
  input logic m_axi_rlast,
  input logic m_axi_rvalid,
  output logic m_axi_rready,
  output logic [4*AXI_ADDR_W-1:0] m_ddr_araddr,
  output logic [31:0] m_ddr_arlen,
  output logic [11:0] m_ddr_arsize,
  output logic [7:0] m_ddr_arburst,
  output logic [3:0] m_ddr_arvalid,
  input logic [3:0] m_ddr_arready,
  input logic [511:0] m_ddr_rdata,
  input logic [7:0] m_ddr_rresp,
  input logic [3:0] m_ddr_rlast,
  input logic [3:0] m_ddr_rvalid,
  output logic [3:0] m_ddr_rready
);
  import tile_prefetch_package::*;
  localparam integer PAYLOAD_W = STRIPED_DDR ? 512 : AXI_DATA_W;
  localparam integer SLOT_COUNT = NUM_OCM_BANKS * SLOTS_PER_BANK;
  localparam integer SLOT_W = (SLOT_COUNT <= 1) ? 1 : $clog2(SLOT_COUNT);

  logic dma_burst_start;
  logic dma_burst_done;
  logic [31:0] dma_burst_bytes;
  logic dma_done;
  tile_descriptor_t dma_active_descriptor;
  descriptor_token_t dma_active_token;
  logic [GENERATION_W-1:0] dma_active_slot_generation;
  logic [31:0] dma_active_remaining_bytes;
  logic adapter_burst_complete;
  logic adapter_busy;
  logic core_memory_drained;
  assign memory_drained = core_memory_drained && !adapter_busy;
  logic payload_valid;
  logic [15:0] payload_slot;
  logic [15:0] payload_epoch;
  logic [GENERATION_W-1:0] payload_slot_generation;
  logic [31:0] payload_offset;
  logic [PAYLOAD_W-1:0] payload_data;
  logic [PAYLOAD_W/8-1:0] payload_keep;
  logic payload_last;
  logic logical_promotion_unused;
  assign dma_completion_valid = dma_done;
  assign dma_completion_descriptor = dma_active_descriptor;
  logic backend_gu_start;
  logic backend_down_start;
  logic backend_gu_done;
  logic backend_nonlinear_done;
  logic backend_down_done;
  logic backend_read_valid;
  logic [15:0] backend_read_slot;
  logic [31:0] backend_read_offset;
  logic [127:0] backend_read_data;

  tile_prefetch_top #(
    .DESCRIPTOR_TABLE_ENTRIES(DESCRIPTOR_TABLE_ENTRIES),
    .NUM_OCM_BANKS(NUM_OCM_BANKS),
    .SLOTS_PER_BANK(SLOTS_PER_BANK),
    .DMA_FIFO_DEPTH(DMA_FIFO_DEPTH),
    .MAX_BURST_BYTES(MAX_BURST_BYTES),
    .DEMAND_RESERVE_SLOTS(DEMAND_RESERVE_SLOTS),
    .GENERATION_W(GENERATION_W),
    .USE_EXTERNAL_DMA(1'b1),
    .USE_REAL_COMPUTE(USE_REAL_COMPUTE)
  ) core (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
    .setup_cycles(setup_cycles),
    .bytes_per_cycle(32'd1),
    .dma_external_burst_done(adapter_burst_complete),
    .sequence_active(sequence_active),
    .layer_done(layer_done),
    .scheduler_active(scheduler_active),
    .logical_valid(logical_valid),
    .logical_gu_descriptor(logical_gu_descriptor),
    .logical_down_descriptor(logical_down_descriptor),
    .existing_pair_live(existing_pair_live),
    .existing_pair_is_speculative(existing_pair_is_speculative),
    .logical_ready(logical_ready),
    .logical_promotion_valid(logical_promotion_unused),
    .physical_accept_valid(physical_accept_valid),
    .physical_accept_token(physical_accept_token),
    .physical_accept_descriptor(physical_accept_descriptor),
    .physical_accept_slot_generation(physical_accept_slot_generation),
    .route_valid(route_valid),
    .route_token(route_token),
    .route_truth(route_truth),
    .route_ready(route_ready),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(compute_req_ready),
    .compute_gu_descriptor(compute_gu_descriptor),
    .compute_gu_slot_generation(compute_gu_slot_generation),
    .compute_down_descriptor(compute_down_descriptor),
    .compute_down_slot_generation(compute_down_slot_generation),
    .compute_gu_cycles(compute_gu_cycles),
    .compute_nonlinear_cycles(compute_nonlinear_cycles),
    .compute_down_cycles(compute_down_cycles),
    .backend_gu_done(backend_gu_done),
    .backend_nonlinear_done(backend_nonlinear_done),
    .backend_down_done(backend_down_done),
    .backend_gu_start(backend_gu_start),
    .backend_down_start(backend_down_start),
    .compute_busy(compute_busy),
    .memory_drained(core_memory_drained),
    .memory_ingress_idle(memory_ingress_idle),
    .compute_done(compute_done),
    .compute_gu_release_valid(compute_gu_release_valid),
    .compute_waiting_gu(compute_waiting_gu),
    .compute_waiting_down(compute_waiting_down),
    .dma_burst_start(dma_burst_start),
    .dma_burst_done(dma_burst_done),
    .dma_burst_bytes(dma_burst_bytes),
    .dma_done(dma_done),
    .dma_active_descriptor(dma_active_descriptor),
    .dma_active_token(dma_active_token),
    .dma_active_slot_generation(dma_active_slot_generation),
    .dma_active_remaining_bytes(dma_active_remaining_bytes),
    .demand_stall_cycles(demand_stall_cycles),
    .stale_event_count(stale_event_count),
    .discarded_dma_write_count(discarded_dma_write_count),
    .tombstone_skip_count(tombstone_skip_count),
    .telemetry_total_cycles(telemetry_total_cycles),
    .telemetry_speculative_allocated_bytes(telemetry_speculative_allocated_bytes),
    .telemetry_speculative_issued_bytes(telemetry_speculative_issued_bytes),
    .telemetry_demand_issued_bytes(telemetry_demand_issued_bytes),
    .telemetry_compute_wait_gu_cycles(telemetry_compute_wait_gu_cycles),
    .telemetry_compute_wait_down_cycles(telemetry_compute_wait_down_cycles),
    .telemetry_ocm_peak_occupancy(telemetry_ocm_peak_occupancy),
    .telemetry_commit_count(telemetry_commit_count),
    .telemetry_cancel_count(telemetry_cancel_count),
    .telemetry_useful_speculative_bytes(telemetry_useful_speculative_bytes),
    .telemetry_wasted_speculative_bytes(telemetry_wasted_speculative_bytes),
    .telemetry_cancelled_speculative_bytes(telemetry_cancelled_speculative_bytes)
  );

  generate if (STRIPED_DDR) begin : four_bank_dma
    assign m_axi_araddr = 0;
    assign m_axi_arlen = 0;
    assign m_axi_arsize = 0;
    assign m_axi_arburst = 0;
    assign m_axi_arvalid = 0;
    assign m_axi_rready = 0;
  striped_ddr_read_adapter #(
    .AXI_ADDR_W(AXI_ADDR_W),
    .GENERATION_W(GENERATION_W)
  ) dma_adapter (
    .clk(clk),
    .rst_n(rst_n),
    .burst_start(dma_burst_start),
    .active_descriptor(dma_active_descriptor),
    .active_slot_generation(dma_active_slot_generation),
    .burst_bytes(dma_burst_bytes),
    .active_remaining_bytes(dma_active_remaining_bytes),
    .burst_complete(adapter_burst_complete),
    .busy(adapter_busy),
    .payload_valid(payload_valid),
    .payload_ready(1'b1),
    .payload_slot(payload_slot),
    .payload_epoch(payload_epoch),
    .payload_slot_generation(payload_slot_generation),
    .payload_offset(payload_offset),
    .payload_data(payload_data),
    .payload_keep(payload_keep),
    .payload_last(payload_last),
    .response_error_count(dma_response_error_count),
    .m_ddr_araddr(m_ddr_araddr),
    .m_ddr_arlen(m_ddr_arlen),
    .m_ddr_arsize(m_ddr_arsize),
    .m_ddr_arburst(m_ddr_arburst),
    .m_ddr_arvalid(m_ddr_arvalid),
    .m_ddr_arready(m_ddr_arready),
    .m_ddr_rdata(m_ddr_rdata),
    .m_ddr_rresp(m_ddr_rresp),
    .m_ddr_rlast(m_ddr_rlast),
    .m_ddr_rvalid(m_ddr_rvalid),
    .m_ddr_rready(m_ddr_rready)
  );

  end else begin : single_bank_dma
    assign m_ddr_araddr = 0;
    assign m_ddr_arlen = 0;
    assign m_ddr_arsize = 0;
    assign m_ddr_arburst = 0;
    assign m_ddr_arvalid = 0;
    assign m_ddr_rready = 0;
  axi_read_adapter #(
    .AXI_ADDR_W(AXI_ADDR_W),
    .AXI_DATA_W(AXI_DATA_W),
    .GENERATION_W(GENERATION_W)
  ) dma_adapter (
    .clk(clk),
    .rst_n(rst_n),
    .burst_start(dma_burst_start),
    .active_descriptor(dma_active_descriptor),
    .active_slot_generation(dma_active_slot_generation),
    .burst_bytes(dma_burst_bytes),
    .active_remaining_bytes(dma_active_remaining_bytes),
    .burst_complete(adapter_burst_complete),
    .busy(adapter_busy),
    .payload_valid(payload_valid),
    .payload_ready(1'b1),
    .payload_slot(payload_slot),
    .payload_epoch(payload_epoch),
    .payload_slot_generation(payload_slot_generation),
    .payload_offset(payload_offset),
    .payload_data(payload_data),
    .payload_keep(payload_keep),
    .payload_last(payload_last),
    .response_error_count(dma_response_error_count),
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

  end endgenerate

  ocm_payload_ram #(
    .SLOT_COUNT(SLOT_COUNT),
    .BYTES_PER_SLOT(OCM_SLOT_BYTES),
    .DATA_W(PAYLOAD_W),
    .READ_DATA_W(128),
    .GENERATION_W(GENERATION_W)
  ) payload_ram (
    .clk(clk),
    .rst_n(rst_n),
    .slot_bind_valid(physical_accept_valid),
    .slot_bind_slot(physical_accept_descriptor.ocm_slot[SLOT_W-1:0]),
    .slot_bind_epoch(physical_accept_descriptor.epoch_id),
    .slot_bind_generation(physical_accept_slot_generation),
    .slot_release_valid(compute_gu_release_valid),
    .slot_release_slot(compute_gu_descriptor.ocm_slot[SLOT_W-1:0]),
    .slot_release_epoch(compute_gu_descriptor.epoch_id),
    .slot_release_generation(compute_gu_slot_generation),
    .slot_release2_valid(compute_done),
    .slot_release2_slot(compute_down_descriptor.ocm_slot[SLOT_W-1:0]),
    .slot_release2_epoch(compute_down_descriptor.epoch_id),
    .slot_release2_generation(compute_down_slot_generation),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .dma_write_valid(payload_valid),
    .dma_write_slot(payload_slot[SLOT_W-1:0]),
    .dma_write_offset(payload_offset[$clog2(OCM_SLOT_BYTES)-1:0]),
    .dma_write_data(payload_data),
    .dma_write_strobe(payload_keep),
    .dma_write_epoch(payload_epoch),
    .dma_write_generation(payload_slot_generation),
    .dma_write_accept(),
    .read_valid(backend_read_valid),
    .read_slot(backend_read_slot[SLOT_W-1:0]),
    .read_offset(backend_read_offset[$clog2(OCM_SLOT_BYTES)-1:0]),
    .read_data(backend_read_data),
    .discarded_write_count(payload_discarded_write_count)
  );

  generate if (USE_REAL_COMPUTE) begin : real_compute
    int8_swiglu_tile_backend #(
      .H(H),
      .F_TILE(F_TILE),
      .SWITCH_RELU(SWITCH_RELU),
      .NUM_ROUTER_EXPERTS(NUM_ROUTER_EXPERTS)
    ) backend (
      .clk(clk),
      .rst_n(rst_n),
      .clear(soft_flush_valid),
      .router_start(router_start),
      .router_done(router_done),
      .router_read_valid(router_read_valid),
      .router_read_offset(router_read_offset),
      .router_read_data(router_read_data),
      .router_logits(router_logits),
      .gu_start(backend_gu_start),
      .down_start(backend_down_start),
      .activation(compute_activation),
      .gu_requant_shift(gu_requant_shift),
      .gu_slot(compute_gu_descriptor.ocm_slot),
      .down_slot(compute_down_descriptor.ocm_slot),
      .read_valid(backend_read_valid),
      .read_slot(backend_read_slot),
      .read_offset(backend_read_offset),
      .read_data(backend_read_data),
      .gu_done(backend_gu_done),
      .nonlinear_done(backend_nonlinear_done),
      .down_done(backend_down_done),
      .partial_y(compute_partial_y)
    );
  end else begin : timing_compute
    assign router_done = 0;
    assign router_read_valid = 0;
    assign router_read_offset = 0;
    assign router_logits = 0;
    assign backend_read_valid = 0;
    assign backend_read_slot = 0;
    assign backend_read_offset = 0;
    assign backend_gu_done = 0;
    assign backend_nonlinear_done = 0;
    assign backend_down_done = 0;
    assign compute_partial_y = 0;
  end endgenerate
endmodule
