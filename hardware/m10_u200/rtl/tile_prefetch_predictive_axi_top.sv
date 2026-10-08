`timescale 1ns/1ps
/**
 * Prediction transaction、tile allocator、Router truth 与 AXI/OCM data plane 的
 * prefetch 与数值执行前端。candidate allocator 与 true-tile demander
 * 共用 logical descriptor admission；Router 的 commit/cancel 经 generation-safe
 * token path 返回 memory core，避免携带可变 descriptor copy。
 */
module tile_prefetch_predictive_axi_top #(
  parameter bit ENABLE_CROSS_LAYER = 0,
  parameter integer LAYER_COUNT = 8,
  parameter integer MAX_CANDIDATES = 16,
  parameter integer NUM_EXPERTS = 64,
  parameter integer TILES_PER_EXPERT = 16,
  parameter integer CHANNEL_COUNT = 4,
  parameter integer DESCRIPTOR_TABLE_ENTRIES = 64,
  parameter integer NUM_OCM_BANKS = 4,
  parameter integer SLOTS_PER_BANK = 32,
  parameter integer OCM_SLOT_BYTES = 4096,
  parameter integer DMA_FIFO_DEPTH = 16,
  parameter integer MAX_BURST_BYTES = 256,
  parameter integer GENERATION_W = 8,
  parameter integer AXI_ADDR_W = 64,
  parameter integer AXI_DATA_W = 128,
  parameter bit USE_REAL_COMPUTE = 1'b1,
  parameter bit USE_REAL_ROUTER = 1'b0,
  parameter bit STRIPED_DDR = 1'b0,
  parameter bit AUTO_COMPUTE = 1'b1,
  parameter bit WAIT_FULL_EXPERT = 1'b0,
  parameter bit USE_SEQUENCE_CONTROL = 1'b0,
  parameter bit SWITCH_RELU = 1'b0,
  parameter integer GU_BYTE_RATIO = SWITCH_RELU ? 1 : 2,
  parameter integer H = 16,
  parameter integer F_TILE = 16
) (
  input logic clk,
  input logic rst_n,
  input logic bandwidth_enable,
  input logic [31:0] setup_cycles,
  input logic sequence_active,
  input logic sequence_start,
  input logic [31:0] compute_block_cycles,
  input logic [31:0] bandwidth_block_cycles,
  output logic sequence_done,
  output logic [2:0] current_layer,
  input logic layer_done,
  input logic prediction_begin_valid,
  output logic prediction_begin_ready,
  input logic [15:0] prediction_epoch,
  input logic [15:0] prediction_sequence,
  input logic [2:0] prediction_layer,
  input logic [4:0] prediction_count,
  input logic [31:0] speculative_budget_bytes,
  input logic [CHANNEL_COUNT*32-1:0] channel_credit_bytes,
  input logic candidate_valid,
  output logic candidate_ready,
  input logic [9:0] candidate_expert,
  input logic [4:0] candidate_rank,
  input logic [11:0] candidate_confidence,
  input logic candidate_last,
  input logic route_begin_valid,
  output logic route_begin_ready,
  input logic [15:0] route_epoch,
  input logic [2:0] route_layer,
  input logic [4:0] route_top_k,
  input logic route_expert_valid,
  output logic route_expert_ready,
  input logic [9:0] route_expert,
  input logic route_last,
  input logic [15:0] route_weight_q15,
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic [31:0] physical_tile_bytes,
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
  input logic [4:0] router_top_k_config,
  input logic [4:0] router_logit_shift,
  input logic router_weight_reload_valid,
  output logic router_weight_reload_ready,
  input logic [2:0] router_weight_layer,
  input logic router_weight_valid,
  output logic router_weight_ready,
  input logic [31:0] router_weight_word_index,
  input logic [127:0] router_weight_data,
  input logic router_weight_last,
  output logic router_weights_loaded,
  output logic router_weight_error,
  output logic router_gemm_done,
  output logic [H*32-1:0] compute_partial_y,
  output logic result_valid,
  output logic [H*32-1:0] result_y,
  output logic physical_accept_valid,
  output tile_prefetch_package::descriptor_token_t physical_accept_token,
  output tile_prefetch_package::tile_descriptor_t physical_accept_descriptor,
  output logic [GENERATION_W-1:0] physical_accept_slot_generation,
  output logic compute_busy,
  output logic transaction_drained,
  output logic context_retire_ready,
  input logic context_switch_pending,
  input logic clear_sequence,
  input logic future_logical_valid,
  output logic future_logical_ready,
  input tile_prefetch_package::tile_descriptor_t future_gu_descriptor,
  input tile_prefetch_package::tile_descriptor_t future_down_descriptor,
  output wire [LAYER_COUNT-1:0] layer_weights_done,
  output logic compute_done,
  output logic compute_waiting_gu,
  output logic compute_waiting_down,
  output logic parser_error,
  output logic config_error,
  output logic transaction_error,
  output logic [31:0] parser_stale_transaction_count,
  output logic allocation_active,
  output logic allocation_done,
  output logic [31:0] remaining_credit_bytes,
  output logic resolver_full,
  output logic [31:0] resolver_dropped_capture_count,
  output logic [31:0] demand_stall_cycles,
  output logic [31:0] stale_event_count,
  output logic [31:0] discarded_dma_write_count,
  output logic [31:0] payload_discarded_write_count,
  output logic [31:0] dma_response_error_count,
  output logic [31:0] tombstone_skip_count,
  output logic [63:0] telemetry_speculative_issued_bytes,
  output logic [63:0] telemetry_demand_issued_bytes,
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

  logic workload_busy;
  logic compute_gu_release_valid;
  logic sequence_compute_ready;
  logic memory_drained;
  logic memory_ingress_idle;
  logic [15:0] active_epoch;
  logic [2:0] active_layer;
  logic effective_route_begin_valid;
  logic effective_route_begin_ready;
  logic [15:0] effective_route_epoch;
  logic [2:0] effective_route_layer;
  logic [4:0] effective_route_top_k;
  logic effective_route_expert_valid;
  logic effective_route_expert_ready;
  logic [9:0] effective_route_expert;
  logic effective_route_last;
  logic [15:0] effective_route_weight_q15;
  logic router_start_pending;
  logic router_start_pulse;
  logic router_read_valid;
  logic [31:0] router_read_offset;
  logic [127:0] router_read_data;
  logic [NUM_EXPERTS*32-1:0] router_logits;
  logic [2:0] router_loaded_layer;
  logic [4:0] router_k_hold;
  logic [4:0] router_shift_hold;
  assign router_start_pulse = USE_REAL_ROUTER && router_start_pending &&
    sequence_compute_ready && !soft_flush_valid && !transaction_error;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) router_start_pending <= 0;
    else if (soft_flush_valid || router_start_pulse) router_start_pending <= 0;
    else if (prediction_begin_valid && prediction_begin_ready) router_start_pending <= USE_REAL_ROUTER;
  end
  always_ff @(posedge clk) if (prediction_begin_valid && prediction_begin_ready) router_k_hold <= router_top_k_config;
  always_ff @(posedge clk) if (prediction_begin_valid && prediction_begin_ready) router_shift_hold <= router_logit_shift;
  generate if (USE_REAL_ROUTER) begin : real_router_control
    router_weight_store #(
      .H(H),
      .NUM_EXPERTS(NUM_EXPERTS)
    ) weights (
      .clk(clk),
      .rst_n(rst_n),
      .host_enable(!workload_busy && !prediction_begin_valid),
      .reload_valid(router_weight_reload_valid),
      .reload_ready(router_weight_reload_ready),
      .reload_layer(router_weight_layer),
      .write_valid(router_weight_valid),
      .write_ready(router_weight_ready),
      .write_word_index(router_weight_word_index),
      .write_data(router_weight_data),
      .write_last(router_weight_last),
      .loaded(router_weights_loaded),
      .loaded_layer(router_loaded_layer),
      .load_error(router_weight_error),
      .read_valid(router_read_valid),
      .read_offset(router_read_offset),
      .read_data(router_read_data)
    );
    router_topk_softmax #(
      .NUM_EXPERTS(NUM_EXPERTS)
    ) selection (
      .clk(clk),
      .rst_n(rst_n),
      .flush(soft_flush_valid),
      .start(router_gemm_done),
      .logits(router_logits),
      .top_k(router_k_hold),
      .logit_shift(router_shift_hold),
      .busy(),
      .begin_valid(effective_route_begin_valid),
      .begin_ready(effective_route_begin_ready),
      .expert_valid(effective_route_expert_valid),
      .expert_ready(effective_route_expert_ready),
      .expert_id(effective_route_expert),
      .weight_q15(effective_route_weight_q15),
      .last(effective_route_last)
    );
    assign effective_route_epoch = active_epoch;
    assign effective_route_layer = active_layer;
    assign effective_route_top_k = router_k_hold;
    assign route_begin_ready = 0;
    assign route_expert_ready = 0;
  end else begin : external_router_control
    assign router_weight_reload_ready = 0;
    assign router_weight_ready = 0;
    assign router_weights_loaded = 0;
    assign router_weight_error = 0;
    assign router_loaded_layer = 0;
    assign router_read_data = 0;
    assign effective_route_begin_valid = route_begin_valid;
    assign effective_route_epoch = route_epoch;
    assign effective_route_layer = route_layer;
    assign effective_route_top_k = route_top_k;
    assign effective_route_expert_valid = route_expert_valid;
    assign effective_route_expert = route_expert;
    assign effective_route_last = route_last;
    assign effective_route_weight_q15 = route_weight_q15;
    assign route_begin_ready = effective_route_begin_ready;
    assign route_expert_ready = effective_route_expert_ready;
  end endgenerate


  logic candidate_accept;
  logic [15:0] candidate_accept_epoch;
  logic [2:0] candidate_accept_layer;
  logic [9:0] candidate_accept_expert;
  logic [4:0] candidate_accept_rank;
  logic [11:0] candidate_accept_confidence;
  logic candidate_accept_last;
  logic route_accept;
  logic [15:0] route_accept_epoch;
  logic [2:0] route_accept_layer;
  logic [9:0] route_accept_expert;
  logic route_accept_last;
  logic logical_valid;
  logic logical_ready;
  tile_descriptor_t logical_gu_descriptor;
  tile_descriptor_t logical_down_descriptor;
  logic resolver_route_valid;
  descriptor_token_t resolver_route_token;
  logic resolver_route_truth;
  logic resolver_route_ready;
  logic resolver_accept_ready;
  logic resolver_done;
  logic parser_prediction_ready;
  logic parser_route_begin_ready;
  logic parser_route_expert_ready;
  logic route_started;
  logic parser_error_internal;
  logic [31:0] parser_stale_count;
  logic [31:0] context_stale_count;
  logic protocol_error;
  assign parser_stale_transaction_count = parser_stale_count + context_stale_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) context_stale_count <= 0;
    else if (!protocol_error && effective_route_begin_valid && (effective_route_epoch != active_epoch || effective_route_layer != active_layer))
      context_stale_count <= context_stale_count + 1;
  end
  assign parser_error = parser_error_internal ||
    (effective_route_expert_valid && effective_route_weight_q15 > 32768) ||
    (effective_route_begin_valid && (effective_route_epoch != active_epoch || effective_route_layer != active_layer));
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) protocol_error <= 0;
    else if (soft_flush_valid) protocol_error <= 0;
    else if (parser_error) protocol_error <= 1;
  end
  logic [31:0] response_error_baseline;
  assign transaction_error = protocol_error || dma_response_error_count != response_error_baseline;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) response_error_baseline <= 0;
    else if (soft_flush_valid) response_error_baseline <= dma_response_error_count;
  end
  logic sequence_running;
  logic sequence_bandwidth_ready;
  logic sequence_layer_start;
  generate if (USE_SEQUENCE_CONTROL) begin : sequence_control
    sequence_controller controller (
      .clk(clk),
      .rst_n(rst_n),
      .start(sequence_start),
      .layer_done(result_valid || layer_done),
      .compute_block_cycles(compute_block_cycles),
      .bandwidth_block_cycles(bandwidth_block_cycles),
      .sequence_active(sequence_running),
      .sequence_done(sequence_done),
      .current_layer(current_layer),
      .layer_start_pulse(sequence_layer_start),
      .compute_ready(sequence_compute_ready),
      .bandwidth_ready(sequence_bandwidth_ready)
    );
  end else begin : external_sequence_control
    assign sequence_running = sequence_active;
    assign sequence_done = 0;
    assign current_layer = prediction_layer;
    assign sequence_compute_ready = 1;
    assign sequence_bandwidth_ready = 1;
    assign sequence_layer_start = 0;
  end endgenerate
  logic [63:0] gu_base_hold;
  logic [63:0] down_base_hold;
  logic [31:0] stride_hold;
  logic [31:0] tile_bytes_hold;
  logic [15:0] sequence_hold;
  logic [H*8-1:0] activation_hold;
  logic [4:0] shift_hold;
  initial begin
    if (MAX_CANDIDATES < 1 || MAX_CANDIDATES > 31 || NUM_EXPERTS < 1 || NUM_EXPERTS > 1024 ||
        GU_BYTE_RATIO < 1 || DESCRIPTOR_TABLE_ENTRIES < 2 || OCM_SLOT_BYTES < 16)
      $fatal(1, "unsupported ABI parameter range");
    if (USE_REAL_ROUTER && (!USE_REAL_COMPUTE || !AUTO_COMPUTE))
      $fatal(1, "real Router requires shared real MAC and automatic execution");
    if (STRIPED_DDR && (MAX_BURST_BYTES < 64 || MAX_BURST_BYTES%64 != 0 || MAX_BURST_BYTES > 4096))
      $fatal(1, "striped DMA requires 64-byte burst geometry");
  end
  assign config_error = prediction_begin_valid &&
    (gu_base_addr[3:0] != 0 || down_base_addr[3:0] != 0 || physical_tile_bytes == 0 ||
     physical_tile_bytes[3:0] != 0 || expert_stride_bytes[3:0] != 0 ||
     {32'd0, expert_stride_bytes} < {32'd0, physical_tile_bytes} * GU_BYTE_RATIO * TILES_PER_EXPERT ||
     physical_tile_bytes > OCM_SLOT_BYTES/GU_BYTE_RATIO ||
     (USE_REAL_COMPUTE && (GU_BYTE_RATIO != (SWITCH_RELU ? 1 : 2) || physical_tile_bytes != H*F_TILE)) ||
     (STRIPED_DDR && (gu_base_addr[5:0] != 0 || down_base_addr[5:0] != 0 ||
       physical_tile_bytes[5:0] != 0 || expert_stride_bytes[5:0] != 0)) ||
     (USE_REAL_ROUTER && (!router_weights_loaded || router_weight_error ||
       router_loaded_layer != prediction_layer || router_top_k_config == 0 ||
       router_top_k_config > 8 || int'(router_top_k_config) > NUM_EXPERTS)));
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      gu_base_hold <= 0;
      down_base_hold <= 0;
      stride_hold <= 0;
      tile_bytes_hold <= 0;
      sequence_hold <= 0;
      activation_hold <= 0;
      shift_hold <= 0;
      active_epoch <= 0;
      active_layer <= 0;
    end else if (prediction_begin_valid && prediction_begin_ready) begin
      gu_base_hold <= gu_base_addr;
      down_base_hold <= down_base_addr;
      stride_hold <= expert_stride_bytes;
      tile_bytes_hold <= physical_tile_bytes;
      sequence_hold <= prediction_sequence;
      activation_hold <= compute_activation;
      shift_hold <= gu_requant_shift;
      active_epoch <= prediction_epoch;
      active_layer <= prediction_layer;
    end
  end
  logic demand_active;
  assign transaction_drained = memory_drained && !allocation_active && !demand_active &&
    resolver_accept_ready && !compute_busy;
  assign context_retire_ready = memory_ingress_idle && !allocation_active && !demand_active &&
    resolver_accept_ready && !compute_busy;
  assign prediction_begin_ready = parser_prediction_ready && !workload_busy && !allocation_active && !demand_active && resolver_accept_ready && !config_error &&
    (!ENABLE_CROSS_LAYER || memory_ingress_idle);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) workload_busy <= 0;
    else if (prediction_begin_valid && prediction_begin_ready) workload_busy <= 1;
    else if (result_valid || layer_done || soft_flush_valid) workload_busy <= 0;
  end
  logic auto_req_valid;
  logic automatic_result_valid;
  assign result_valid = automatic_result_valid && !transaction_error;
  logic memory_compute_ready;
  tile_descriptor_t auto_gu_descriptor;
  tile_descriptor_t auto_down_descriptor;
  logic [GENERATION_W-1:0] auto_gu_generation;
  logic [GENERATION_W-1:0] auto_down_generation;
  logic dma_completion_valid;
  tile_descriptor_t dma_completion_descriptor;
  logic [15:0] route_weight_hold;
  assign compute_req_ready = AUTO_COMPUTE ? 1'b0 : memory_compute_ready;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) route_weight_hold <= 0;
    else if (effective_route_expert_valid && effective_route_expert_ready) route_weight_hold <= effective_route_weight_q15;
  end
  logic demand_done;
  logic demand_valid;
  tile_descriptor_t demand_gu_descriptor;
  tile_descriptor_t demand_down_descriptor;
  logic memory_logical_valid;
  tile_descriptor_t memory_gu_descriptor;
  tile_descriptor_t memory_down_descriptor;
  wire current_pair_admitted;
  wire current_spec_selected = logical_valid && !route_started && !resolver_full && !current_pair_admitted;
  wire future_selected = ENABLE_CROSS_LAYER && !demand_active && !current_spec_selected &&
    !prediction_begin_valid && !resolver_full && !context_switch_pending;
  wire current_spec_fire = !soft_flush_valid && !demand_active && current_spec_selected && logical_ready;
  assign future_logical_ready = !soft_flush_valid && future_selected && logical_ready;
  assign memory_logical_valid = !soft_flush_valid && (demand_active ? demand_valid :
    current_spec_selected || (future_selected && future_logical_valid));
  assign memory_gu_descriptor = demand_active ? demand_gu_descriptor :
    (current_spec_selected || !ENABLE_CROSS_LAYER) ? logical_gu_descriptor : future_gu_descriptor;
  assign memory_down_descriptor = demand_active ? demand_down_descriptor :
    (current_spec_selected || !ENABLE_CROSS_LAYER) ? logical_down_descriptor : future_down_descriptor;
  assign effective_route_begin_ready = parser_route_begin_ready && resolver_accept_ready && workload_busy &&
    effective_route_epoch == active_epoch && effective_route_layer == active_layer;
  assign effective_route_expert_ready = parser_route_expert_ready && resolver_accept_ready && effective_route_weight_q15 <= 32768;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) route_started <= 0;
    else if (prediction_begin_valid && prediction_begin_ready) route_started <= 0;
    else if (effective_route_begin_valid && effective_route_begin_ready) route_started <= 1;
  end

  control_transaction_parser #(
    .MAX_EXPERTS(NUM_EXPERTS),
    .EXPERT_W(10),
    .MAX_COUNT(MAX_CANDIDATES)
  ) parser (
    .clk(clk),
    .rst_n(rst_n),
    .clear(soft_flush_valid),
    .prediction_begin_valid(prediction_begin_valid && prediction_begin_ready),
    .prediction_begin_ready(parser_prediction_ready),
    .prediction_epoch(prediction_epoch),
    .prediction_layer(prediction_layer),
    .prediction_count(prediction_count),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(candidate_expert),
    .candidate_rank(candidate_rank),
    .candidate_confidence(candidate_confidence),
    .candidate_last(candidate_last),
    .route_begin_valid(effective_route_begin_valid && effective_route_begin_ready),
    .route_begin_ready(parser_route_begin_ready),
    .route_epoch(effective_route_epoch),
    .route_layer(effective_route_layer),
    .route_top_k(effective_route_top_k),
    .route_expert_valid(effective_route_expert_valid && effective_route_expert_ready),
    .route_expert_ready(parser_route_expert_ready),
    .route_expert(effective_route_expert),
    .route_last(effective_route_last),
    .candidate_accept(candidate_accept),
    .candidate_accept_epoch(candidate_accept_epoch),
    .candidate_accept_layer(candidate_accept_layer),
    .candidate_accept_expert(candidate_accept_expert),
    .candidate_accept_rank(candidate_accept_rank),
    .candidate_accept_confidence(candidate_accept_confidence),
    .candidate_accept_last(candidate_accept_last),
    .route_accept(route_accept),
    .route_accept_epoch(route_accept_epoch),
    .route_accept_layer(route_accept_layer),
    .route_accept_expert(route_accept_expert),
    .route_accept_last(route_accept_last),
    .parser_error(parser_error_internal),
    .stale_transaction_count(parser_stale_count)
  );

  candidate_prefetch_scheduler #(
    .MAX_CANDIDATES(MAX_CANDIDATES),
    .TILES_PER_EXPERT(TILES_PER_EXPERT),
    .CHANNEL_COUNT(CHANNEL_COUNT),
    .EXPERT_W(10),
    .GENERATION_W(GENERATION_W),
    .GU_BYTE_RATIO(GU_BYTE_RATIO)
  ) scheduler (
    .clk(clk),
    .rst_n(rst_n),
    .transaction_begin(prediction_begin_valid && prediction_begin_ready),
    .stop_allocation(route_started || soft_flush_valid),
    .transaction_epoch(prediction_epoch),
    .transaction_sequence(prediction_sequence),
    .transaction_layer(prediction_layer),
    .speculative_budget_bytes(speculative_budget_bytes),
    .channel_credit_bytes(channel_credit_bytes),
    .candidate_accept(candidate_accept && !soft_flush_valid),
    .candidate_epoch(candidate_accept_epoch),
    .candidate_layer(candidate_accept_layer),
    .candidate_expert(candidate_accept_expert),
    .candidate_rank(candidate_accept_rank),
    .candidate_confidence(candidate_accept_confidence),
    .candidate_last(candidate_accept_last),
    .gu_base_addr(gu_base_hold),
    .down_base_addr(down_base_hold),
    .expert_stride_bytes(stride_hold),
    .physical_tile_bytes(tile_bytes_hold),
    .logical_valid(logical_valid),
    .logical_ready(!route_started && (current_pair_admitted || (logical_ready && !demand_active))),
    .logical_gu_descriptor(logical_gu_descriptor),
    .logical_down_descriptor(logical_down_descriptor),
    .allocation_active(allocation_active),
    .allocation_done(allocation_done),
    .remaining_credit_bytes(remaining_credit_bytes)
  );

  generate if (AUTO_COMPUTE) begin : automatic_execution
    tile_execution_scheduler #(
      .SLOT_COUNT(NUM_OCM_BANKS*SLOTS_PER_BANK),
      .TILES_PER_EXPERT(TILES_PER_EXPERT),
      .H(H),
      .GENERATION_W(GENERATION_W),
      .PRESERVE_CONTEXTS(ENABLE_CROSS_LAYER),
      .WAIT_FULL_EXPERT(WAIT_FULL_EXPERT)
    ) execution (
      .clk(clk),
      .rst_n(rst_n),
      .clear((prediction_begin_valid && prediction_begin_ready) || soft_flush_valid),
      .context_epoch(active_epoch),
      .context_layer(active_layer),
      .flush_valid(soft_flush_valid),
      .flush_epoch(soft_flush_epoch),
      .capture_valid(physical_accept_valid),
      .capture_descriptor(physical_accept_descriptor),
      .capture_generation(physical_accept_slot_generation),
      .dma_done(dma_completion_valid),
      .dma_descriptor(dma_completion_descriptor),
      .route_accept(route_accept && !soft_flush_valid),
      .route_expert(route_accept_expert),
      .route_weight_q15(route_weight_hold),
      .route_last(route_accept_last),
      .compute_req_valid(auto_req_valid),
      .compute_req_ready(memory_compute_ready && sequence_compute_ready && !transaction_error),
      .gu_descriptor(auto_gu_descriptor),
      .down_descriptor(auto_down_descriptor),
      .gu_generation(auto_gu_generation),
      .down_generation(auto_down_generation),
      .compute_done(compute_done),
      .gu_release_valid(compute_gu_release_valid),
      .partial_y(compute_partial_y),
      .layer_done(automatic_result_valid),
      .layer_y(result_y)
    );
  end else begin : manual_execution
    assign auto_req_valid = 0;
    assign auto_gu_descriptor = '0;
    assign auto_down_descriptor = '0;
    assign auto_gu_generation = 0;
    assign auto_down_generation = 0;
    assign automatic_result_valid = 0;
    assign result_y = 0;
  end endgenerate

  generate if (ENABLE_CROSS_LAYER) begin : cross_layer_demand
    multilayer_tile_demander #(
      .LAYERS(LAYER_COUNT),
      .EXPERTS(NUM_EXPERTS),
      .TILES_PER_EXPERT(TILES_PER_EXPERT),
      .GU_BYTE_RATIO(GU_BYTE_RATIO)
    ) demander (
      .clk(clk),
      .rst_n(rst_n),
      .clear((prediction_begin_valid && prediction_begin_ready) || soft_flush_valid),
      .clear_sequence(clear_sequence),
      .speculative_accept(current_spec_fire || (future_logical_valid && future_logical_ready)),
      .speculative_descriptor(current_spec_fire ? logical_gu_descriptor : future_gu_descriptor),
      .query_descriptor(logical_gu_descriptor),
      .query_admitted(current_pair_admitted),
      .route_accept(route_accept && !soft_flush_valid),
      .route_epoch(route_accept_epoch),
      .route_layer(route_accept_layer),
      .route_expert(route_accept_expert),
      .start(resolver_done),
      .sequence_id(sequence_hold),
      .gu_base_addr(gu_base_hold),
      .down_base_addr(down_base_hold),
      .expert_stride_bytes(stride_hold),
      .physical_tile_bytes(tile_bytes_hold),
      .logical_valid(demand_valid),
      .logical_ready(logical_ready),
      .gu_descriptor(demand_gu_descriptor),
      .down_descriptor(demand_down_descriptor),
      .active(demand_active),
      .done(demand_done)
    );
  end else begin : legacy_demand
  assign current_pair_admitted=0;
  true_tile_demander #(
    .MAX_CANDIDATES(MAX_CANDIDATES),
    .TILES_PER_EXPERT(TILES_PER_EXPERT),
    .GU_BYTE_RATIO(GU_BYTE_RATIO)
  ) demander (
    .clk(clk),
    .rst_n(rst_n),
    .clear((prediction_begin_valid && prediction_begin_ready) || soft_flush_valid),
    .speculative_accept(current_spec_fire),
    .speculative_descriptor(logical_gu_descriptor),
    .route_accept(route_accept && !soft_flush_valid),
    .route_epoch(route_accept_epoch),
    .route_layer(route_accept_layer),
    .route_expert(route_accept_expert),
    .start(resolver_done),
    .sequence_id(sequence_hold),
    .gu_base_addr(gu_base_hold),
    .down_base_addr(down_base_hold),
    .expert_stride_bytes(stride_hold),
    .physical_tile_bytes(tile_bytes_hold),
    .logical_valid(demand_valid),
    .logical_ready(logical_ready),
    .gu_descriptor(demand_gu_descriptor),
    .down_descriptor(demand_down_descriptor),
    .active(demand_active),
    .done(demand_done)
  );
  end endgenerate

  layer_weight_barrier #(
    .LAYERS(LAYER_COUNT),
    .EXPERTS(NUM_EXPERTS),
    .TILES(TILES_PER_EXPERT)
  ) weight_barrier (
    .clk(clk),
    .rst_n(rst_n),
    .clear_sequence(clear_sequence === 1'b1),
    .route_accept(route_accept && !soft_flush_valid),
    .route_layer(route_accept_layer),
    .route_expert(route_accept_expert),
    .route_last(route_accept_last),
    .dma_done(dma_completion_valid && !transaction_error),
    .dma_descriptor(dma_completion_descriptor),
    .weights_done(layer_weights_done)
  );

  route_token_resolver #(
    .MAX_LIVE(DESCRIPTOR_TABLE_ENTRIES),
    .TOP_K_MAX(8),
    .EXPERT_W(10)
  ) resolver (
    .clk(clk),
    .rst_n(rst_n),
    .capture_valid(physical_accept_valid && !physical_accept_descriptor.request_class),
    .capture_token(physical_accept_token),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .capture_descriptor(physical_accept_descriptor),
    .route_accept(route_accept && !soft_flush_valid),
    .route_epoch(route_accept_epoch),
    .route_layer(route_accept_layer),
    .route_expert(route_accept_expert),
    .route_last(route_accept_last),
    .route_accept_ready(resolver_accept_ready),
    .resolution_done(resolver_done),
    .route_valid(resolver_route_valid),
    .route_token(resolver_route_token),
    .route_truth(resolver_route_truth),
    .route_ready(resolver_route_ready),
    .resolver_full(resolver_full),
    .dropped_capture_count(resolver_dropped_capture_count)
  );

  tile_prefetch_axi_top #(
    .DESCRIPTOR_TABLE_ENTRIES(DESCRIPTOR_TABLE_ENTRIES),
    .NUM_OCM_BANKS(NUM_OCM_BANKS),
    .SLOTS_PER_BANK(SLOTS_PER_BANK),
    .OCM_SLOT_BYTES(OCM_SLOT_BYTES),
    .DEMAND_RESERVE_SLOTS(WAIT_FULL_EXPERT ? 2*TILES_PER_EXPERT : 2),
    .DMA_FIFO_DEPTH(DMA_FIFO_DEPTH),
    .MAX_BURST_BYTES(MAX_BURST_BYTES),
    .GENERATION_W(GENERATION_W),
    .AXI_ADDR_W(AXI_ADDR_W),
    .AXI_DATA_W(AXI_DATA_W),
    .STRIPED_DDR(STRIPED_DDR),
    .NUM_ROUTER_EXPERTS(NUM_EXPERTS),
    .SWITCH_RELU(SWITCH_RELU),
    .USE_REAL_COMPUTE(USE_REAL_COMPUTE),
    .H(H),
    .F_TILE(F_TILE)
  ) memory_dataflow (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable && sequence_bandwidth_ready),
    .setup_cycles(setup_cycles),
    .sequence_active(sequence_running),
    .layer_done(layer_done || result_valid),
    .scheduler_active(allocation_active || demand_active),
    .logical_valid(memory_logical_valid),
    .logical_gu_descriptor(memory_gu_descriptor),
    .logical_down_descriptor(memory_down_descriptor),
    .existing_pair_live(1'b0),
    .existing_pair_is_speculative(1'b0),
    .logical_ready(logical_ready),
    .physical_accept_valid(physical_accept_valid),
    .physical_accept_token(physical_accept_token),
    .physical_accept_descriptor(physical_accept_descriptor),
    .physical_accept_slot_generation(physical_accept_slot_generation),
    .route_valid(resolver_route_valid),
    .route_token(resolver_route_token),
    .route_truth(resolver_route_truth),
    .route_ready(resolver_route_ready),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .compute_req_valid((AUTO_COMPUTE ? auto_req_valid : compute_req_valid) && sequence_compute_ready && !transaction_error),
    .compute_req_ready(memory_compute_ready),
    .compute_gu_descriptor(AUTO_COMPUTE ? auto_gu_descriptor : compute_gu_descriptor),
    .compute_gu_slot_generation(AUTO_COMPUTE ? auto_gu_generation : compute_gu_slot_generation),
    .compute_down_descriptor(AUTO_COMPUTE ? auto_down_descriptor : compute_down_descriptor),
    .compute_down_slot_generation(AUTO_COMPUTE ? auto_down_generation : compute_down_slot_generation),
    .compute_gu_cycles(compute_gu_cycles),
    .compute_nonlinear_cycles(compute_nonlinear_cycles),
    .compute_down_cycles(compute_down_cycles),
    .compute_activation(activation_hold),
    .gu_requant_shift(shift_hold),
    .router_start(router_start_pulse),
    .router_done(router_gemm_done),
    .router_read_valid(router_read_valid),
    .router_read_offset(router_read_offset),
    .router_read_data(router_read_data),
    .router_logits(router_logits),
    .compute_partial_y(compute_partial_y),
    .dma_completion_valid(dma_completion_valid),
    .dma_completion_descriptor(dma_completion_descriptor),
    .compute_busy(compute_busy),
    .memory_drained(memory_drained),
    .memory_ingress_idle(memory_ingress_idle),
    .compute_done(compute_done),
    .compute_gu_release_valid(compute_gu_release_valid),
    .compute_waiting_gu(compute_waiting_gu),
    .compute_waiting_down(compute_waiting_down),
    .demand_stall_cycles(demand_stall_cycles),
    .stale_event_count(stale_event_count),
    .discarded_dma_write_count(discarded_dma_write_count),
    .payload_discarded_write_count(payload_discarded_write_count),
    .dma_response_error_count(dma_response_error_count),
    .tombstone_skip_count(tombstone_skip_count),
    .telemetry_total_cycles(),
    .telemetry_speculative_allocated_bytes(),
    .telemetry_speculative_issued_bytes(telemetry_speculative_issued_bytes),
    .telemetry_demand_issued_bytes(telemetry_demand_issued_bytes),
    .telemetry_useful_speculative_bytes(telemetry_useful_speculative_bytes),
    .telemetry_wasted_speculative_bytes(telemetry_wasted_speculative_bytes),
    .telemetry_cancelled_speculative_bytes(telemetry_cancelled_speculative_bytes),
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
    .m_axi_rready(m_axi_rready),
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
endmodule
