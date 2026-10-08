`timescale 1ns/1ps
/** U200 四 bank core-clock 接口：真实共享 Router/Expert MAC，外部 predictor。
 * 不包含 DDR PHY、板级 shell、CDC 或 PCIe。allocator 使用一个聚合 credit，
 * 每个 tile 均匀 stripe 到四个 bank；不可把逻辑 channel 当作独立物理 DMA。
 */
module u200_moe_top #(
  parameter integer MAX_CANDIDATES = 16,
  parameter integer NUM_EXPERTS = 64,
  parameter integer TILES_PER_EXPERT = 16,
  parameter integer CHANNEL_COUNT = 1,
  parameter integer DESCRIPTOR_TABLE_ENTRIES = 64,
  parameter integer NUM_OCM_BANKS = 4,
  parameter integer SLOTS_PER_BANK = 32,
  parameter integer OCM_SLOT_BYTES = 4096,
  parameter integer DMA_FIFO_DEPTH = 16,
  parameter integer MAX_BURST_BYTES = 256,
  parameter integer GENERATION_W = 8,
  parameter integer AXI_ADDR_W = 64,
  parameter bit SWITCH_RELU = 1'b0,
  parameter integer GU_BYTE_RATIO = SWITCH_RELU ? 1 : 2,
  parameter integer H = 16,
  parameter integer F_TILE = 16,
  parameter integer LAYER_COUNT = 1,
  parameter bit WAIT_FULL_EXPERT = 1'b0,
  parameter bit ENABLE_LAYER_CHAIN = LAYER_COUNT > 1,
  parameter bit ENABLE_CROSS_LAYER = ENABLE_LAYER_CHAIN

) (
  input logic clk,
  input logic rst_n,
  input logic bandwidth_enable,
  input logic [31:0] setup_cycles,
  input logic sequence_active,
  input logic sequence_start,
  output logic sequence_start_ready,
  output logic sequence_busy,
  input logic [4:0] layer_output_requant_shift,
  input logic future_begin_valid,
  output logic future_begin_ready,
  input logic [2:0] future_target_layer,
  input logic [15:0] future_epoch,
  input logic [15:0] future_sequence,
  input logic [4:0] future_candidate_count,
  input logic [31:0] future_budget_bytes,
  input logic [CHANNEL_COUNT*32-1:0] future_channel_credit_bytes,
  input logic [63:0] future_gu_base_addr,
  input logic [63:0] future_down_base_addr,
  input logic [31:0] future_expert_stride_bytes,
  input logic future_candidate_valid,
  output logic future_candidate_ready,
  input logic [9:0] future_candidate_expert,
  input logic [4:0] future_candidate_rank,
  input logic [11:0] future_candidate_confidence,
  input logic future_candidate_last,
  output wire [LAYER_COUNT-1:0] layer_weights_done,
  output logic [H*32-1:0] sequence_result_y,
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
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic [31:0] physical_tile_bytes,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
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
  logic core_prediction_ready;
  initial begin
    if(WAIT_FULL_EXPERT && NUM_OCM_BANKS*SLOTS_PER_BANK < 2*TILES_PER_EXPERT)
      $fatal(1,"WAIT_FULL_EXPERT requires OCM slots for all GU/DOWN tiles");
    if(LAYER_COUNT<1 || LAYER_COUNT>8 || (ENABLE_CROSS_LAYER && !ENABLE_LAYER_CHAIN))
      $fatal(1,"cross-layer mode requires layer chain and 1..8 layers");
  end
  logic core_sequence_done;
  logic [2:0] core_current_layer;
  logic core_drained;
  logic core_retire_ready;
  logic core_transaction_error;
  logic future_error;
  logic future_context_ok;
  logic future_logical_valid;
  logic future_logical_ready;
  tile_prefetch_package::tile_descriptor_t future_gu_descriptor;
  tile_prefetch_package::tile_descriptor_t future_down_descriptor;
  logic future_flush;
  logic [15:0] future_flush_epoch;
  logic sequence_context_live;
  wire sequence_start_fire=ENABLE_LAYER_CHAIN && sequence_start && sequence_start_ready;
  assign transaction_error=core_transaction_error || future_error;
  logic chain_layer_ready;
  logic chain_flush;
  logic chain_retiring;
  logic [H*8-1:0] chain_activation;
  logic [15:0] active_epoch;
  logic [15:0] active_sequence;
  logic chain_accept;
  logic chain_context_ok;
  logic effective_prediction_valid;
  logic effective_flush;
  logic [15:0] effective_flush_epoch;
  assign chain_context_ok = prediction_layer == current_layer &&
    (current_layer == 0 || prediction_sequence == active_sequence) &&
    (!ENABLE_CROSS_LAYER || future_context_ok);
  assign effective_prediction_valid = prediction_begin_valid &&
    (!ENABLE_LAYER_CHAIN || (chain_layer_ready && chain_context_ok)) && !soft_flush_valid;
  assign prediction_begin_ready = core_prediction_ready &&
    (!ENABLE_LAYER_CHAIN || (chain_layer_ready && chain_context_ok)) && !soft_flush_valid;
  assign chain_accept = effective_prediction_valid && core_prediction_ready;
  assign effective_flush = soft_flush_valid || future_flush || chain_flush;
  assign effective_flush_epoch = soft_flush_valid ? soft_flush_epoch :
    future_flush ? future_flush_epoch : active_epoch;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) sequence_context_live<=0;
    else if(sequence_start_fire || soft_flush_valid || transaction_error) sequence_context_live<=0;
    else if(chain_accept) sequence_context_live<=1;
  end
  always_ff @(posedge clk) begin
    if (chain_accept) active_epoch <= prediction_epoch;
  end
  always_ff @(posedge clk) begin
    if (chain_accept) active_sequence <= prediction_sequence;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sequence_result_y <= 0;
    else if (ENABLE_LAYER_CHAIN && sequence_start && sequence_start_ready) sequence_result_y <= 0;
    else if (result_valid) sequence_result_y <= result_y;
  end
  generate if (ENABLE_LAYER_CHAIN) begin : multilayer
    moe_layer_chain #(
      .LAYER_COUNT(LAYER_COUNT),
      .H(H),
      .NONBLOCKING_RETIRE(ENABLE_CROSS_LAYER)
    ) controller (
      .clk(clk),
      .rst_n(rst_n),
      .start(sequence_start),
      .start_ready(sequence_start_ready),
      .abort(soft_flush_valid || transaction_error),
      .layer_accept(chain_accept),
      .layer_result_valid(result_valid),
      .layer_result(result_y),
      .transaction_drained(core_drained),
      .context_retire_ready(core_retire_ready),
      .initial_activation(compute_activation),
      .output_requant_shift(layer_output_requant_shift),
      .activation(chain_activation),
      .layer_ready(chain_layer_ready),
      .current_layer(current_layer),
      .busy(sequence_busy),
      .done(sequence_done),
      .context_flush(chain_flush),
      .context_retiring(chain_retiring)
    );
  end else begin : single_layer
    assign sequence_start_ready = 0;
    assign sequence_busy = sequence_active;
    assign sequence_done = core_sequence_done;
    assign current_layer = core_current_layer;
    assign chain_layer_ready = 1;
    assign chain_flush = 0;
    assign chain_retiring = 0;
    assign chain_activation = compute_activation;
  end endgenerate
  generate if(ENABLE_CROSS_LAYER) begin : future_contexts
    cross_layer_prefetch #(
      .LAYERS(LAYER_COUNT),
      .MAX_CANDIDATES(MAX_CANDIDATES),
      .EXPERTS(NUM_EXPERTS),
      .TILES(TILES_PER_EXPERT),
      .CHANNELS(CHANNEL_COUNT),
      .GU_BYTE_RATIO(GU_BYTE_RATIO),
      .H(H),
      .F_TILE(F_TILE)
    ) prefetch (
      .clk(clk),
      .rst_n(rst_n),
      .clear_sequence(sequence_start_fire),
      .sequence_busy(sequence_busy && sequence_context_live),
      .abort(soft_flush_valid || transaction_error),
      .current_layer(current_layer),
      .current_begin(chain_accept),
      .current_epoch(prediction_epoch),
      .current_sequence(prediction_sequence),
      .sequence_id(active_sequence),
      .begin_valid(future_begin_valid === 1'b1),
      .begin_ready(future_begin_ready),
      .target_layer(future_target_layer),
      .target_epoch(future_epoch),
      .target_sequence(future_sequence),
      .candidate_count(future_candidate_count),
      .budget_bytes(future_budget_bytes),
      .channel_credit_bytes(future_channel_credit_bytes),
      .gu_base_addr(future_gu_base_addr),
      .down_base_addr(future_down_base_addr),
      .expert_stride_bytes(future_expert_stride_bytes),
      .candidate_valid(future_candidate_valid === 1'b1),
      .candidate_ready(future_candidate_ready),
      .candidate_expert(future_candidate_expert),
      .candidate_rank(future_candidate_rank),
      .candidate_confidence(future_candidate_confidence),
      .candidate_last(future_candidate_last),
      .logical_valid(future_logical_valid),
      .logical_ready(future_logical_ready),
      .gu_descriptor(future_gu_descriptor),
      .down_descriptor(future_down_descriptor),
      .current_context_ok(future_context_ok),
      .flush_valid(future_flush),
      .flush_epoch(future_flush_epoch),
      .flush_ready(!soft_flush_valid),
      .error(future_error)
    );
  end else begin : no_future_contexts
    assign future_begin_ready=0;
    assign future_candidate_ready=0;
    assign future_context_ok=1;
    assign future_logical_valid=0;
    assign future_gu_descriptor='0;
    assign future_down_descriptor='0;
    assign future_flush=0;
    assign future_flush_epoch=0;
    assign future_error=0;
  end endgenerate
  tile_prefetch_predictive_axi_top #(
    .ENABLE_CROSS_LAYER(ENABLE_CROSS_LAYER),
    .LAYER_COUNT(LAYER_COUNT),
    .WAIT_FULL_EXPERT(WAIT_FULL_EXPERT),
    .MAX_CANDIDATES(MAX_CANDIDATES),
    .NUM_EXPERTS(NUM_EXPERTS),
    .TILES_PER_EXPERT(TILES_PER_EXPERT),
    .CHANNEL_COUNT(CHANNEL_COUNT),
    .DESCRIPTOR_TABLE_ENTRIES(DESCRIPTOR_TABLE_ENTRIES),
    .NUM_OCM_BANKS(NUM_OCM_BANKS),
    .SLOTS_PER_BANK(SLOTS_PER_BANK),
    .OCM_SLOT_BYTES(OCM_SLOT_BYTES),
    .DMA_FIFO_DEPTH(DMA_FIFO_DEPTH),
    .MAX_BURST_BYTES(MAX_BURST_BYTES),
    .GENERATION_W(GENERATION_W),
    .AXI_ADDR_W(AXI_ADDR_W),
    .GU_BYTE_RATIO(GU_BYTE_RATIO),
    .SWITCH_RELU(SWITCH_RELU),
    .H(H),
    .F_TILE(F_TILE),
    .USE_REAL_ROUTER(1'b1),
    .STRIPED_DDR(1'b1),
    .USE_REAL_COMPUTE(1'b1),
    .AUTO_COMPUTE(1'b1),
    .USE_SEQUENCE_CONTROL(1'b0)
  ) core (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
    .setup_cycles(setup_cycles),
    .sequence_active(ENABLE_LAYER_CHAIN ? sequence_busy : sequence_active),
    .sequence_start('0),
    .compute_block_cycles('0),
    .bandwidth_block_cycles('0),
    .sequence_done(core_sequence_done),
    .current_layer(core_current_layer),
    .layer_done(ENABLE_LAYER_CHAIN ? 1'b0 : layer_done),
    .prediction_begin_valid(effective_prediction_valid),
    .prediction_begin_ready(core_prediction_ready),
    .prediction_epoch(prediction_epoch),
    .prediction_sequence(prediction_sequence),
    .prediction_layer(prediction_layer),
    .prediction_count(prediction_count),
    .speculative_budget_bytes(speculative_budget_bytes),
    .channel_credit_bytes(channel_credit_bytes),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(candidate_expert),
    .candidate_rank(candidate_rank),
    .candidate_confidence(candidate_confidence),
    .candidate_last(candidate_last),
    .route_begin_valid('0),
    .route_begin_ready(),
    .route_epoch('0),
    .route_layer('0),
    .route_top_k('0),
    .route_expert_valid('0),
    .route_expert_ready(),
    .route_expert('0),
    .route_last('0),
    .route_weight_q15('0),
    .gu_base_addr(gu_base_addr),
    .down_base_addr(down_base_addr),
    .expert_stride_bytes(expert_stride_bytes),
    .physical_tile_bytes(physical_tile_bytes),
    .soft_flush_valid(effective_flush),
    .soft_flush_epoch(effective_flush_epoch),
    .compute_req_valid('0),
    .compute_req_ready(),
    .compute_gu_descriptor('0),
    .compute_gu_slot_generation('0),
    .compute_down_descriptor('0),
    .compute_down_slot_generation('0),
    .compute_gu_cycles('0),
    .compute_nonlinear_cycles('0),
    .compute_down_cycles('0),
    .compute_activation(chain_activation),
    .gu_requant_shift(gu_requant_shift),
    .router_top_k_config(router_top_k_config),
    .router_logit_shift(router_logit_shift),
    .router_weight_reload_valid(router_weight_reload_valid),
    .router_weight_reload_ready(router_weight_reload_ready),
    .router_weight_layer(router_weight_layer),
    .router_weight_valid(router_weight_valid),
    .router_weight_ready(router_weight_ready),
    .router_weight_word_index(router_weight_word_index),
    .router_weight_data(router_weight_data),
    .router_weight_last(router_weight_last),
    .router_weights_loaded(router_weights_loaded),
    .router_weight_error(router_weight_error),
    .router_gemm_done(router_gemm_done),
    .compute_partial_y(compute_partial_y),
    .result_valid(result_valid),
    .result_y(result_y),
    .physical_accept_valid(physical_accept_valid),
    .physical_accept_token(physical_accept_token),
    .physical_accept_descriptor(physical_accept_descriptor),
    .physical_accept_slot_generation(physical_accept_slot_generation),
    .compute_busy(compute_busy),
    .transaction_drained(core_drained),
    .context_retire_ready(core_retire_ready),
    .context_switch_pending(chain_retiring),
    .clear_sequence(sequence_start_fire),
    .future_logical_valid(future_logical_valid),
    .future_logical_ready(future_logical_ready),
    .future_gu_descriptor(future_gu_descriptor),
    .future_down_descriptor(future_down_descriptor),
    .layer_weights_done(layer_weights_done),
    .compute_done(compute_done),
    .compute_waiting_gu(compute_waiting_gu),
    .compute_waiting_down(compute_waiting_down),
    .parser_error(parser_error),
    .config_error(config_error),
    .transaction_error(core_transaction_error),
    .parser_stale_transaction_count(parser_stale_transaction_count),
    .allocation_active(allocation_active),
    .allocation_done(allocation_done),
    .remaining_credit_bytes(remaining_credit_bytes),
    .resolver_full(resolver_full),
    .resolver_dropped_capture_count(resolver_dropped_capture_count),
    .demand_stall_cycles(demand_stall_cycles),
    .stale_event_count(stale_event_count),
    .discarded_dma_write_count(discarded_dma_write_count),
    .payload_discarded_write_count(payload_discarded_write_count),
    .dma_response_error_count(dma_response_error_count),
    .tombstone_skip_count(tombstone_skip_count),
    .telemetry_speculative_issued_bytes(telemetry_speculative_issued_bytes),
    .telemetry_demand_issued_bytes(telemetry_demand_issued_bytes),
    .telemetry_useful_speculative_bytes(telemetry_useful_speculative_bytes),
    .telemetry_wasted_speculative_bytes(telemetry_wasted_speculative_bytes),
    .telemetry_cancelled_speculative_bytes(telemetry_cancelled_speculative_bytes),
    .m_axi_araddr(),
    .m_axi_arlen(),
    .m_axi_arsize(),
    .m_axi_arburst(),
    .m_axi_arvalid(),
    .m_axi_arready('0),
    .m_axi_rdata('0),
    .m_axi_rresp('0),
    .m_axi_rlast('0),
    .m_axi_rvalid('0),
    .m_axi_rready(),
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
  initial begin
    if (CHANNEL_COUNT != 1) $fatal(1, "striped U200 uses aggregate allocator credit");
  end
endmodule
