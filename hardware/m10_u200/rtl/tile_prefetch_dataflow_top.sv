`timescale 1ns/1ps
/**
 * M10.1 v0.9 权威 memory-side 数据通路顶层。
 *
 * Host/trace 提供已展开的 GU/DOWN 地址模板与 route truth；本模块负责原子
 * logical admission、slot/table/token ownership、burst DMA 以及 OCM state
 * update。该边界故意不绑定某个 U200 AXI 平台或 compute implementation。
 */
module tile_prefetch_top #(
  parameter integer DESCRIPTOR_TABLE_ENTRIES = 32,
  parameter integer NUM_OCM_BANKS = 4,
  parameter integer SLOTS_PER_BANK = 32,
  parameter integer DMA_FIFO_DEPTH = 8,
  parameter integer MAX_BURST_BYTES = 256,
  parameter integer REQUEST_ID_W = (DESCRIPTOR_TABLE_ENTRIES <= 1) ? 1 : $clog2(DESCRIPTOR_TABLE_ENTRIES),
  parameter integer GENERATION_W = 8,
  parameter integer DEMAND_RESERVE_SLOTS = 2,
  parameter integer DEMAND_RESERVE_DESCRIPTORS = 2,
  parameter bit USE_EXTERNAL_DMA = 1'b0,
  parameter bit USE_REAL_COMPUTE = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic bandwidth_enable,
  input logic [31:0] setup_cycles,
  input logic [31:0] bytes_per_cycle,
  input logic dma_external_burst_done,
  input logic sequence_active,
  input logic layer_done,
  input logic scheduler_active,
  input logic logical_valid,
  input tile_prefetch_package::tile_descriptor_t logical_gu_descriptor,
  input tile_prefetch_package::tile_descriptor_t logical_down_descriptor,
  input logic existing_pair_live,
  input logic existing_pair_is_speculative,
  output logic logical_ready,
  output logic logical_promotion_valid,
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
  input logic backend_gu_done,
  input logic backend_nonlinear_done,
  input logic backend_down_done,
  output logic backend_gu_start,
  output logic backend_down_start,
  output logic compute_busy,
  output logic memory_drained,
  output logic memory_ingress_idle,
  output logic compute_done,
  output logic compute_gu_release_valid,
  output logic compute_waiting_gu,
  output logic compute_waiting_down,
  output logic dma_burst_start,
  output logic dma_burst_done,
  output logic [31:0] dma_burst_bytes,
  output logic dma_done,
  output tile_prefetch_package::tile_descriptor_t dma_active_descriptor,
  output tile_prefetch_package::descriptor_token_t dma_active_token,
  output logic [GENERATION_W-1:0] dma_active_slot_generation,
  output logic [31:0] dma_active_remaining_bytes,
  output logic [31:0] demand_stall_cycles,
  output logic [31:0] stale_event_count,
  output logic [31:0] discarded_dma_write_count,
  output logic [31:0] tombstone_skip_count
  ,output logic [63:0] telemetry_total_cycles
  ,output logic [63:0] telemetry_speculative_allocated_bytes
  ,output logic [63:0] telemetry_speculative_issued_bytes
  ,output logic [63:0] telemetry_demand_issued_bytes
  ,output logic [63:0] telemetry_compute_wait_gu_cycles
  ,output logic [63:0] telemetry_compute_wait_down_cycles
  ,output logic [63:0] telemetry_ocm_peak_occupancy
  ,output logic [63:0] telemetry_commit_count
  ,output logic [63:0] telemetry_cancel_count
  ,output logic [63:0] telemetry_useful_speculative_bytes
  ,output logic [63:0] telemetry_wasted_speculative_bytes
  ,output logic [63:0] telemetry_cancelled_speculative_bytes
);
  import tile_prefetch_package::*;
  localparam integer SLOT_W = $clog2(SLOTS_PER_BANK);
  localparam integer TOTAL_SLOT_W = $clog2(NUM_OCM_BANKS) + SLOT_W;

  logic manager_reserve_pair, manager_promotion, manager_gu_valid, manager_gu_ready, manager_down_valid, manager_down_ready;
  tile_descriptor_t manager_gu_desc, manager_down_desc;
  logic [REQUEST_ID_W:0] table_free_entries;
  logic table_allocate_valid, table_allocate_ready;
  logic [REQUEST_ID_W-1:0] table_allocate_id;
  logic [GENERATION_W-1:0] table_allocate_generation;
  tile_descriptor_t table_allocate_descriptor;
  logic [GENERATION_W-1:0] table_allocate_slot_generation;
  logic table_mark_promoted, table_mark_cancelled, table_mark_start, table_mark_done, table_retire;
  logic [REQUEST_ID_W-1:0] table_query_id, table_route_query_id;
  logic table_query_valid, table_query_cancelled, table_query_promoted, table_query_inflight, table_query_drop;
  logic [GENERATION_W-1:0] table_query_generation, table_query_slot_generation;
  tile_descriptor_t table_query_descriptor;
  logic route_query_valid, route_query_cancelled, route_query_inflight;
  logic route_query_completed;
  logic [GENERATION_W-1:0] route_query_generation, route_query_slot_generation;
  tile_descriptor_t route_query_descriptor;
  logic ocm_allocate_valid, ocm_allocate_ready, ocm_pair_reserve_ready;
  tile_descriptor_t ocm_allocate_input, ocm_allocated_descriptor;
  logic [GENERATION_W-1:0] ocm_allocated_slot_generation;
  logic ocm_dma_write_accept;
  logic demand_enqueue_valid, demand_enqueue_ready, promoted_enqueue_valid, promoted_enqueue_ready;
  logic speculative_enqueue_valid, speculative_enqueue_ready, demand_pair_ready, speculative_pair_ready;
  descriptor_token_t table_allocate_token;
  logic descriptor_lookup_valid, descriptor_entry_valid;
  descriptor_token_t descriptor_lookup_token, claim_token, active_token, tombstone_token;
  logic claim_valid, active, tombstone_retired_valid, tombstone_retired_cancelled;
  tile_descriptor_t active_descriptor, tombstone_descriptor;
  logic [GENERATION_W-1:0] active_slot_generation, tombstone_slot_generation;
  logic [31:0] unused_burst_bytes, unused_active_remaining, unused_fifo_peak, unused_promoted_peak, unused_speculative_peak;
  logic route_matches;
  logic physical_accept;
  logic use_demand_queue;
  wire logical_is_demand = logical_gu_descriptor.request_class == REQUEST_DEMAND;
  wire descriptor_pair_credit_ready = (32'(table_free_entries) >= 2 + DEMAND_RESERVE_DESCRIPTORS) ||
    (logical_is_demand && table_free_entries >= 2);
  metadata_state_t compute_gu_state, compute_down_state;
  logic compute_gu_start, compute_down_start, compute_release;
  logic compute_gu_release;
  logic dma_drained;
  assign memory_ingress_idle = !manager_gu_valid && !manager_down_valid;
  assign memory_drained = dma_drained && 32'(table_free_entries) == DESCRIPTOR_TABLE_ENTRIES &&
    !manager_gu_valid && !manager_down_valid && !logical_valid;
  logic [15:0] compute_release_gu_slot, compute_release_down_slot, compute_release_epoch;
  logic [GENERATION_W-1:0] compute_release_gu_generation, compute_release_down_generation;
  logic compute_gu_ready_state, compute_down_ready_state;
  logic compute_gu_payload_complete, compute_down_payload_complete;
  logic [15:0] compute_gu_epoch, compute_down_epoch;
  logic [GENERATION_W-1:0] compute_gu_generation, compute_down_generation;
  logic [31:0] ocm_occupancy;
  logic [31:0] telemetry_bytes;
  logic [31:0] ocm_stale_event_count;
  logic [31:0] router_stale_event_count;
  assign stale_event_count = ocm_stale_event_count + router_stale_event_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) router_stale_event_count <= 0;
    else if (route_valid && route_ready && !route_matches)
      router_stale_event_count <= router_stale_event_count + 1;
  end

  assign table_route_query_id = route_token.request_id[REQUEST_ID_W-1:0];
  assign route_matches = 32'(route_token.request_id) < DESCRIPTOR_TABLE_ENTRIES && route_query_valid && route_query_generation == route_token.generation;
  /* An inflight true request already owns DMA and only needs OCM commit. A
     queued true request additionally needs room for its promoted token. */
  wire route_owner_active = (active || dma_done || claim_valid) && active_token == route_token;
  wire route_needs_promotion_token = !route_query_completed && !route_owner_active;
  assign route_ready = !route_valid || !route_matches || (route_matches &&
    (!route_truth || !route_needs_promotion_token || promoted_enqueue_ready));

  descriptor_manager manager (
    .clk(clk),
    .rst_n(rst_n),
    .clear(soft_flush_valid),
    .logical_valid(logical_valid && !soft_flush_valid),
    .logical_gu_descriptor(logical_gu_descriptor),
    .logical_down_descriptor(logical_down_descriptor),
    .existing_pair_live(existing_pair_live),
    .existing_pair_is_speculative(existing_pair_is_speculative),
    .logical_ready(logical_ready),
    .gu_slot_reserve_ready(ocm_pair_reserve_ready &&
      (logical_gu_descriptor.request_class || ocm_occupancy + 2 + DEMAND_RESERVE_SLOTS <= NUM_OCM_BANKS*SLOTS_PER_BANK)),
    .down_slot_reserve_ready(ocm_pair_reserve_ready),
    // 未解析的未来层 descriptor 可长期驻留，必须为当前 demand 留一对 credit。
    .descriptor_pair_reserve_ready(descriptor_pair_credit_ready),
    .queue_pair_reserve_ready(logical_is_demand ? demand_pair_ready : speculative_pair_ready),
    .promotion_ready(1'b1),
    .reserve_pair_valid(manager_reserve_pair),
    .promotion_valid(manager_promotion),
    .gu_valid(manager_gu_valid),
    .gu_descriptor(manager_gu_desc),
    .gu_ready(manager_gu_ready),
    .down_valid(manager_down_valid),
    .down_descriptor(manager_down_desc),
    .down_ready(manager_down_ready));
  assign logical_promotion_valid = manager_promotion;

  assign use_demand_queue = manager_gu_valid ? manager_gu_desc.request_class :
    manager_down_desc.request_class;
  assign physical_accept = !soft_flush_valid && (manager_gu_valid || manager_down_valid) && table_allocate_ready && ocm_allocate_ready &&
    (use_demand_queue ? demand_enqueue_ready : speculative_enqueue_ready);
  assign manager_gu_ready = manager_gu_valid && physical_accept;
  assign manager_down_ready = manager_down_valid && physical_accept;
  assign ocm_allocate_valid = physical_accept;
  assign ocm_allocate_input = manager_gu_valid ? manager_gu_desc : manager_down_desc;
  assign table_allocate_valid = physical_accept;
  assign table_allocate_descriptor = ocm_allocated_descriptor;
  assign table_allocate_slot_generation = ocm_allocated_slot_generation;
  assign table_allocate_token.request_id = {{(16-REQUEST_ID_W){1'b0}}, table_allocate_id};
  assign table_allocate_token.generation = table_allocate_generation;
  assign demand_enqueue_valid = physical_accept && use_demand_queue;
  assign speculative_enqueue_valid = physical_accept && !use_demand_queue;
  assign promoted_enqueue_valid = route_valid && route_ready && route_truth && route_matches && route_needs_promotion_token;

  descriptor_table #(.ENTRY_COUNT(DESCRIPTOR_TABLE_ENTRIES),
    .ID_W(REQUEST_ID_W),
    .GENERATION_W(GENERATION_W),
    .HOLD_FOR_ROUTE(1'b1)) descriptors (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(table_allocate_valid),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .allocate_descriptor(table_allocate_descriptor),
    .allocate_slot_generation(table_allocate_slot_generation),
    .allocate_ready(table_allocate_ready),
    .free_entry_count(table_free_entries),
    .allocate_request_id(table_allocate_id),
    .allocate_generation(table_allocate_generation),
    .mark_promoted_valid(table_mark_promoted),
    .promotion_token_created(promoted_enqueue_valid),
    .mark_promoted_id(table_route_query_id),
    .mark_promoted_generation(route_token.generation),
    .mark_cancelled_valid(table_mark_cancelled),
    .mark_cancelled_id(table_route_query_id),
    .mark_cancelled_generation(route_token.generation),
    .mark_dma_start_valid(table_mark_start),
    .mark_dma_start_id(claim_token.request_id[REQUEST_ID_W-1:0]),
    .mark_dma_start_generation(claim_token.generation),
    .mark_dma_done_valid(table_mark_done),
    .mark_dma_done_id(active_token.request_id[REQUEST_ID_W-1:0]),
    .mark_dma_done_generation(active_token.generation),
    .retire_refs_valid(table_retire),
    .retire_refs_id(dma_done ? active_token.request_id[REQUEST_ID_W-1:0] : tombstone_token.request_id[REQUEST_ID_W-1:0]),
    .retire_refs_generation(dma_done ? active_token.generation : tombstone_token.generation),
    .stale_event(),
    .query_id(table_query_id),
    .query_valid(table_query_valid),
    .query_generation(table_query_generation),
    .query_cancelled(table_query_cancelled),
    .query_promoted(table_query_promoted),
    .query_inflight(table_query_inflight),
    .query_drop_on_complete(table_query_drop),
    .query_slot_generation(table_query_slot_generation),
    .query_descriptor(table_query_descriptor),
    .route_query_id(table_route_query_id),
    .route_query_valid(route_query_valid),
    .route_query_generation(route_query_generation),
    .route_query_cancelled(route_query_cancelled),
    .route_query_inflight(route_query_inflight),
    .route_query_completed(route_query_completed),
    .route_query_slot_generation(route_query_slot_generation),
    .route_query_descriptor(route_query_descriptor));
  assign table_mark_promoted = route_valid && route_ready && route_truth && route_matches;
  assign table_mark_cancelled = route_valid && route_ready && !route_truth && route_matches;
  assign table_mark_start = claim_valid;
  assign table_mark_done = dma_done;
  assign table_retire = dma_done || tombstone_retired_valid;

  ocm_slot_ownership #(.NUM_OCM_BANKS(NUM_OCM_BANKS),
    .SLOTS_PER_BANK(SLOTS_PER_BANK),
    .GENERATION_W(GENERATION_W),
    .USE_PAIR_PRIMARY(1'b1)) ocm (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(ocm_allocate_valid),
    .allocate_descriptor(ocm_allocate_input),
    .pair_check_descriptor(logical_down_descriptor),
    .pair_primary_descriptor(logical_gu_descriptor),
    .allocate_ready(ocm_allocate_ready),
    .pair_reserve_ready(ocm_pair_reserve_ready),
    .allocated_descriptor(ocm_allocated_descriptor),
    .allocated_slot_generation(ocm_allocated_slot_generation),
    .dma_start_valid(claim_valid),
    .dma_start_slot(active_descriptor.ocm_slot),
    .dma_start_epoch(active_descriptor.epoch_id),
    .dma_start_slot_generation(active_slot_generation),
    .dma_write_valid(dma_burst_done),
    .dma_write_last(dma_done),
    .dma_write_slot(active_descriptor.ocm_slot),
    .dma_write_epoch(active_descriptor.epoch_id),
    .dma_write_slot_generation(active_slot_generation),
    .dma_write_accept(ocm_dma_write_accept),
    .commit_valid(route_valid && route_ready && route_truth && route_matches),
    .commit_slot(route_query_descriptor.ocm_slot),
    .commit_epoch(route_query_descriptor.epoch_id),
    .commit_slot_generation(route_query_slot_generation),
    .cancel_valid(route_valid && route_ready && !route_truth && route_matches),
    .cancel_slot(route_query_descriptor.ocm_slot),
    .cancel_epoch(route_query_descriptor.epoch_id),
    .cancel_slot_generation(route_query_slot_generation),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .retire_cancelled_valid(tombstone_retired_valid && tombstone_retired_cancelled),
    .retire_cancelled_slot(tombstone_descriptor.ocm_slot),
    .retire_cancelled_epoch(tombstone_descriptor.epoch_id),
    .retire_cancelled_slot_generation(tombstone_slot_generation),
    .consumer_release_valid(compute_gu_release_valid),
    .consumer_release_slot(compute_release_gu_slot),
    .consumer_release_epoch(compute_release_epoch),
    .consumer_release_slot_generation(compute_release_gu_generation),
    .consumer_release2_valid(compute_release),
    .consumer_release2_slot(compute_release_down_slot),
    .consumer_release2_epoch(compute_release_epoch),
    .consumer_release2_slot_generation(compute_release_down_generation),
    .compute_start_valid(compute_gu_start),
    .compute_start_slot(compute_gu_descriptor.ocm_slot),
    .compute_start_epoch(compute_gu_descriptor.epoch_id),
    .compute_start_slot_generation(compute_gu_slot_generation),
    .compute_start2_valid(compute_down_start),
    .compute_start2_slot(compute_down_descriptor.ocm_slot),
    .compute_start2_epoch(compute_down_descriptor.epoch_id),
    .compute_start2_slot_generation(compute_down_slot_generation),
    .query_slot(compute_gu_descriptor.ocm_slot),
    .query_state(compute_gu_state),
    .query_epoch(compute_gu_epoch),
    .query_slot_generation(compute_gu_generation),
    .query_payload_complete(compute_gu_payload_complete),
    .query_quarantined(),
    .query2_slot(compute_down_descriptor.ocm_slot),
    .query2_state(compute_down_state),
    .query2_epoch(compute_down_epoch),
    .query2_slot_generation(compute_down_generation),
    .query2_payload_complete(compute_down_payload_complete),
    .query2_quarantined(),
    .stale_event_count(ocm_stale_event_count),
    .discarded_dma_write_count(discarded_dma_write_count),
    .occupancy(ocm_occupancy));

  assign table_query_id = descriptor_lookup_token.request_id[REQUEST_ID_W-1:0];
  assign descriptor_entry_valid = table_query_valid && table_query_generation == descriptor_lookup_token.generation;
  burst_dma_channel #(.FIFO_DEPTH(DMA_FIFO_DEPTH),
    .PROGRESS_ENTRIES(DESCRIPTOR_TABLE_ENTRIES),
    .MAX_BURST_BYTES(MAX_BURST_BYTES),
    .EXTERNAL_BURST_COMPLETION(USE_EXTERNAL_DMA)) dma (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
    .setup_cycles(setup_cycles),
    .bytes_per_cycle(bytes_per_cycle),
    .external_burst_done(dma_external_burst_done),
    .promotion_notify_valid(table_mark_promoted),
    .promotion_notify_token(route_token),
    .demand_enqueue_valid(demand_enqueue_valid),
    .demand_enqueue_token(table_allocate_token),
    .demand_enqueue_ready(demand_enqueue_ready),
    .promoted_enqueue_valid(promoted_enqueue_valid),
    .promoted_enqueue_token(route_token),
    .promoted_enqueue_ready(promoted_enqueue_ready),
    .speculative_enqueue_valid(speculative_enqueue_valid),
    .speculative_enqueue_token(table_allocate_token),
    .speculative_enqueue_ready(speculative_enqueue_ready),
    .demand_pair_reserve_ready(demand_pair_ready),
    .speculative_pair_reserve_ready(speculative_pair_ready),
    .descriptor_lookup_valid(descriptor_lookup_valid),
    .descriptor_lookup_token(descriptor_lookup_token),
    .descriptor_entry_valid(descriptor_entry_valid),
    .descriptor_entry_cancelled(table_query_cancelled),
    .descriptor_entry_promoted(table_query_promoted),
    .descriptor_entry_inflight(table_query_inflight),
    .descriptor_lookup_slot_generation(table_query_slot_generation),
    .descriptor_lookup_descriptor(table_query_descriptor),
    .dma_claim_valid(claim_valid),
    .dma_claim_token(claim_token),
    .dma_burst_start(dma_burst_start),
    .dma_burst_done(dma_burst_done),
    .dma_burst_bytes(dma_burst_bytes),
    .dma_done(dma_done),
    .active(active),
    .drained(dma_drained),
    .active_token(active_token),
    .active_descriptor(active_descriptor),
    .active_slot_generation(active_slot_generation),
    .active_remaining_bytes(unused_active_remaining),
    .demand_stall_cycles(demand_stall_cycles),
    .demand_fifo_peak(unused_fifo_peak),
    .promoted_fifo_peak(unused_promoted_peak),
    .speculative_fifo_peak(unused_speculative_peak),
    .tombstone_skip_count(tombstone_skip_count),
    .tombstone_retired_valid(tombstone_retired_valid),
    .tombstone_retired_cancelled(tombstone_retired_cancelled),
    .tombstone_retired_token(tombstone_token),
    .tombstone_retired_descriptor(tombstone_descriptor),
    .tombstone_retired_slot_generation(tombstone_slot_generation));
  assign dma_active_descriptor = active_descriptor;
  assign dma_active_token = active_token;
  assign dma_active_slot_generation = active_slot_generation;
  assign dma_active_remaining_bytes = unused_active_remaining;
  always_comb begin
    telemetry_bytes = active_descriptor.byte_count;
    if (physical_accept) telemetry_bytes = ocm_allocated_descriptor.byte_count;
    else if (route_valid && route_matches) telemetry_bytes = route_query_descriptor.byte_count;
  end
  /* Explicit state encodings avoid tool-specific enum-expression coercions on
     unpacked metadata RAM read ports. 5=RESIDENT_SPEC, 6=COMMITTED. */
  assign compute_gu_ready_state = compute_gu_state == META_COMMITTED && compute_gu_payload_complete &&
    compute_gu_epoch == compute_gu_descriptor.epoch_id && compute_gu_generation == compute_gu_slot_generation;
  assign compute_down_ready_state = compute_down_state == META_COMMITTED && compute_down_payload_complete &&
    compute_down_epoch == compute_down_descriptor.epoch_id && compute_down_generation == compute_down_slot_generation;
  assign backend_gu_start = compute_gu_start;
  assign backend_down_start = compute_down_start;
  assign compute_gu_release_valid = compute_gu_release && !soft_flush_valid;
  tile_compute_controller #(.GENERATION_W(GENERATION_W),
    .EXTERNAL_COMPUTE(USE_REAL_COMPUTE)) compute (
    .clk(clk),
    .rst_n(rst_n),
    .clear(soft_flush_valid),
    .compute_req_valid(compute_req_valid && !soft_flush_valid),
    .compute_req_ready(compute_req_ready),
    .compute_gu_descriptor(compute_gu_descriptor),
    .compute_gu_slot_generation(compute_gu_slot_generation),
    .compute_down_descriptor(compute_down_descriptor),
    .compute_down_slot_generation(compute_down_slot_generation),
    .gu_resident(compute_gu_ready_state),
    .down_resident(compute_down_ready_state),
    .gu_cycles(compute_gu_cycles),
    .nonlinear_cycles(compute_nonlinear_cycles),
    .down_cycles(compute_down_cycles),
    .backend_gu_done(backend_gu_done),
    .backend_nonlinear_done(backend_nonlinear_done),
    .backend_down_done(backend_down_done),
    .busy(compute_busy),
    .done(compute_done),
    .waiting_gu(compute_waiting_gu),
    .waiting_down(compute_waiting_down),
    .gu_compute_start(compute_gu_start),
    .down_compute_start(compute_down_start),
    .consumer_gu_release_valid(compute_gu_release),
    .consumer_release_valid(compute_release),
    .consumer_release_gu_slot(compute_release_gu_slot),
    .consumer_release_down_slot(compute_release_down_slot),
    .consumer_release_epoch(compute_release_epoch),
    .consumer_release_gu_generation(compute_release_gu_generation),
    .consumer_release_down_generation(compute_release_down_generation));
  telemetry telemetry_counters (
    .clk(clk),
    .rst_n(rst_n),
    .sequence_active(sequence_active),
    .layer_done(layer_done),
    .speculative_alloc(physical_accept_valid && !physical_accept_descriptor.request_class),
    .speculative_issue(dma_burst_done && !active_descriptor.request_class),
    .demand_issue(dma_burst_done && active_descriptor.request_class),
    .issue_bytes(dma_burst_bytes),
    /* Classification after a route/DMA race is intentionally deferred; this
       counter remains zero until the route resolver finalizer is added. */
    .speculative_useful(1'b0),
    .speculative_wasted(1'b0),
    .speculative_cancel(route_valid && route_ready && !route_truth && route_matches && !route_query_inflight),
    .inflight_wasted(route_valid && route_ready && !route_truth && route_matches && route_query_inflight),
    .demand_stall(1'b0),
    .compute_wait_gu(compute_waiting_gu),
    .compute_wait_down(compute_waiting_down),
    .scheduler_active(scheduler_active),
    .allocation(physical_accept_valid),
    .commit(table_mark_promoted),
    .cancel(table_mark_cancelled),
    .bytes(physical_accept_valid ? physical_accept_descriptor.byte_count : telemetry_bytes),
    .ocm_occupancy(ocm_occupancy),
    .total_cycles(telemetry_total_cycles),
    .completed_layers(),
    .speculative_allocated_bytes(telemetry_speculative_allocated_bytes),
    .speculative_issued_bytes(telemetry_speculative_issued_bytes),
    .demand_issued_bytes(telemetry_demand_issued_bytes),
    .useful_speculative_bytes(),
    .wasted_speculative_bytes(),
    .cancelled_before_issue_bytes(),
    .inflight_wasted_bytes(),
    .demand_stall_cycles(),
    .compute_wait_gu_cycles(telemetry_compute_wait_gu_cycles),
    .compute_wait_down_cycles(telemetry_compute_wait_down_cycles),
    .ocm_peak_occupancy(telemetry_ocm_peak_occupancy),
    .scheduler_cycles(),
    .allocation_count(),
    .commit_count(telemetry_commit_count),
    .cancel_count(telemetry_cancel_count));
  speculative_byte_finalizer #(
    .ENTRIES(DESCRIPTOR_TABLE_ENTRIES)
  ) byte_finalizer (
    .clk(clk),
    .rst_n(rst_n),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .capture_valid(physical_accept_valid && !physical_accept_descriptor.request_class),
    .capture_token(physical_accept_token),
    .capture_bytes(physical_accept_descriptor.byte_count),
    .capture_epoch(physical_accept_descriptor.epoch_id),
    .burst_done(dma_burst_done),
    .burst_token(active_token),
    .burst_bytes(dma_burst_bytes),
    .dma_done(dma_done),
    .route_valid(route_valid && route_ready && route_matches),
    .route_token(route_token),
    .route_truth(route_truth),
    .cancelled_retired(tombstone_retired_valid && tombstone_retired_cancelled),
    .retired_token(tombstone_token),
    .useful_bytes(telemetry_useful_speculative_bytes),
    .wasted_bytes(telemetry_wasted_speculative_bytes),
    .cancelled_bytes(telemetry_cancelled_speculative_bytes)
  );
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      physical_accept_valid <= 1'b0;
      physical_accept_token <= '0;
      physical_accept_descriptor <= '0;
      physical_accept_slot_generation <= '0;
    end else begin
      physical_accept_valid <= physical_accept;
      if (physical_accept) begin
        physical_accept_token <= table_allocate_token;
        physical_accept_descriptor <= ocm_allocated_descriptor;
        physical_accept_slot_generation <= ocm_allocated_slot_generation;
      end
    end
  end
endmodule
