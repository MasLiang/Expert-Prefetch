`timescale 1ns/1ps
/**
 * 动态 OCM 槽位所有权与 DMA 写保护。
 *
 * 本模块只保存 metadata；payload RAM 可在后续替换为具有同一 write_guard ABI 的
 * 实现。每个槽位的 (epoch_id, slot_generation) 是写回、flush 与释放的唯一
 * ownership tag。软 flush 绝不会提前释放 inflight 槽位。
 */
module ocm_slot_ownership #(
  parameter integer NUM_OCM_BANKS = 4,
  parameter integer SLOTS_PER_BANK = 32,
  parameter integer SLOT_W = (SLOTS_PER_BANK > 1) ? $clog2(SLOTS_PER_BANK) : 1,
  parameter integer BANK_W = (NUM_OCM_BANKS > 1) ? $clog2(NUM_OCM_BANKS) : 1,
  parameter integer SLOT_COUNT = NUM_OCM_BANKS * SLOTS_PER_BANK,
  parameter integer GENERATION_W = 8,
  parameter bit USE_PAIR_PRIMARY = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic allocate_valid,
  input tile_prefetch_package::tile_descriptor_t allocate_descriptor,
  input tile_prefetch_package::tile_descriptor_t pair_check_descriptor,
  input tile_prefetch_package::tile_descriptor_t pair_primary_descriptor,
  output logic allocate_ready,
  output logic pair_reserve_ready,
  output tile_prefetch_package::tile_descriptor_t allocated_descriptor,
  output logic [GENERATION_W-1:0] allocated_slot_generation,
  input logic dma_start_valid,
  input logic [15:0] dma_start_slot,
  input logic [15:0] dma_start_epoch,
  input logic [GENERATION_W-1:0] dma_start_slot_generation,
  input logic dma_write_valid,
  input logic dma_write_last,
  input logic [15:0] dma_write_slot,
  input logic [15:0] dma_write_epoch,
  input logic [GENERATION_W-1:0] dma_write_slot_generation,
  output logic dma_write_accept,
  input logic commit_valid,
  input logic [15:0] commit_slot,
  input logic [15:0] commit_epoch,
  input logic [GENERATION_W-1:0] commit_slot_generation,
  input logic cancel_valid,
  input logic [15:0] cancel_slot,
  input logic [15:0] cancel_epoch,
  input logic [GENERATION_W-1:0] cancel_slot_generation,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic retire_cancelled_valid,
  input logic [15:0] retire_cancelled_slot,
  input logic [15:0] retire_cancelled_epoch,
  input logic [GENERATION_W-1:0] retire_cancelled_slot_generation,
  input logic consumer_release_valid,
  input logic [15:0] consumer_release_slot,
  input logic [15:0] consumer_release_epoch,
  input logic [GENERATION_W-1:0] consumer_release_slot_generation,
  input logic consumer_release2_valid,
  input logic [15:0] consumer_release2_slot,
  input logic [15:0] consumer_release2_epoch,
  input logic [GENERATION_W-1:0] consumer_release2_slot_generation,
  input logic compute_start_valid,
  input logic [15:0] compute_start_slot,
  input logic [15:0] compute_start_epoch,
  input logic [GENERATION_W-1:0] compute_start_slot_generation,
  input logic compute_start2_valid,
  input logic [15:0] compute_start2_slot,
  input logic [15:0] compute_start2_epoch,
  input logic [GENERATION_W-1:0] compute_start2_slot_generation,
  input logic [15:0] query_slot,
  output tile_prefetch_package::metadata_state_t query_state,
  output logic [15:0] query_epoch,
  output logic [GENERATION_W-1:0] query_slot_generation,
  output logic query_quarantined,
  output logic query_payload_complete,
  input logic [15:0] query2_slot,
  output tile_prefetch_package::metadata_state_t query2_state,
  output logic [15:0] query2_epoch,
  output logic [GENERATION_W-1:0] query2_slot_generation,
  output logic query2_quarantined,
  output logic query2_payload_complete,
  output logic [31:0] stale_event_count,
  output logic [31:0] discarded_dma_write_count,
  output logic [31:0] occupancy
);
  import tile_prefetch_package::*;
  localparam integer TOTAL_SLOT_W = (SLOT_COUNT > 1) ? $clog2(SLOT_COUNT) : 1;
  localparam integer OCCUPANCY_W = $clog2(SLOT_COUNT + 1);
  localparam integer BANK_TREE_NODES = 2 * SLOTS_PER_BANK;
  metadata_state_t state_mem [0:SLOT_COUNT-1];
  logic [15:0] epoch_mem [0:SLOT_COUNT-1];
  logic [GENERATION_W-1:0] generation_mem [0:SLOT_COUNT-1];
  logic quarantined_mem [0:SLOT_COUNT-1];
  logic cancelled_mem [0:SLOT_COUNT-1];
  logic payload_complete_mem [0:SLOT_COUNT-1];
  metadata_state_t state_next [0:SLOT_COUNT-1];
  metadata_state_t state_effective [0:SLOT_COUNT-1];
  logic [15:0] epoch_next [0:SLOT_COUNT-1];
  logic [GENERATION_W-1:0] generation_next [0:SLOT_COUNT-1];
  logic quarantined_next [0:SLOT_COUNT-1];
  logic cancelled_next [0:SLOT_COUNT-1];
  logic cancelled_effective [0:SLOT_COUNT-1];
  logic payload_complete_next [0:SLOT_COUNT-1];
  logic [31:0] stale_event_count_next;
  logic [31:0] discarded_dma_write_count_next;
  integer next_index;
  integer flush_index;
  logic [BANK_W-1:0] allocate_bank;
  logic [BANK_W-1:0] pair_check_bank;
  logic [BANK_W-1:0] pair_primary_bank;
  logic [SLOT_W-1:0] allocate_local_slot;
  logic allocate_found;
  logic [TOTAL_SLOT_W-1:0] allocate_slot_index;
  logic [TOTAL_SLOT_W-1:0] dma_start_index;
  logic [TOTAL_SLOT_W-1:0] dma_write_index;
  logic [TOTAL_SLOT_W-1:0] retire_index;
  logic [TOTAL_SLOT_W-1:0] release_index;
  logic [TOTAL_SLOT_W-1:0] compute_start_index;
  logic [TOTAL_SLOT_W-1:0] release2_index;
  logic [TOTAL_SLOT_W-1:0] compute_start2_index;
  logic [TOTAL_SLOT_W-1:0] commit_index;
  logic [TOTAL_SLOT_W-1:0] cancel_index;
  logic [15:0] allocate_hash;
  logic [15:0] pair_check_hash;
  logic [15:0] pair_primary_hash;
  // 每个节点由独立 continuous assignment 驱动；split_var 保留元素级 DAG。
  wire bank_has_free [0:NUM_OCM_BANKS*BANK_TREE_NODES-1] /* verilator split_var */;
  wire bank_has_two_free [0:NUM_OCM_BANKS*BANK_TREE_NODES-1] /* verilator split_var */;
  wire [SLOT_W-1:0] bank_first_free [0:NUM_OCM_BANKS*BANK_TREE_NODES-1] /* verilator split_var */;
  wire [NUM_OCM_BANKS-1:0] root_has_free;
  wire [NUM_OCM_BANKS-1:0] root_has_two_free;
  wire [SLOT_W-1:0] root_first_free [0:NUM_OCM_BANKS-1];
  wire [OCCUPANCY_W-1:0] occupied_tree [1:2*SLOT_COUNT-1] /* verilator split_var */;

  initial begin
    if ((NUM_OCM_BANKS & (NUM_OCM_BANKS - 1)) != 0)
      $error("NUM_OCM_BANKS must be a power of two");
    if ((SLOTS_PER_BANK & (SLOTS_PER_BANK - 1)) != 0)
      $error("SLOTS_PER_BANK must be a power of two");
  end

  for (genvar bank = 0; bank < NUM_OCM_BANKS; bank = bank + 1) begin : free_bank_tree
    localparam integer BASE = bank * BANK_TREE_NODES;
    assign bank_has_free[BASE] = 1'b0;
    assign bank_has_two_free[BASE] = 1'b0;
    assign bank_first_free[BASE] = '0;
    assign root_has_free[bank] = bank_has_free[BASE+1];
    assign root_has_two_free[bank] = bank_has_two_free[BASE+1];
    assign root_first_free[bank] = bank_first_free[BASE+1];
    for (genvar leaf = 0; leaf < SLOTS_PER_BANK; leaf = leaf + 1) begin : leaves
      assign bank_has_free[BASE+SLOTS_PER_BANK+leaf] =
        state_mem[bank*SLOTS_PER_BANK+leaf] == META_FREE;
      assign bank_has_two_free[BASE+SLOTS_PER_BANK+leaf] = 1'b0;
      assign bank_first_free[BASE+SLOTS_PER_BANK+leaf] = SLOT_W'(leaf);
    end
    for (genvar node = 1; node < SLOTS_PER_BANK; node = node + 1) begin : branches
      assign bank_has_free[BASE+node] = bank_has_free[BASE+2*node] || bank_has_free[BASE+2*node+1];
      assign bank_has_two_free[BASE+node] = bank_has_two_free[BASE+2*node] ||
        bank_has_two_free[BASE+2*node+1] ||
        (bank_has_free[BASE+2*node] && bank_has_free[BASE+2*node+1]);
      assign bank_first_free[BASE+node] = bank_has_free[BASE+2*node] ?
        bank_first_free[BASE+2*node] : bank_first_free[BASE+2*node+1];
    end
  end

  for (genvar leaf = 0; leaf < SLOT_COUNT; leaf = leaf + 1) begin : occupied_leaves
    assign occupied_tree[SLOT_COUNT+leaf] = OCCUPANCY_W'(state_mem[leaf] != META_FREE);
  end
  for (genvar node = 1; node < SLOT_COUNT; node = node + 1) begin : occupied_branches
    assign occupied_tree[node] = occupied_tree[2*node] + occupied_tree[2*node+1];
  end

  always_comb begin
    allocate_hash = allocate_descriptor.epoch_id ^ allocate_descriptor.layer_id ^
      {{(16-10){1'b0}}, allocate_descriptor.expert_id} ^ {{(16-8){1'b0}}, allocate_descriptor.tile_id} ^
      16'(allocate_descriptor.matrix_kind);
    pair_check_hash = pair_check_descriptor.epoch_id ^ pair_check_descriptor.layer_id ^
      {{(16-10){1'b0}}, pair_check_descriptor.expert_id} ^ {{(16-8){1'b0}}, pair_check_descriptor.tile_id} ^
      16'(pair_check_descriptor.matrix_kind);
    pair_primary_hash = pair_primary_descriptor.epoch_id ^ pair_primary_descriptor.layer_id ^
      16'(pair_primary_descriptor.expert_id) ^ 16'(pair_primary_descriptor.tile_id) ^ 16'(pair_primary_descriptor.matrix_kind);
    allocate_bank = (NUM_OCM_BANKS == 1) ? '0 : allocate_hash[BANK_W-1:0];
    pair_check_bank = (NUM_OCM_BANKS == 1) ? '0 : pair_check_hash[BANK_W-1:0];
    pair_primary_bank = USE_PAIR_PRIMARY ?
      ((NUM_OCM_BANKS == 1) ? '0 : pair_primary_hash[BANK_W-1:0]) : allocate_bank;
    allocate_found = root_has_free[allocate_bank];
    allocate_local_slot = allocate_found ? root_first_free[allocate_bank] : '0;
    allocate_slot_index = TOTAL_SLOT_W'(32'(allocate_bank)*SLOTS_PER_BANK + 32'(allocate_local_slot));
    allocate_ready = allocate_found;
    pair_reserve_ready = (pair_primary_bank == pair_check_bank) ?
      root_has_two_free[pair_primary_bank] :
      (root_has_free[pair_primary_bank] && root_has_free[pair_check_bank]);
    allocated_descriptor = allocate_descriptor;
    allocated_descriptor.ocm_bank = 4'(allocate_bank);
    allocated_descriptor.ocm_slot = {{(16-TOTAL_SLOT_W){1'b0}}, allocate_slot_index};
    allocated_slot_generation = generation_mem[allocate_slot_index] + 1'b1;

    dma_start_index = dma_start_slot[TOTAL_SLOT_W-1:0];
    dma_write_index = dma_write_slot[TOTAL_SLOT_W-1:0];
    retire_index = retire_cancelled_slot[TOTAL_SLOT_W-1:0];
    release_index = consumer_release_slot[TOTAL_SLOT_W-1:0];
    compute_start_index = compute_start_slot[TOTAL_SLOT_W-1:0];
    release2_index = consumer_release2_slot[TOTAL_SLOT_W-1:0];
    compute_start2_index = compute_start2_slot[TOTAL_SLOT_W-1:0];
    commit_index = commit_slot[TOTAL_SLOT_W-1:0];
    cancel_index = cancel_slot[TOTAL_SLOT_W-1:0];
    if (32'(query_slot) < SLOT_COUNT)
      query_state = state_mem[query_slot[TOTAL_SLOT_W-1:0]];
    else
      query_state = META_FREE;
    query_epoch = (32'(query_slot) < SLOT_COUNT) ? epoch_mem[query_slot[TOTAL_SLOT_W-1:0]] : '0;
    query_slot_generation = (32'(query_slot) < SLOT_COUNT) ? generation_mem[query_slot[TOTAL_SLOT_W-1:0]] : '0;
    query_quarantined = (32'(query_slot) < SLOT_COUNT) ? quarantined_mem[query_slot[TOTAL_SLOT_W-1:0]] : '0;
    query_payload_complete = (32'(query_slot) < SLOT_COUNT) ? payload_complete_mem[query_slot[TOTAL_SLOT_W-1:0]] : '0;
    if (32'(query2_slot) < SLOT_COUNT)
      query2_state = state_mem[query2_slot[TOTAL_SLOT_W-1:0]];
    else
      query2_state = META_FREE;
    query2_epoch = (32'(query2_slot) < SLOT_COUNT) ? epoch_mem[query2_slot[TOTAL_SLOT_W-1:0]] : '0;
    query2_slot_generation = (32'(query2_slot) < SLOT_COUNT) ? generation_mem[query2_slot[TOTAL_SLOT_W-1:0]] : '0;
    query2_quarantined = (32'(query2_slot) < SLOT_COUNT) ? quarantined_mem[query2_slot[TOTAL_SLOT_W-1:0]] : '0;
    query2_payload_complete = (32'(query2_slot) < SLOT_COUNT) ? payload_complete_mem[query2_slot[TOTAL_SLOT_W-1:0]] : '0;
    dma_write_accept = dma_write_valid && 32'(dma_write_slot) < SLOT_COUNT &&
      epoch_mem[dma_write_index] == dma_write_epoch &&
      generation_mem[dma_write_index] == dma_write_slot_generation &&
      (state_mem[dma_write_index] == META_INFLIGHT_SPEC ||
       state_mem[dma_write_index] == META_INFLIGHT_DEMAND ||
       state_mem[dma_write_index] == META_COMMITTED ||
       state_mem[dma_write_index] == META_DROP_ON_COMPLETE);
    occupancy = 32'(occupied_tree[1]);
  end

  /*
   * 同槽更新顺序：allocate/start/commit/cancel -> flush -> retire ->
   * final DMA write -> release/compute。事件合法性默认观察周期开始时状态；
   * flush-effective 视图仅供后续 retire/release/compute，防止已 flush 的
   * COMMITTED 槽被重新置为 RUNNING。DMA final 可在 flush 同拍安全退休。
   */
  always_comb begin
    stale_event_count_next = stale_event_count;
    discarded_dma_write_count_next = discarded_dma_write_count;
    for (next_index = 0; next_index < SLOT_COUNT; next_index = next_index + 1) begin
      state_next[next_index] = state_mem[next_index];
      state_effective[next_index] = state_mem[next_index];
      epoch_next[next_index] = epoch_mem[next_index];
      generation_next[next_index] = generation_mem[next_index];
      quarantined_next[next_index] = quarantined_mem[next_index];
      cancelled_next[next_index] = cancelled_mem[next_index];
      cancelled_effective[next_index] = cancelled_mem[next_index];
      payload_complete_next[next_index] = payload_complete_mem[next_index];
    end
      if (allocate_valid && allocate_ready) begin
        if (allocate_descriptor.request_class == REQUEST_DEMAND)
          state_next[allocate_slot_index] = META_QUEUED_DEMAND;
        else
          state_next[allocate_slot_index] = META_QUEUED_SPEC;
        epoch_next[allocate_slot_index] = allocate_descriptor.epoch_id;
        generation_next[allocate_slot_index] = generation_mem[allocate_slot_index] + 1'b1;
        quarantined_next[allocate_slot_index] = 1'b0;
        cancelled_next[allocate_slot_index] = 1'b0;
        payload_complete_next[allocate_slot_index] = 1'b0;
      end
      if (dma_start_valid) begin
        if (32'(dma_start_slot) >= SLOT_COUNT || epoch_mem[dma_start_index] != dma_start_epoch ||
            generation_mem[dma_start_index] != dma_start_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_mem[dma_start_index] == META_QUEUED_SPEC) begin
          state_next[dma_start_index] = META_INFLIGHT_SPEC;
        end else if (state_mem[dma_start_index] == META_QUEUED_DEMAND) begin
          state_next[dma_start_index] = META_INFLIGHT_DEMAND;
        end else if (state_mem[dma_start_index] == META_INFLIGHT_SPEC ||
                     state_mem[dma_start_index] == META_INFLIGHT_DEMAND ||
                     state_mem[dma_start_index] == META_COMMITTED ||
                     state_mem[dma_start_index] == META_DROP_ON_COMPLETE) begin
          /* A paused descriptor resumes with its original ownership tag. */
        end else begin
          stale_event_count_next = stale_event_count + 1'b1;
        end
      end
      if (commit_valid) begin
        if (32'(commit_slot) >= SLOT_COUNT || epoch_mem[commit_index] != commit_epoch ||
            generation_mem[commit_index] != commit_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_mem[commit_index] == META_QUEUED_SPEC) begin
          state_next[commit_index] = META_QUEUED_DEMAND;
        end else if (state_mem[commit_index] == META_INFLIGHT_SPEC ||
                     state_mem[commit_index] == META_RESIDENT_SPEC) begin
          state_next[commit_index] = META_COMMITTED;
        end
      end
      if (cancel_valid) begin
        if (32'(cancel_slot) >= SLOT_COUNT || epoch_mem[cancel_index] != cancel_epoch ||
            generation_mem[cancel_index] != cancel_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_mem[cancel_index] == META_QUEUED_SPEC ||
                     state_mem[cancel_index] == META_QUEUED_DEMAND) begin
          cancelled_next[cancel_index] = 1'b1;
        end else if (state_mem[cancel_index] == META_INFLIGHT_SPEC ||
                     state_mem[cancel_index] == META_INFLIGHT_DEMAND ||
                     state_mem[cancel_index] == META_COMMITTED) begin
          state_next[cancel_index] = META_DROP_ON_COMPLETE;
          quarantined_next[cancel_index] = 1'b1;
        end else if (state_mem[cancel_index] == META_RESIDENT_SPEC) begin
          state_next[cancel_index] = META_FREE;
        end
      end
      if (soft_flush_valid) begin
        for (flush_index = 0; flush_index < SLOT_COUNT;
      flush_index = flush_index + 1) begin
          if (epoch_mem[flush_index] == soft_flush_epoch) begin
            if (dma_start_valid && 32'(dma_start_slot) == flush_index &&
                dma_start_epoch == epoch_mem[flush_index] &&
                dma_start_slot_generation == generation_mem[flush_index] &&
                (state_mem[flush_index] == META_QUEUED_SPEC ||
                 state_mem[flush_index] == META_QUEUED_DEMAND ||
                 state_mem[flush_index] == META_INFLIGHT_SPEC ||
                 state_mem[flush_index] == META_INFLIGHT_DEMAND ||
                 state_mem[flush_index] == META_COMMITTED ||
                 state_mem[flush_index] == META_DROP_ON_COMPLETE)) begin
              state_next[flush_index] = META_DROP_ON_COMPLETE;
              state_effective[flush_index] = META_DROP_ON_COMPLETE;
              quarantined_next[flush_index] = 1;
            end else if (state_mem[flush_index] == META_QUEUED_SPEC || state_mem[flush_index] == META_QUEUED_DEMAND) begin
              cancelled_next[flush_index] = 1'b1;
              cancelled_effective[flush_index] = 1'b1;
            end else if (state_mem[flush_index] == META_INFLIGHT_SPEC ||
                         state_mem[flush_index] == META_INFLIGHT_DEMAND ||
                         (state_mem[flush_index] == META_COMMITTED && !payload_complete_mem[flush_index])) begin
              state_next[flush_index] = META_DROP_ON_COMPLETE;
              state_effective[flush_index] = META_DROP_ON_COMPLETE;
              quarantined_next[flush_index] = 1'b1;
            end else if (state_mem[flush_index] == META_RESIDENT_SPEC ||
                         state_mem[flush_index] == META_COMMITTED ||
                         state_mem[flush_index] == META_RUNNING ||
                         state_mem[flush_index] == META_EVICTABLE) begin
              state_next[flush_index] = META_FREE;
              state_effective[flush_index] = META_FREE;
            end
          end
        end
      end
      if (retire_cancelled_valid) begin
        if (32'(retire_cancelled_slot) >= SLOT_COUNT || epoch_mem[retire_index] != retire_cancelled_epoch ||
            generation_mem[retire_index] != retire_cancelled_slot_generation || !cancelled_effective[retire_index]) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else begin
          state_next[retire_index] = META_FREE;
          cancelled_next[retire_index] = 1'b0;
        end
      end
      if (dma_write_valid) begin
        if (!dma_write_accept) begin
          stale_event_count_next = stale_event_count + 1'b1;
          discarded_dma_write_count_next = discarded_dma_write_count + 1'b1;
        end else if (dma_write_last) begin
          payload_complete_next[dma_write_index] = 1'b1;
          if (cancelled_effective[dma_write_index] || (cancel_valid && cancel_slot == dma_write_slot &&
                cancel_epoch == dma_write_epoch && cancel_slot_generation == dma_write_slot_generation) ||
              (soft_flush_valid && soft_flush_epoch == dma_write_epoch)) begin
            state_next[dma_write_index] = META_FREE;
            quarantined_next[dma_write_index] = 0;
          end else if (commit_valid && commit_slot == dma_write_slot &&
              commit_epoch == dma_write_epoch && commit_slot_generation == dma_write_slot_generation) begin
            state_next[dma_write_index] = META_COMMITTED;
          end else if (state_effective[dma_write_index] == META_INFLIGHT_SPEC) begin
            state_next[dma_write_index] = META_RESIDENT_SPEC;
          end else if (state_effective[dma_write_index] == META_INFLIGHT_DEMAND ||
                       state_effective[dma_write_index] == META_COMMITTED) begin
            state_next[dma_write_index] = META_COMMITTED;
          end else if (state_effective[dma_write_index] == META_DROP_ON_COMPLETE) begin
            state_next[dma_write_index] = META_FREE;
            quarantined_next[dma_write_index] = 1'b0;
          end
        end
      end
      if (consumer_release_valid) begin
        if (32'(consumer_release_slot) >= SLOT_COUNT || epoch_mem[release_index] != consumer_release_epoch ||
            generation_mem[release_index] != consumer_release_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_effective[release_index] == META_RUNNING || state_effective[release_index] == META_COMMITTED ||
                     state_effective[release_index] == META_EVICTABLE || state_effective[release_index] == META_RESIDENT_SPEC) begin
          state_next[release_index] = META_FREE;
          quarantined_next[release_index] = 1'b0;
        end else begin
          stale_event_count_next = stale_event_count + 1'b1;
        end
      end
      if (compute_start_valid) begin
        if (32'(compute_start_slot) >= SLOT_COUNT || epoch_mem[compute_start_index] != compute_start_epoch ||
            generation_mem[compute_start_index] != compute_start_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_effective[compute_start_index] == META_COMMITTED) begin
          state_next[compute_start_index] = META_RUNNING;
        end else begin
          stale_event_count_next = stale_event_count + 1'b1;
        end
      end
      if (consumer_release2_valid) begin
        if (32'(consumer_release2_slot) >= SLOT_COUNT || epoch_mem[release2_index] != consumer_release2_epoch ||
            generation_mem[release2_index] != consumer_release2_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_effective[release2_index] == META_RUNNING || state_effective[release2_index] == META_COMMITTED ||
                     state_effective[release2_index] == META_EVICTABLE || state_effective[release2_index] == META_RESIDENT_SPEC) begin
          state_next[release2_index] = META_FREE;
          quarantined_next[release2_index] = 1'b0;
        end else begin
          stale_event_count_next = stale_event_count + 1'b1;
        end
      end
      if (compute_start2_valid) begin
        if (32'(compute_start2_slot) >= SLOT_COUNT || epoch_mem[compute_start2_index] != compute_start2_epoch ||
            generation_mem[compute_start2_index] != compute_start2_slot_generation) begin
          stale_event_count_next = stale_event_count + 1'b1;
        end else if (state_effective[compute_start2_index] == META_COMMITTED) begin
          state_next[compute_start2_index] = META_RUNNING;
        end else begin
          stale_event_count_next = stale_event_count + 1'b1;
        end
      end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) stale_event_count <= '0;
    else stale_event_count <= stale_event_count_next;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) discarded_dma_write_count <= '0;
    else discarded_dma_write_count <= discarded_dma_write_count_next;
  end

  for (genvar slot = 0; slot < SLOT_COUNT; slot = slot + 1) begin : slot_registers
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) state_mem[slot] <= META_FREE;
      else state_mem[slot] <= state_next[slot];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) epoch_mem[slot] <= '0;
      else epoch_mem[slot] <= epoch_next[slot];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) generation_mem[slot] <= '0;
      else generation_mem[slot] <= generation_next[slot];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) quarantined_mem[slot] <= '0;
      else quarantined_mem[slot] <= quarantined_next[slot];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) cancelled_mem[slot] <= '0;
      else cancelled_mem[slot] <= cancelled_next[slot];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) payload_complete_mem[slot] <= '0;
      else payload_complete_mem[slot] <= payload_complete_next[slot];
    end
  end
endmodule
