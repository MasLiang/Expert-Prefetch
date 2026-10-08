`timescale 1ns/1ps
/**
 * 以 burst 为仲裁边界的单 DMA channel fake-timing 引擎。
 *
 * 三个 FIFO 仅保存 {request_id,generation}；描述符由 DescriptorTable 通过
 * lookup ABI 返回。setup 与单一 burst 均不可抢占。speculative descriptor 在
 * 发现 demand/promoted 工作后，仅在已完成的 burst 边界重新入队，因而不会使
 * demand 饥饿。该模块没有假定 AXI 位宽或外部存储地址映射。
 */
module burst_dma_channel #(
  parameter integer FIFO_DEPTH = 8,
  parameter integer MAX_BURST_BYTES = 256,
  parameter integer ID_W = 16,
  parameter integer GENERATION_W = 8,
  parameter integer PROGRESS_ENTRIES = 256,
  parameter bit EXTERNAL_BURST_COMPLETION = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic bandwidth_enable,
  input logic [31:0] setup_cycles,
  input logic [31:0] bytes_per_cycle,
  input logic external_burst_done,
  input logic promotion_notify_valid,
  input tile_prefetch_package::descriptor_token_t promotion_notify_token,
  input logic demand_enqueue_valid,
  input tile_prefetch_package::descriptor_token_t demand_enqueue_token,
  output logic demand_enqueue_ready,
  input logic promoted_enqueue_valid,
  input tile_prefetch_package::descriptor_token_t promoted_enqueue_token,
  output logic promoted_enqueue_ready,
  input logic speculative_enqueue_valid,
  input tile_prefetch_package::descriptor_token_t speculative_enqueue_token,
  output logic speculative_enqueue_ready,
  output logic demand_pair_reserve_ready,
  output logic speculative_pair_reserve_ready,
  output logic descriptor_lookup_valid,
  output tile_prefetch_package::descriptor_token_t descriptor_lookup_token,
  input logic descriptor_entry_valid,
  input logic descriptor_entry_cancelled,
  input logic descriptor_entry_promoted,
  input logic descriptor_entry_inflight,
  input logic [7:0] descriptor_lookup_slot_generation,
  input tile_prefetch_package::tile_descriptor_t descriptor_lookup_descriptor,
  output logic dma_claim_valid,
  output tile_prefetch_package::descriptor_token_t dma_claim_token,
  output logic dma_burst_start,
  output logic dma_burst_done,
  output logic [31:0] dma_burst_bytes,
  output logic dma_done,
  output logic active,
  output logic drained,
  output tile_prefetch_package::descriptor_token_t active_token,
  output tile_prefetch_package::tile_descriptor_t active_descriptor,
  output logic [7:0] active_slot_generation,
  output logic [31:0] active_remaining_bytes,
  output logic [31:0] demand_stall_cycles,
  output logic [31:0] demand_fifo_peak,
  output logic [31:0] promoted_fifo_peak,
  output logic [31:0] speculative_fifo_peak,
  output logic [31:0] tombstone_skip_count
  ,output logic tombstone_retired_valid
  ,output logic tombstone_retired_cancelled
  ,output tile_prefetch_package::descriptor_token_t tombstone_retired_token
  ,output tile_prefetch_package::tile_descriptor_t tombstone_retired_descriptor
  ,output logic [7:0] tombstone_retired_slot_generation
);
  import tile_prefetch_package::*;
  localparam integer PTR_W = (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH);
  typedef enum logic [1:0] {IDLE, SETUP, BURST} dma_state_t;
  dma_state_t state;
  descriptor_token_t demand_fifo [0:FIFO_DEPTH-1];
  descriptor_token_t promoted_fifo [0:FIFO_DEPTH-1];
  descriptor_token_t speculative_fifo [0:FIFO_DEPTH-1];
  logic [PTR_W-1:0] demand_head, demand_tail, promoted_head, promoted_tail, spec_head, spec_tail;
  logic [PTR_W:0] demand_count, promoted_count, spec_count;
  assign drained = state == IDLE && demand_count == 0 && promoted_count == 0 && spec_count == 0 &&
    !demand_enqueue_valid && !promoted_enqueue_valid && !speculative_enqueue_valid;
  logic [31:0] setup_remaining, burst_remaining_cycles, current_burst_bytes;
  request_class_t active_queue_class;
  logic selected_demand, selected_promoted, selected_spec;
  descriptor_token_t selected_token;
  logic [31:0] calculated_burst_bytes, calculated_burst_cycles;
  logic [63:0] burst_address;
  logic [31:0] bytes_to_4k_boundary;
  logic priority_queued;
  wire active_promotion = promotion_notify_valid === 1'b1 && active &&
    promotion_notify_token == active_token;
  wire active_is_speculative = active_queue_class == REQUEST_SPECULATIVE && !active_promotion;
  logic [31:0] progress_bytes [0:PROGRESS_ENTRIES-1];
  logic [7:0] progress_generation [0:PROGRESS_ENTRIES-1];
  logic progress_valid [0:PROGRESS_ENTRIES-1];
  integer reset_progress;

  initial begin
    if ((MAX_BURST_BYTES & (MAX_BURST_BYTES - 1)) != 0)
      $error("MAX_BURST_BYTES must be a power of two");
    if (FIFO_DEPTH < 2)
      $error("FIFO_DEPTH must reserve one token for a preempted speculative request");
  end

  function automatic [PTR_W-1:0] next_ptr(input logic [PTR_W-1:0] pointer);
    if (32'(pointer) == FIFO_DEPTH - 1) next_ptr = '0;
    else next_ptr = pointer + 1'b1;
  endfunction

  always_comb begin
    priority_queued = demand_count != 0 || promoted_count != 0;
    selected_demand = demand_count != 0;
    selected_promoted = !selected_demand && promoted_count != 0;
    selected_spec = !selected_demand && !selected_promoted && spec_count != 0;
    selected_token = '0;
    if (selected_demand) selected_token = demand_fifo[demand_head];
    else if (selected_promoted) selected_token = promoted_fifo[promoted_head];
    else if (selected_spec) selected_token = speculative_fifo[spec_head];
    /* A FIFO push and a head pop of the same class require pointer/count
       bypass logic. The M10.1 control plane instead defers arbitration by one
       cycle whenever any enqueue handshakes, preserving deterministic priority
       and avoiding a multi-writer count update. */
    descriptor_lookup_valid = state == IDLE && bandwidth_enable && !dma_done &&
      !demand_enqueue_valid && !promoted_enqueue_valid && !speculative_enqueue_valid &&
      (selected_demand || selected_promoted || selected_spec);
    descriptor_lookup_token = selected_token;
    demand_enqueue_ready = 32'(demand_count) < FIFO_DEPTH;
    promoted_enqueue_ready = 32'(promoted_count) < FIFO_DEPTH;
    /* Reserve one speculative slot when a speculative descriptor is active so
       it can be returned at the next priority boundary. */
    speculative_enqueue_ready = 32'(spec_count) < (FIFO_DEPTH - ((active && active_is_speculative) ? 1 : 0)) &&
      !(state == BURST && priority_queued && active_is_speculative);
    demand_pair_reserve_ready = 32'(demand_count) <= FIFO_DEPTH - 2;
    speculative_pair_reserve_ready = 32'(spec_count) <= FIFO_DEPTH - 2 -
      ((active && active_is_speculative) ? 1 : 0);
    calculated_burst_bytes = active_remaining_bytes > MAX_BURST_BYTES ? MAX_BURST_BYTES : active_remaining_bytes;
    burst_address = active_descriptor.external_addr +
      64'(active_descriptor.byte_count) - 64'(active_remaining_bytes);
    bytes_to_4k_boundary = 32'd4096 - {20'b0, burst_address[11:0]};
    if (EXTERNAL_BURST_COMPLETION && calculated_burst_bytes > bytes_to_4k_boundary)
      calculated_burst_bytes = bytes_to_4k_boundary;
    calculated_burst_cycles = EXTERNAL_BURST_COMPLETION ? 0 :
      (calculated_burst_bytes + bytes_per_cycle - 1) / (bytes_per_cycle == 0 ? 1 : bytes_per_cycle);
  end

  // FIFO payload has no reset; empty counts invalidate all previous tokens.
  // Keep each memory on a pure clocked write process to avoid inferred
  // asynchronous set/reset controls on unreset storage.
  always_ff @(posedge clk) begin
    if (rst_n && demand_enqueue_valid && demand_enqueue_ready)
      demand_fifo[demand_tail] <= demand_enqueue_token;
  end

  always_ff @(posedge clk) begin
    if (rst_n && promoted_enqueue_valid && promoted_enqueue_ready)
      promoted_fifo[promoted_tail] <= promoted_enqueue_token;
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (speculative_enqueue_valid && speculative_enqueue_ready)
        speculative_fifo[spec_tail] <= speculative_enqueue_token;
      // The interrupted active token has the same priority as in the original
      // sequential process: its return wins over a concurrent enqueue.
      if (state == BURST &&
          ((EXTERNAL_BURST_COMPLETION && external_burst_done) ||
           (!EXTERNAL_BURST_COMPLETION && burst_remaining_cycles <= 1)) &&
          active_remaining_bytes > current_burst_bytes &&
          active_is_speculative && priority_queued)
        speculative_fifo[spec_tail] <= active_token;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      demand_head <= '0;
      demand_tail <= '0;
      demand_count <= '0;
      promoted_head <= '0;
      promoted_tail <= '0;
      promoted_count <= '0;
      spec_head <= '0;
      spec_tail <= '0;
      spec_count <= '0;
      setup_remaining <= '0;
      burst_remaining_cycles <= '0;
      current_burst_bytes <= '0;
      active <= 1'b0;
      active_token <= '0;
      active_descriptor <= '0;
      active_slot_generation <= '0;
      active_remaining_bytes <= '0;
      dma_claim_valid <= 1'b0;
      dma_claim_token <= '0;
      dma_burst_start <= 1'b0;
      dma_burst_done <= 1'b0;
      dma_burst_bytes <= '0;
      dma_done <= 1'b0;
      demand_stall_cycles <= '0;
      demand_fifo_peak <= '0;
      promoted_fifo_peak <= '0;
      speculative_fifo_peak <= '0;
      tombstone_skip_count <= '0;
      tombstone_retired_valid <= 1'b0;
      tombstone_retired_cancelled <= 1'b0;
      tombstone_retired_token <= '0;
      tombstone_retired_descriptor <= '0;
      tombstone_retired_slot_generation <= '0;
      for (reset_progress = 0; reset_progress < PROGRESS_ENTRIES; reset_progress = reset_progress + 1) begin
        progress_valid[reset_progress] <= 0;
        progress_bytes[reset_progress] <= 0;
        progress_generation[reset_progress] <= 0;
      end
    end else begin
      dma_claim_valid <= 1'b0;
      dma_burst_start <= 1'b0;
      dma_burst_done <= 1'b0;
      dma_done <= 1'b0;
      tombstone_retired_valid <= 1'b0;
      if (demand_enqueue_valid && demand_enqueue_ready) begin
        demand_tail <= next_ptr(demand_tail);
        demand_count <= demand_count + 1'b1;
        if (32'(demand_count) + 1'b1 > demand_fifo_peak) demand_fifo_peak <= 32'(demand_count) + 1'b1;
      end
      if (promoted_enqueue_valid && promoted_enqueue_ready) begin
        promoted_tail <= next_ptr(promoted_tail);
        promoted_count <= promoted_count + 1'b1;
        if (32'(promoted_count) + 1'b1 > promoted_fifo_peak) promoted_fifo_peak <= 32'(promoted_count) + 1'b1;
      end
      if (speculative_enqueue_valid && speculative_enqueue_ready) begin
        spec_tail <= next_ptr(spec_tail);
        spec_count <= spec_count + 1'b1;
        if (32'(spec_count) + 1'b1 > speculative_fifo_peak) speculative_fifo_peak <= 32'(spec_count) + 1'b1;
      end
      if (bandwidth_enable && demand_count != 0 && !(state == IDLE && selected_demand && descriptor_entry_valid && !descriptor_entry_cancelled))
        demand_stall_cycles <= demand_stall_cycles + 1'b1;

      case (state)
        IDLE: begin
          active <= 1'b0;
          if (descriptor_lookup_valid) begin
            if (!descriptor_entry_valid || descriptor_entry_cancelled ||
                (selected_spec && descriptor_entry_promoted && descriptor_entry_inflight !== 1'b1)) begin
              tombstone_skip_count <= tombstone_skip_count + 1'b1;
              tombstone_retired_valid <= 1'b1;
              tombstone_retired_cancelled <= descriptor_entry_cancelled;
              tombstone_retired_token <= selected_token;
              tombstone_retired_descriptor <= descriptor_lookup_descriptor;
              tombstone_retired_slot_generation <= descriptor_lookup_slot_generation;
              if (selected_demand) begin
                demand_head <= next_ptr(demand_head);
                demand_count <= demand_count - 1'b1;
              end else if (selected_promoted) begin
                promoted_head <= next_ptr(promoted_head);
                promoted_count <= promoted_count - 1'b1;
              end else begin
                spec_head <= next_ptr(spec_head);
                spec_count <= spec_count - 1'b1;
              end
            end else begin
              active <= 1'b1;
              active_token <= selected_token;
              active_descriptor <= descriptor_lookup_descriptor;
              active_slot_generation <= descriptor_lookup_slot_generation;
              if (32'(selected_token.request_id) < PROGRESS_ENTRIES &&
                  progress_valid[int'(selected_token.request_id)] &&
                  progress_generation[int'(selected_token.request_id)] == selected_token.generation)
                active_remaining_bytes <= progress_bytes[int'(selected_token.request_id)];
              else active_remaining_bytes <= descriptor_lookup_descriptor.byte_count;
              if (selected_demand) begin
                demand_head <= next_ptr(demand_head);
                demand_count <= demand_count - 1'b1;
              end else if (selected_promoted) begin
                promoted_head <= next_ptr(promoted_head);
                promoted_count <= promoted_count - 1'b1;
              end else begin
                spec_head <= next_ptr(spec_head);
                spec_count <= spec_count - 1'b1;
              end
              dma_claim_valid <= 1'b1;
              dma_claim_token <= selected_token;
              setup_remaining <= setup_cycles;
              state <= SETUP;
            end
          end
        end
        SETUP: begin
          if (bandwidth_enable && setup_remaining == 0) begin
            current_burst_bytes <= calculated_burst_bytes;
            burst_remaining_cycles <= calculated_burst_cycles;
            dma_burst_start <= 1'b1;
            dma_burst_bytes <= calculated_burst_bytes;
            state <= BURST;
          end else if (bandwidth_enable) begin
            setup_remaining <= setup_remaining - 1'b1;
          end
        end
        BURST: begin
          if ((EXTERNAL_BURST_COMPLETION && external_burst_done) ||
              (!EXTERNAL_BURST_COMPLETION && burst_remaining_cycles <= 1)) begin
            dma_burst_done <= 1'b1;
            dma_burst_bytes <= current_burst_bytes;
            if (active_remaining_bytes <= current_burst_bytes) begin
              active_remaining_bytes <= '0;
              if (32'(active_token.request_id) < PROGRESS_ENTRIES)
                progress_valid[int'(active_token.request_id)] <= 0;
              dma_done <= 1'b1;
              active <= 1'b0;
              state <= IDLE;
            end else begin
              active_remaining_bytes <= active_remaining_bytes - current_burst_bytes;
              if (active_is_speculative && priority_queued) begin
                spec_tail <= next_ptr(spec_tail);
                spec_count <= spec_count + 1'b1;
                if (32'(active_token.request_id) < PROGRESS_ENTRIES) begin
                  progress_valid[int'(active_token.request_id)] <= 1;
                  progress_bytes[int'(active_token.request_id)] <= active_remaining_bytes - current_burst_bytes;
                  progress_generation[int'(active_token.request_id)] <= active_token.generation;
                end
                state <= IDLE;
                active <= 1'b0;
              end else begin
                /* Recalculate against the updated remaining byte count and
                   AXI 4KB boundary on the following cycle. */
                setup_remaining <= 0;
                state <= SETUP;
              end
            end
          end else begin
            burst_remaining_cycles <= burst_remaining_cycles - 1'b1;
          end
        end
        default: state <= IDLE;
      endcase
    end
  end
  // Priority changes never alter original descriptor/request-class byte accounting.
  // Include the notification combinationally to cover promotion on burst-done edge.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) active_queue_class <= REQUEST_SPECULATIVE;
    else if (active_promotion) active_queue_class <= REQUEST_DEMAND;
    else if (state == IDLE && descriptor_lookup_valid && descriptor_entry_valid && !descriptor_entry_cancelled) begin
      if (selected_demand || selected_promoted || descriptor_entry_promoted === 1'b1)
        active_queue_class <= REQUEST_DEMAND;
      else active_queue_class <= REQUEST_SPECULATIVE;
    end
  end
endmodule
