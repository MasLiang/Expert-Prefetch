`timescale 1ns/1ps
/**
 * Bounded descriptor metadata table for ID-token FIFOs.
 *
 * An entry is not reusable until its active DMA work has retired and the owner
 * explicitly reports every FIFO/sideband token for its generation retired.
 */
module descriptor_table #(
  parameter integer ENTRY_COUNT = 32,
  parameter integer ID_W = (ENTRY_COUNT <= 1) ? 1 : $clog2(ENTRY_COUNT),
  parameter integer GENERATION_W = 8,
  parameter bit HOLD_FOR_ROUTE = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic allocate_valid,
  input tile_prefetch_package::tile_descriptor_t allocate_descriptor,
  input logic [GENERATION_W-1:0] allocate_slot_generation,
  output logic allocate_ready,
  output logic [ID_W:0] free_entry_count,
  output logic [ID_W-1:0] allocate_request_id,
  output logic [GENERATION_W-1:0] allocate_generation,
  input logic mark_promoted_valid,
  input logic promotion_token_created,
  input logic [ID_W-1:0] mark_promoted_id,
  input logic [GENERATION_W-1:0] mark_promoted_generation,
  input logic mark_cancelled_valid,
  input logic [ID_W-1:0] mark_cancelled_id,
  input logic [GENERATION_W-1:0] mark_cancelled_generation,
  input logic mark_dma_start_valid,
  input logic [ID_W-1:0] mark_dma_start_id,
  input logic [GENERATION_W-1:0] mark_dma_start_generation,
  input logic mark_dma_done_valid,
  input logic [ID_W-1:0] mark_dma_done_id,
  input logic [GENERATION_W-1:0] mark_dma_done_generation,
  input logic retire_refs_valid,
  input logic [ID_W-1:0] retire_refs_id,
  input logic [GENERATION_W-1:0] retire_refs_generation,
  output logic stale_event,
  input logic [ID_W-1:0] query_id,
  output logic query_valid,
  output logic [GENERATION_W-1:0] query_generation,
  output logic query_cancelled,
  output logic query_promoted,
  output logic query_inflight,
  output logic query_drop_on_complete,
  output logic [GENERATION_W-1:0] query_slot_generation,
  output tile_prefetch_package::tile_descriptor_t query_descriptor,
  input logic [ID_W-1:0] route_query_id,
  output logic route_query_valid,
  output logic [GENERATION_W-1:0] route_query_generation,
  output logic route_query_cancelled,
  output logic route_query_inflight,
  output logic route_query_completed,
  output logic [GENERATION_W-1:0] route_query_slot_generation,
  output tile_prefetch_package::tile_descriptor_t route_query_descriptor
);
  import tile_prefetch_package::*;
  logic valid_table [0:ENTRY_COUNT-1];
  logic [GENERATION_W-1:0] generation_table [0:ENTRY_COUNT-1];
  logic cancelled_table [0:ENTRY_COUNT-1];
  logic promoted_table [0:ENTRY_COUNT-1];
  logic inflight_table [0:ENTRY_COUNT-1];
  logic drop_table [0:ENTRY_COUNT-1];
  logic refs_retired_table [0:ENTRY_COUNT-1];
  logic completed_table [0:ENTRY_COUNT-1];
  logic resolved_table [0:ENTRY_COUNT-1];
  logic [1:0] reference_count [0:ENTRY_COUNT-1];
  logic [GENERATION_W-1:0] slot_generation_table [0:ENTRY_COUNT-1];
  tile_descriptor_t descriptor_table_mem [0:ENTRY_COUNT-1];
  integer scan;
  integer reset_index;
  logic free_found;
  logic [ID_W-1:0] free_id;
  tile_descriptor_t flush_descriptor;

  always_comb begin
    free_found = 1'b0;
    free_id = '0;
    free_entry_count = '0;
    for (scan = 0; scan < ENTRY_COUNT;
      scan = scan + 1) begin
      if (!valid_table[scan]) free_entry_count = free_entry_count + 1'b1;
      if (!free_found && !valid_table[scan]) begin
        free_found = 1'b1;
        free_id = scan[ID_W-1:0];
      end
    end
    allocate_ready = free_found;
    allocate_request_id = free_id;
    allocate_generation = generation_table[free_id] + 1'b1;
    query_valid = valid_table[query_id];
    query_generation = generation_table[query_id];
    query_cancelled = cancelled_table[query_id];
    query_promoted = promoted_table[query_id];
    query_inflight = inflight_table[query_id];
    query_drop_on_complete = drop_table[query_id];
    query_slot_generation = slot_generation_table[query_id];
    query_descriptor = descriptor_table_mem[query_id];
    route_query_valid = valid_table[route_query_id];
    route_query_generation = generation_table[route_query_id];
    route_query_cancelled = cancelled_table[route_query_id];
    route_query_inflight = inflight_table[route_query_id];
    route_query_completed = completed_table[route_query_id];
    route_query_slot_generation = slot_generation_table[route_query_id];
    route_query_descriptor = descriptor_table_mem[route_query_id];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stale_event <= 0;
      for (reset_index = 0; reset_index < ENTRY_COUNT;
      reset_index = reset_index + 1) begin
        valid_table[reset_index] <= 0;
        generation_table[reset_index] <= 0;
        cancelled_table[reset_index] <= 0;
        promoted_table[reset_index] <= 0;
        inflight_table[reset_index] <= 0;
        drop_table[reset_index] <= 0;
        refs_retired_table[reset_index] <= 0;
        completed_table[reset_index] <= 0;
        resolved_table[reset_index] <= 0;
        reference_count[reset_index] <= 0;
        slot_generation_table[reset_index] <= '0;
        descriptor_table_mem[reset_index] <= '0;
      end
    end else begin
      stale_event <= 0;
      if (allocate_valid && allocate_ready) begin
        valid_table[free_id] <= 1;
        generation_table[free_id] <= generation_table[free_id] + 1'b1;
        descriptor_table_mem[free_id] <= allocate_descriptor;
        slot_generation_table[free_id] <= allocate_slot_generation;
        cancelled_table[free_id] <= 0;
        promoted_table[free_id] <= 0;
        inflight_table[free_id] <= 0;
        drop_table[free_id] <= 0;
        refs_retired_table[free_id] <= 0;
        completed_table[free_id] <= 0;
        resolved_table[free_id] <= !HOLD_FOR_ROUTE || allocate_descriptor.request_class == REQUEST_DEMAND;
        reference_count[free_id] <= 1;
      end
      if (mark_promoted_valid) begin
        if (!valid_table[mark_promoted_id] || generation_table[mark_promoted_id] != mark_promoted_generation)
          stale_event <= 1;
        else begin
          promoted_table[mark_promoted_id] <= 1;
          resolved_table[mark_promoted_id] <= 1;
        end
      end
      if (mark_cancelled_valid) begin
        if (!valid_table[mark_cancelled_id] || generation_table[mark_cancelled_id] != mark_cancelled_generation)
          stale_event <= 1;
        else if (inflight_table[mark_cancelled_id])
          drop_table[mark_cancelled_id] <= 1;
        else
          cancelled_table[mark_cancelled_id] <= 1;
        if (valid_table[mark_cancelled_id] && generation_table[mark_cancelled_id] == mark_cancelled_generation)
          resolved_table[mark_cancelled_id] <= 1;
      end
      if (mark_dma_start_valid) begin
        if (!valid_table[mark_dma_start_id] || generation_table[mark_dma_start_id] != mark_dma_start_generation)
          stale_event <= 1;
        else
          inflight_table[mark_dma_start_id] <= 1;
      end
      if (mark_dma_done_valid) begin
        if (!valid_table[mark_dma_done_id] || generation_table[mark_dma_done_id] != mark_dma_done_generation)
          stale_event <= 1;
        else begin
          inflight_table[mark_dma_done_id] <= 0;
          completed_table[mark_dma_done_id] <= 1;
          if (refs_retired_table[mark_dma_done_id] ||
              (retire_refs_valid && 32'(retire_refs_id) == 32'(mark_dma_done_id) &&
               retire_refs_generation == mark_dma_done_generation)) begin
            if (!HOLD_FOR_ROUTE && resolved_table[mark_dma_done_id])
            valid_table[mark_dma_done_id] <= 0;
          end
        end
      end
      if (retire_refs_valid) begin
        if (!valid_table[retire_refs_id] || generation_table[retire_refs_id] != retire_refs_generation)
          stale_event <= 1;
        else begin
          refs_retired_table[retire_refs_id] <= 1;
          if (!HOLD_FOR_ROUTE && !inflight_table[retire_refs_id] && resolved_table[retire_refs_id]) valid_table[retire_refs_id] <= 0;
        end
      end
      /* DMA completion may precede token retirement; final release happens only
         once the owner has retired every reference for this generation. */
      if (mark_dma_done_valid && valid_table[mark_dma_done_id] &&
          generation_table[mark_dma_done_id] == mark_dma_done_generation &&
          !HOLD_FOR_ROUTE && refs_retired_table[mark_dma_done_id] && resolved_table[mark_dma_done_id]) begin
        valid_table[mark_dma_done_id] <= 0;
      end
      for (reset_index = 0; reset_index < ENTRY_COUNT; reset_index = reset_index + 1) begin
        flush_descriptor = descriptor_table_mem[reset_index];
        if (soft_flush_valid === 1'b1 && valid_table[reset_index] && flush_descriptor.epoch_id == soft_flush_epoch) begin
          cancelled_table[reset_index] <= !(inflight_table[reset_index] ||
            (mark_dma_start_valid && 32'(mark_dma_start_id) == reset_index &&
             mark_dma_start_generation == generation_table[reset_index]));
          resolved_table[reset_index] <= 1;
          if (inflight_table[reset_index] || (mark_dma_start_valid && 32'(mark_dma_start_id) == reset_index &&
              mark_dma_start_generation == generation_table[reset_index])) drop_table[reset_index] <= 1;
        end
        if (HOLD_FOR_ROUTE && valid_table[reset_index]) begin
          case ({mark_promoted_valid && 32'(mark_promoted_id) == reset_index &&
                 mark_promoted_generation == generation_table[reset_index] &&
                 (promotion_token_created === 1'b1 ||
                  (promotion_token_created !== 1'b0 && !inflight_table[reset_index] && !completed_table[reset_index])) &&
                 !promoted_table[reset_index],
                 retire_refs_valid && 32'(retire_refs_id) == reset_index &&
                 retire_refs_generation == generation_table[reset_index]})
            2'b10: reference_count[reset_index] <= reference_count[reset_index] + 1'b1;
            2'b01: if (reference_count[reset_index] != 0)
              reference_count[reset_index] <= reference_count[reset_index] - 1'b1;
            default: begin end
          endcase
        end
        if (valid_table[reset_index] &&
            (HOLD_FOR_ROUTE ? reference_count[reset_index] == 0 : refs_retired_table[reset_index]) &&
            !inflight_table[reset_index] && resolved_table[reset_index]) begin
          valid_table[reset_index] <= 0;
        end
      end
    end
  end
endmodule
