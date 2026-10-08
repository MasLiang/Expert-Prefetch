`timescale 1ns/1ps
/** 已 route-resolved 的真实 Expert 缺失 tile demand producer；已 admission 的
 * speculative prefix 不重复分配。每次 logical handshake 后才推进 tile 游标。 */
module true_tile_demander #(
  parameter integer MAX_CANDIDATES = 16,
  parameter integer TOP_K_MAX = 8,
  parameter integer TILES_PER_EXPERT = 16,
  parameter integer GU_BYTE_RATIO = 2
) (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic speculative_accept,
  input tile_prefetch_package::tile_descriptor_t speculative_descriptor,
  input logic route_accept,
  input logic [15:0] route_epoch,
  input logic [2:0] route_layer,
  input logic [9:0] route_expert,
  input logic start,
  input logic [15:0] sequence_id,
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic [31:0] physical_tile_bytes,
  output logic logical_valid,
  input logic logical_ready,
  output tile_prefetch_package::tile_descriptor_t gu_descriptor,
  output tile_prefetch_package::tile_descriptor_t down_descriptor,
  output logic active,
  output logic done
);
  import tile_prefetch_package::*;
  logic [9:0] candidate_ids [0:MAX_CANDIDATES-1];
  logic [8:0] prefix_depth [0:MAX_CANDIDATES-1];
  logic [9:0] true_ids [0:TOP_K_MAX-1];
  integer truth_count;
  integer expert_index;
  integer tile_index;
  logic [15:0] epoch_hold;
  logic [2:0] layer_hold;
  logic already_allocated;
  integer scan;

  always_comb begin
    already_allocated = 0;
    for (scan = 0; scan < MAX_CANDIDATES; scan = scan + 1) begin
      if (candidate_ids[scan] == true_ids[expert_index] && tile_index < prefix_depth[scan])
        already_allocated = 1;
    end
    logical_valid = active && !already_allocated;
    gu_descriptor = '0;
    gu_descriptor.epoch_id = epoch_hold;
    gu_descriptor.layer_id = {13'b0, layer_hold};
    gu_descriptor.sequence_id = sequence_id;
    gu_descriptor.expert_id = true_ids[expert_index];
    gu_descriptor.tile_id = 8'(tile_index);
    gu_descriptor.matrix_kind = MATRIX_GU;
    gu_descriptor.byte_count = physical_tile_bytes * GU_BYTE_RATIO;
    gu_descriptor.external_addr = gu_base_addr + true_ids[expert_index] * expert_stride_bytes +
      tile_index * physical_tile_bytes * GU_BYTE_RATIO;
    gu_descriptor.request_class = REQUEST_DEMAND;
    down_descriptor = gu_descriptor;
    down_descriptor.matrix_kind = MATRIX_DOWN;
    down_descriptor.byte_count = physical_tile_bytes;
    down_descriptor.external_addr = down_base_addr + true_ids[expert_index] * expert_stride_bytes +
      tile_index * physical_tile_bytes;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active <= 0;
      done <= 0;
      truth_count <= 0;
      expert_index <= 0;
      tile_index <= 0;
      epoch_hold <= 0;
      layer_hold <= 0;
      for (integer i = 0; i < MAX_CANDIDATES; i = i + 1) begin
        candidate_ids[i] <= 0;
        prefix_depth[i] <= 0;
      end
      for (integer i = 0; i < TOP_K_MAX; i = i + 1) true_ids[i] <= 0;
    end else begin
      done <= 0;
      if (clear) begin
        truth_count <= 0;
        active <= 0;
        for (integer i = 0; i < MAX_CANDIDATES; i = i + 1) prefix_depth[i] <= 0;
      end else begin
        // Abort wins over a registered Router accept, resolution start, and
        // demand cursor advancement on the same edge.
        if (speculative_accept && 32'(speculative_descriptor.candidate_rank) < MAX_CANDIDATES) begin
          candidate_ids[int'(speculative_descriptor.candidate_rank)] <= speculative_descriptor.expert_id;
          prefix_depth[int'(speculative_descriptor.candidate_rank)] <= {1'b0, speculative_descriptor.tile_id} + 1'b1;
        end
        if (route_accept && truth_count < TOP_K_MAX) begin
          true_ids[truth_count] <= route_expert;
          truth_count <= truth_count + 1;
          epoch_hold <= route_epoch;
          layer_hold <= route_layer;
        end
        if (start && truth_count != 0) begin
          active <= 1;
          expert_index <= 0;
          tile_index <= 0;
        end
        if (active && (already_allocated || logical_ready)) begin
          if (tile_index == TILES_PER_EXPERT-1) begin
            tile_index <= 0;
            if (expert_index == truth_count-1) begin
              active <= 0;
              done <= 1;
            end else expert_index <= expert_index + 1;
          end else tile_index <= tile_index + 1;
        end
      end
    end
  end
endmodule
