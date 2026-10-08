`timescale 1ns/1ps
/**
 * 将已校验的 predictor candidate 表变成 prefix-closed speculative F-tile 请求。
 *
 * 该模块保存一个 prediction transaction 的 immutable candidate metadata，并在
 * 每次下游 logical descriptor handshake 后才更新 HardwareGreedy-v1 的 depth、
 * global credit 和 per-channel credit。地址几何由 host 以统一 stride 提供；它
 * 是 address-table 的参数化紧凑表示，不推断任何板级 DDR 地址。
 */
module candidate_prefetch_scheduler #(
  parameter integer MAX_CANDIDATES = 16,
  parameter integer TILES_PER_EXPERT = 16,
  parameter integer CHANNEL_COUNT = 4,
  parameter integer EXPERT_W = 10,
  parameter integer GENERATION_W = 8,
  parameter integer GU_BYTE_RATIO = 1,
  parameter integer CHANNEL_W = (CHANNEL_COUNT <= 1) ? 1 : $clog2(CHANNEL_COUNT)
) (
  input logic clk,
  input logic rst_n,
  input logic transaction_begin,
  input logic stop_allocation,
  input logic [15:0] transaction_epoch,
  input logic [15:0] transaction_sequence,
  input logic [2:0] transaction_layer,
  input logic [31:0] speculative_budget_bytes,
  input logic [CHANNEL_COUNT*32-1:0] channel_credit_bytes,
  input logic candidate_accept,
  input logic [15:0] candidate_epoch,
  input logic [2:0] candidate_layer,
  input logic [EXPERT_W-1:0] candidate_expert,
  input logic [4:0] candidate_rank,
  input logic [11:0] candidate_confidence,
  input logic candidate_last,
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic [31:0] physical_tile_bytes,
  output logic logical_valid,
  input logic logical_ready,
  output tile_prefetch_package::tile_descriptor_t logical_gu_descriptor,
  output tile_prefetch_package::tile_descriptor_t logical_down_descriptor,
  output logic allocation_active,
  output logic allocation_done,
  output logic [31:0] remaining_credit_bytes
);
  import tile_prefetch_package::*;
  localparam integer DEPTH_W = 8;
  localparam integer CHLOAD_W = $clog2(MAX_CANDIDATES * TILES_PER_EXPERT + 1);
  localparam integer INDEX_W = (MAX_CANDIDATES <= 1) ? 1 : $clog2(MAX_CANDIDATES);

  logic [MAX_CANDIDATES-1:0] candidate_valid_mem;
  logic [EXPERT_W-1:0] expert_mem [0:MAX_CANDIDATES-1];
  logic [4:0] rank_mem [0:MAX_CANDIDATES-1];
  logic [11:0] confidence_mem [0:MAX_CANDIDATES-1];
  logic [DEPTH_W-1:0] depth_mem [0:MAX_CANDIDATES-1];
  logic [31:0] channel_remaining [0:CHANNEL_COUNT-1];
  logic [CHLOAD_W-1:0] channel_load [0:CHANNEL_COUNT-1];
  logic [MAX_CANDIDATES*EXPERT_W-1:0] packed_expert;
  logic [MAX_CANDIDATES*5-1:0] packed_rank;
  logic [MAX_CANDIDATES*12-1:0] packed_confidence;
  logic [MAX_CANDIDATES*DEPTH_W-1:0] packed_depth;
  logic [MAX_CANDIDATES*32-1:0] packed_logical_bytes;
  logic [CHANNEL_COUNT*32-1:0] packed_channel_remaining;
  logic [CHANNEL_COUNT*CHLOAD_W-1:0] packed_channel_load;
  logic winner_valid;
  logic [INDEX_W-1:0] winner_index;
  logic [CHANNEL_W-1:0] winner_channel;
  logic selected_valid;
  logic [INDEX_W-1:0] selected_index;
  logic [CHANNEL_W-1:0] selected_channel;
  logic [31:0] selected_global_credit_after;
  logic [31:0] selected_channel_credit_after;
  logic selection_inflight;
  logic selection_launch;
  logic selection_done;
  assign selection_launch = allocation_active && !selected_valid && !selection_inflight &&
    !transaction_begin && stop_allocation !== 1'b1;
  logic [15:0] epoch_hold;
  logic [15:0] sequence_hold;
  logic [2:0] layer_hold;

  initial begin
    if ((CHANNEL_COUNT & (CHANNEL_COUNT - 1)) != 0) begin
      $error("CHANNEL_COUNT must be a power of two");
    end
    if (TILES_PER_EXPERT > 256) begin
      $error("tile_id ABI supports at most 256 F tiles");
    end
  end

  always_comb begin
    for (integer index = 0; index < MAX_CANDIDATES; index = index + 1) begin
      packed_expert[index*EXPERT_W +: EXPERT_W] = expert_mem[index];
      packed_rank[index*5 +: 5] = rank_mem[index];
      packed_confidence[index*12 +: 12] = confidence_mem[index];
      packed_depth[index*DEPTH_W +: DEPTH_W] = depth_mem[index];
      packed_logical_bytes[index*32 +: 32] = physical_tile_bytes * (GU_BYTE_RATIO + 1);
    end
    for (integer index = 0; index < CHANNEL_COUNT; index = index + 1) begin
      packed_channel_remaining[index*32 +: 32] = channel_remaining[index];
      packed_channel_load[index*CHLOAD_W +: CHLOAD_W] = channel_load[index];
    end
  end

  hardware_greedy_v1 #(
    .CANDIDATE_COUNT(MAX_CANDIDATES),
    .EXPERT_W(EXPERT_W),
    .DEPTH_W(DEPTH_W),
    .CHANNEL_COUNT(CHANNEL_COUNT),
    .MAX_ALLOCATIONS(MAX_CANDIDATES * TILES_PER_EXPERT),
    .UNIFORM_TILE_BYTES(1'b1),
    .REGISTERED_PIPELINE(1'b1)
  ) greedy (
    .clk(clk),
    .rst_n(rst_n),
    .select_start(selection_launch),
    .cancel_selection(transaction_begin || stop_allocation === 1'b1),
    .selection_done(selection_done),
    .candidate_valid(candidate_valid_mem),
    .candidate_expert_id(packed_expert),
    .candidate_rank(packed_rank),
    .candidate_confidence_q12(packed_confidence),
    .candidate_next_depth(packed_depth),
    .candidate_logical_bytes(packed_logical_bytes),
    .global_credit_bytes(remaining_credit_bytes),
    .channel_credit_bytes(packed_channel_remaining),
    .channel_load_allocations(packed_channel_load),
    .winner_valid(winner_valid),
    .winner_index(winner_index),
    .winner_channel(winner_channel)
  );

  always_comb begin
    logical_valid = allocation_active && selected_valid && stop_allocation !== 1'b1;
    logical_gu_descriptor = '0;
    logical_down_descriptor = '0;
    if (logical_valid) begin
      logical_gu_descriptor.epoch_id = epoch_hold;
      logical_gu_descriptor.layer_id = {{13{1'b0}}, layer_hold};
      logical_gu_descriptor.sequence_id = sequence_hold;
      logical_gu_descriptor.expert_id = expert_mem[selected_index];
      logical_gu_descriptor.tile_id = depth_mem[selected_index];
      logical_gu_descriptor.matrix_kind = MATRIX_GU;
      logical_gu_descriptor.external_addr = gu_base_addr +
        expert_mem[selected_index] * expert_stride_bytes +
        depth_mem[selected_index] * physical_tile_bytes * GU_BYTE_RATIO;
      logical_gu_descriptor.byte_count = physical_tile_bytes * GU_BYTE_RATIO;
      logical_gu_descriptor.memory_channel = 4'(selected_channel);
      logical_gu_descriptor.request_class = REQUEST_SPECULATIVE;
      logical_gu_descriptor.candidate_rank = rank_mem[selected_index];
      logical_gu_descriptor.confidence_q12 = confidence_mem[selected_index];
      logical_down_descriptor = logical_gu_descriptor;
      logical_down_descriptor.matrix_kind = MATRIX_DOWN;
      logical_down_descriptor.byte_count = physical_tile_bytes;
      logical_down_descriptor.external_addr = down_base_addr +
        expert_mem[selected_index] * expert_stride_bytes +
        depth_mem[selected_index] * physical_tile_bytes;
    end
  end

  // 选择与地址生成之间的寄存级。暂停期间 winner 和输出 descriptor 保持稳定；
  // 下一次选择必须等待本次 handshake 更新 depth/credit，不能复用旧 credit。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected_valid <= 1'b0;
    else if (transaction_begin || stop_allocation === 1'b1)
      selected_valid <= 1'b0;
    else if (logical_valid && logical_ready) selected_valid <= 1'b0;
    else if (allocation_active && selection_done && winner_valid)
      selected_valid <= 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected_index <= '0;
    else if (allocation_active && selection_done && winner_valid)
      selected_index <= winner_index;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected_channel <= '0;
    else if (allocation_active && selection_done && winner_valid)
      selected_channel <= winner_channel;
  end

  // 与winner同时快照扣费后的credit，避免descriptor admission的长ready路径
  // 再穿过32-bit减法器。selection期间无其他issue，credit在此期间保持稳定。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected_global_credit_after <= 0;
    else if (transaction_begin) selected_global_credit_after <= 0;
    else if (allocation_active && selection_done && winner_valid)
      selected_global_credit_after <= remaining_credit_bytes -
        32'(physical_tile_bytes * (GU_BYTE_RATIO + 1));
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected_channel_credit_after <= 0;
    else if (transaction_begin) selected_channel_credit_after <= 0;
    else if (allocation_active && selection_done && winner_valid)
      selected_channel_credit_after <= channel_remaining[winner_channel] -
        32'(physical_tile_bytes * (GU_BYTE_RATIO + 1));
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selection_inflight <= 0;
    else if (transaction_begin || stop_allocation === 1'b1 || selection_done)
      selection_inflight <= 0;
    else if (selection_launch) selection_inflight <= 1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      candidate_valid_mem <= '0;
      allocation_active <= 1'b0;
      allocation_done <= 1'b0;
      remaining_credit_bytes <= '0;
      epoch_hold <= '0;
      sequence_hold <= '0;
      layer_hold <= '0;
      for (integer index = 0; index < MAX_CANDIDATES; index = index + 1) begin
        expert_mem[index] <= '0;
        rank_mem[index] <= '0;
        confidence_mem[index] <= '0;
        depth_mem[index] <= '0;
      end
      for (integer index = 0; index < CHANNEL_COUNT; index = index + 1) begin
        channel_remaining[index] <= '0;
        channel_load[index] <= '0;
      end
    end else begin
      allocation_done <= 1'b0;
      if (transaction_begin) begin
        candidate_valid_mem <= '0;
        allocation_active <= 1'b0;
        remaining_credit_bytes <= speculative_budget_bytes;
        epoch_hold <= transaction_epoch;
        sequence_hold <= transaction_sequence;
        layer_hold <= transaction_layer;
        for (integer index = 0; index < CHANNEL_COUNT; index = index + 1) begin
          channel_remaining[index] <= channel_credit_bytes[index*32 +: 32];
          channel_load[index] <= '0;
        end
      end
      if (candidate_accept) begin
        if (32'(candidate_rank) < MAX_CANDIDATES) begin
          candidate_valid_mem[int'(candidate_rank)] <= candidate_epoch == epoch_hold &&
            candidate_layer == layer_hold;
          expert_mem[int'(candidate_rank)] <= candidate_expert;
          rank_mem[int'(candidate_rank)] <= candidate_rank;
          confidence_mem[int'(candidate_rank)] <= candidate_confidence;
          depth_mem[int'(candidate_rank)] <= '0;
        end
        if (candidate_last) begin
          allocation_active <= 1'b1;
        end
      end
      if (logical_valid && logical_ready) begin
        if (!transaction_begin) begin
          remaining_credit_bytes <= selected_global_credit_after;
          channel_remaining[selected_channel] <= selected_channel_credit_after;
        end
        channel_load[selected_channel] <= channel_load[selected_channel] + 1'b1;
        if (32'(depth_mem[selected_index]) + 1 >= TILES_PER_EXPERT) begin
          candidate_valid_mem[selected_index] <= 1'b0;
        end else begin
          depth_mem[selected_index] <= depth_mem[selected_index] + 1'b1;
        end
      end
      if (allocation_active && selection_done && !winner_valid) begin
        allocation_active <= 1'b0;
        allocation_done <= 1'b1;
      end
      if (stop_allocation === 1'b1) begin
        allocation_active <= 0;
      end
    end
  end
endmodule
