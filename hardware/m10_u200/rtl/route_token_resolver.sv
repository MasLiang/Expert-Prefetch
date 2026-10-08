`timescale 1ns/1ps
/**
 * 将 Router 的 expert truth transaction 解析为 descriptor-token commit/cancel。
 *
 * 每个 accepted physical descriptor 在 route 到达前保存在有界 KeyResolver 中。
 * 收到 route_last 后，resolver 对同一 epoch/layer 的所有 live descriptor 发出
 * 一个 token：true expert commit，其他 speculative expert cancel。entry 仅在
 * token 被 core 接收后退休，因而不会在 route backpressure 下丢失请求。
 */
module route_token_resolver #(
  parameter integer MAX_LIVE = 64,
  parameter integer TOP_K_MAX = 8,
  parameter integer EXPERT_W = 10
) (
  input logic clk,
  input logic rst_n,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic capture_valid,
  input tile_prefetch_package::descriptor_token_t capture_token,
  input tile_prefetch_package::tile_descriptor_t capture_descriptor,
  input logic route_accept,
  input logic [15:0] route_epoch,
  input logic [2:0] route_layer,
  input logic [EXPERT_W-1:0] route_expert,
  input logic route_last,
  output logic route_accept_ready,
  output logic resolution_done,
  output logic route_valid,
  output tile_prefetch_package::descriptor_token_t route_token,
  output logic route_truth,
  input logic route_ready,
  output logic resolver_full,
  output logic [31:0] dropped_capture_count
);
  import tile_prefetch_package::*;
  localparam integer INDEX_W = (MAX_LIVE <= 1) ? 1 : $clog2(MAX_LIVE);
  localparam integer TOPK_W = (TOP_K_MAX <= 1) ? 1 : $clog2(TOP_K_MAX + 1);

  logic valid_mem [0:MAX_LIVE-1];
  descriptor_token_t token_mem [0:MAX_LIVE-1];
  tile_descriptor_t descriptor_mem [0:MAX_LIVE-1];
  logic [EXPERT_W-1:0] truth_mem [0:TOP_K_MAX-1];
  logic [TOPK_W-1:0] truth_count;
  logic [15:0] route_epoch_hold;
  logic [2:0] route_layer_hold;
  logic scanning;
  logic [2:0] drain_count;
  logic [INDEX_W:0] scan_index;
  logic capture_found;
  logic [INDEX_W-1:0] capture_index;
  logic selected_matches_truth;
  logic selected_valid;
  descriptor_token_t selected_token;
  tile_descriptor_t selected_descriptor;
  tile_descriptor_t flush_descriptor;

  assign route_accept_ready = !scanning && drain_count == 0;

  always_comb begin
    capture_found = 1'b0;
    capture_index = '0;
    for (integer index = 0; index < MAX_LIVE; index = index + 1) begin
      if (!capture_found && !valid_mem[index]) begin
        capture_found = 1'b1;
        capture_index = index[INDEX_W-1:0];
      end
    end
    resolver_full = !capture_found;
    selected_valid = 1'b0;
    selected_token = '0;
    selected_descriptor = '0;
    if (32'(scan_index) < MAX_LIVE) begin
      selected_valid = valid_mem[int'(scan_index)];
      selected_token = token_mem[int'(scan_index)];
      selected_descriptor = descriptor_mem[int'(scan_index)];
    end
    selected_matches_truth = 1'b0;
    for (integer index = 0; index < TOP_K_MAX; index = index + 1) begin
      if (index < truth_count && selected_descriptor.expert_id == truth_mem[index]) begin
        selected_matches_truth = 1'b1;
      end
    end
    route_valid = scanning && 32'(scan_index) < MAX_LIVE &&
      selected_valid &&
      selected_descriptor.epoch_id == route_epoch_hold &&
      selected_descriptor.layer_id[2:0] == route_layer_hold;
    route_token = selected_token;
    route_truth = selected_matches_truth;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      truth_count <= '0;
      route_epoch_hold <= '0;
      route_layer_hold <= '0;
      scanning <= 1'b0;
      drain_count <= 0;
      resolution_done <= 0;
      scan_index <= '0;
      dropped_capture_count <= '0;
      for (integer index = 0; index < MAX_LIVE; index = index + 1) begin
        valid_mem[index] <= 1'b0;
        token_mem[index] <= '0;
        descriptor_mem[index] <= '0;
      end
      for (integer index = 0; index < TOP_K_MAX; index = index + 1) begin
        truth_mem[index] <= '0;
      end
    end else begin
      resolution_done <= 0;
      if (capture_valid) begin
        if (capture_found) begin
          valid_mem[capture_index] <= 1'b1;
          token_mem[capture_index] <= capture_token;
          descriptor_mem[capture_index] <= capture_descriptor;
        end else begin
          dropped_capture_count <= dropped_capture_count + 1'b1;
        end
      end
      if (route_accept && route_accept_ready) begin
        if (32'(truth_count) == 0) begin
          route_epoch_hold <= route_epoch;
          route_layer_hold <= route_layer;
        end
        if (32'(truth_count) < TOP_K_MAX) begin
          truth_mem[int'(truth_count)] <= route_expert;
          truth_count <= truth_count + 1'b1;
        end
        if (route_last) begin
          drain_count <= 4;
          scan_index <= '0;
        end
      end
      if (drain_count != 0) begin
        drain_count <= drain_count - 1'b1;
        if (drain_count == 1) scanning <= 1'b1;
      end
      if (scanning) begin
        if (32'(scan_index) >= MAX_LIVE) begin
          scanning <= 1'b0;
          truth_count <= '0;
          resolution_done <= 1;
        end else if (route_valid) begin
          if (route_ready) begin
            valid_mem[scan_index[INDEX_W-1:0]] <= 1'b0;
            scan_index <= scan_index + 1'b1;
          end
        end else begin
          scan_index <= scan_index + 1'b1;
        end
      end
      if (soft_flush_valid === 1'b1) begin
        scanning <= 0;
        drain_count <= 0;
        truth_count <= 0;
        for (integer i = 0; i < MAX_LIVE; i = i + 1) begin
          flush_descriptor = descriptor_mem[i];
          if (flush_descriptor.epoch_id == soft_flush_epoch) valid_mem[i] <= 0;
        end
      end
    end
  end
endmodule
