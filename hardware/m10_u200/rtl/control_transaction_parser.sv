`timescale 1ns/1ps
/** Predictor candidate 与 Router truth 的 valid/ready transaction parser. */
module control_transaction_parser #(
  parameter integer MAX_EXPERTS = 64, parameter integer EXPERT_W = $clog2(MAX_EXPERTS), parameter integer MAX_COUNT = 31,
  parameter integer TOP_K_MAX = 8
) (
  input logic clk,input logic rst_n,
  // Transaction abort is synchronous; global reset must not be gated by flush.
  input logic clear,
  input logic prediction_begin_valid,output logic prediction_begin_ready,input logic [15:0] prediction_epoch,input logic [2:0] prediction_layer,input logic [4:0] prediction_count,
  input logic candidate_valid,output logic candidate_ready,input logic [EXPERT_W-1:0] candidate_expert,input logic [4:0] candidate_rank,input logic [11:0] candidate_confidence,input logic candidate_last,
  input logic route_begin_valid,output logic route_begin_ready,input logic [15:0] route_epoch,input logic [2:0] route_layer,input logic [4:0] route_top_k,
  input logic route_expert_valid,output logic route_expert_ready,input logic [EXPERT_W-1:0] route_expert,input logic route_last,
  output logic candidate_accept,output logic [15:0] candidate_accept_epoch,
  output logic [2:0] candidate_accept_layer,
  output logic [EXPERT_W-1:0] candidate_accept_expert,
  output logic [4:0] candidate_accept_rank,
  output logic [11:0] candidate_accept_confidence,
  output logic candidate_accept_last,
  output logic route_accept,output logic [15:0] route_accept_epoch,
  output logic [2:0] route_accept_layer,
  output logic [EXPERT_W-1:0] route_accept_expert,
  output logic route_accept_last,
  output logic parser_error,output logic [31:0] stale_transaction_count
);
  logic pred_active;
  logic route_active;
  logic [4:0] pred_expected;
  logic [4:0] pred_seen;
  logic [4:0] route_expected;
  logic [4:0] route_seen;
  logic [15:0] pred_epoch_hold;
  logic [2:0] pred_layer_hold;
  logic [15:0] route_epoch_hold;
  logic [2:0] route_layer_hold;
  logic [MAX_EXPERTS-1:0] pred_seen_ids;
  logic [MAX_EXPERTS-1:0] route_seen_ids;

  assign prediction_begin_ready = !clear && !pred_active;
  assign route_begin_ready = !clear && !route_active;
  assign candidate_ready = !clear && pred_active;
  assign route_expert_ready = !clear && route_active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pred_active <= 0;
      route_active <= 0;
      pred_expected <= 0;
      pred_seen <= 0;
      route_expected <= 0;
      route_seen <= 0;
      pred_epoch_hold <= 0;
      pred_layer_hold <= 0;
      route_epoch_hold <= 0;
      route_layer_hold <= 0;
      pred_seen_ids <= 0;
      route_seen_ids <= 0;
      candidate_accept <= 0;
      candidate_accept_epoch <= 0;
      candidate_accept_layer <= 0;
      candidate_accept_expert <= 0;
      candidate_accept_rank <= 0;
      candidate_accept_confidence <= 0;
      candidate_accept_last <= 0;
      route_accept <= 0;
      route_accept_epoch <= 0;
      route_accept_layer <= 0;
      route_accept_expert <= 0;
      route_accept_last <= 0;
      parser_error <= 0;
      stale_transaction_count <= 0;
    end else if (clear) begin
      pred_active <= 0;
      route_active <= 0;
      pred_expected <= 0;
      pred_seen <= 0;
      route_expected <= 0;
      route_seen <= 0;
      pred_epoch_hold <= 0;
      pred_layer_hold <= 0;
      route_epoch_hold <= 0;
      route_layer_hold <= 0;
      pred_seen_ids <= 0;
      route_seen_ids <= 0;
      candidate_accept <= 0;
      candidate_accept_epoch <= 0;
      candidate_accept_layer <= 0;
      candidate_accept_expert <= 0;
      candidate_accept_rank <= 0;
      candidate_accept_confidence <= 0;
      candidate_accept_last <= 0;
      route_accept <= 0;
      route_accept_epoch <= 0;
      route_accept_layer <= 0;
      route_accept_expert <= 0;
      route_accept_last <= 0;
      parser_error <= 0;
      stale_transaction_count <= 0;
    end else begin
      candidate_accept <= 0;
      route_accept <= 0;
      parser_error <= 0;
      if (prediction_begin_valid && prediction_begin_ready) begin
        pred_active <= prediction_count != 0 && 32'(prediction_count) <= MAX_COUNT;
        pred_expected <= prediction_count;
        pred_seen <= 0;
        pred_seen_ids <= 0;
        pred_epoch_hold <= prediction_epoch;
        pred_layer_hold <= prediction_layer;
        if (32'(prediction_count) == 0 || 32'(prediction_count) > MAX_COUNT) begin
          parser_error <= 1;
        end
      end
      if (route_begin_valid && route_begin_ready) begin
        route_active <= route_top_k != 0 && 32'(route_top_k) <= TOP_K_MAX;
        route_expected <= route_top_k;
        route_seen <= 0;
        route_seen_ids <= 0;
        route_epoch_hold <= route_epoch;
        route_layer_hold <= route_layer;
        if (32'(route_top_k) == 0 || 32'(route_top_k) > TOP_K_MAX) begin
          parser_error <= 1;
        end
      end
      if (candidate_valid && candidate_ready) begin
        if (32'(candidate_expert) >= MAX_EXPERTS || pred_seen_ids[int'(candidate_expert)] || candidate_rank != pred_seen || candidate_confidence == 0 ||
            candidate_last != (pred_seen + 1 == pred_expected)) begin
          parser_error <= 1;
          pred_active <= 0;
        end else begin
          candidate_accept <= 1;
          candidate_accept_epoch <= pred_epoch_hold;
          candidate_accept_layer <= pred_layer_hold;
          candidate_accept_expert <= candidate_expert;
          candidate_accept_rank <= candidate_rank;
          candidate_accept_confidence <= candidate_confidence;
          candidate_accept_last <= candidate_last;
          pred_seen_ids[int'(candidate_expert)] <= 1;
          pred_seen <= pred_seen + 1;
          if (candidate_last != (pred_seen + 1 == pred_expected)) begin
            parser_error <= 1;
          end
          if (pred_seen + 1 >= pred_expected) begin
            pred_active <= 0;
          end
        end
      end
      if (route_expert_valid && route_expert_ready) begin
        if (32'(route_expert) >= MAX_EXPERTS || route_seen_ids[int'(route_expert)] || route_last != (route_seen + 1 == route_expected)) begin
          parser_error <= 1;
          route_active <= 0;
        end else begin
          route_accept <= 1;
          route_accept_epoch <= route_epoch_hold;
          route_accept_layer <= route_layer_hold;
          route_accept_expert <= route_expert;
          route_accept_last <= route_last;
          route_seen_ids[int'(route_expert)] <= 1;
          route_seen <= route_seen + 1;
          if (route_last != (route_seen + 1 == route_expected)) begin
            parser_error <= 1;
          end
          if (route_seen + 1 >= route_expected) begin
            route_active <= 0;
          end
        end
      end
    end
  end
endmodule
