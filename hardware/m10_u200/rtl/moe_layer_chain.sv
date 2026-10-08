`timescale 1ns/1ps
/** 顺序MoE-only层链：真实result依赖、配置requant、排空后清理，不实现未来层预取。 */
module moe_layer_chain #(
  parameter integer LAYER_COUNT = 3,
  parameter integer H = 16,
  parameter bit NONBLOCKING_RETIRE = 0
) (
  input logic clk,
  input logic rst_n,
  input logic start,
  output logic start_ready,
  input logic abort,
  input logic layer_accept,
  input logic layer_result_valid,
  input logic [H*32-1:0] layer_result,
  input logic transaction_drained,
  input logic context_retire_ready,
  input logic [H*8-1:0] initial_activation,
  input logic [4:0] output_requant_shift,
  output logic [H*8-1:0] activation,
  output logic layer_ready,
  output logic [2:0] current_layer,
  output logic busy,
  output logic done,
  output logic context_flush,
  output wire context_retiring
);
  typedef enum logic [2:0] {IDLE, READY, EXECUTE, DRAIN, FLUSH, ADVANCE} state_t;
  state_t state;
  logic [4:0] shift_hold;
  wire start_fire = start && start_ready;
  assign start_ready = state == IDLE && transaction_drained && !abort;
  assign layer_ready = state == READY && !abort;
  assign busy = state != IDLE;
  assign context_flush = state == FLUSH && !abort;
  assign context_retiring = state==DRAIN || state==FLUSH || state==ADVANCE ||
    (state==EXECUTE && layer_result_valid);
  function automatic logic signed [7:0] quantize(input logic signed [31:0] value, input logic [4:0] shift);
    logic signed [31:0] shifted;
    shifted = value >>> shift;
    if (shifted > 127) return 8'sd127;
    if (shifted < -128) return -8'sd128;
    return shifted[7:0];
  endfunction
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else if (abort) state <= IDLE;
    else case (state)
      IDLE: if (start_fire) state <= READY;
      READY: if (layer_accept) state <= EXECUTE;
      EXECUTE: if (layer_result_valid) state <= DRAIN;
      DRAIN: if (NONBLOCKING_RETIRE ? context_retire_ready : transaction_drained) state <= FLUSH;
      FLUSH: state <= ADVANCE;
      ADVANCE: begin
        if (int'(current_layer) == LAYER_COUNT-1) state <= IDLE;
        else state <= READY;
      end
      default: state <= IDLE;
    endcase
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) current_layer <= 0;
    else if (start_fire) current_layer <= 0;
    else if (!abort && state == ADVANCE && int'(current_layer) != LAYER_COUNT-1)
      current_layer <= current_layer+1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) done <= 0;
    else if (abort) done <= 0;
    else done <= state == ADVANCE && int'(current_layer) == LAYER_COUNT-1;
  end
  always_ff @(posedge clk) begin
    if (layer_accept && layer_ready) shift_hold <= output_requant_shift;
  end
  for (genvar lane = 0; lane < H; lane = lane+1) begin : activation_lane
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) activation[lane*8 +: 8] <= 0;
      else if (start_fire) activation[lane*8 +: 8] <= initial_activation[lane*8 +: 8];
      else if (!abort && state == EXECUTE && layer_result_valid)
        activation[lane*8 +: 8] <= quantize($signed(layer_result[lane*32 +: 32]),shift_hold);
    end
  end
  initial begin
    if (LAYER_COUNT < 1 || LAYER_COUNT > 8 || H < 1) $fatal(1,"layer chain supports 1..8 layers and positive H");
  end
endmodule
