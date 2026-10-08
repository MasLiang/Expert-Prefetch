`timescale 1ns/1ps
/**
 * 选中 logits 的确定性定点 softmax。difference >> logit_shift 为 Q4 指数距离。
 * 距离 > 8 时指数截为零；Q15 向下取整，最后一个 gate 接收余数，和为 32768。
 * 32-cycle restoring divider 避免组合除法。事务输出支持 backpressure。
 */
module router_topk_softmax #(
  parameter integer NUM_EXPERTS = 64,
  parameter integer MAX_TOP_K = 8
) (
  input logic clk,
  input logic rst_n,
  input logic flush,
  input logic start,
  input logic [NUM_EXPERTS*32-1:0] logits,
  input logic [4:0] top_k,
  input logic [4:0] logit_shift,
  output logic busy,
  output logic begin_valid,
  input logic begin_ready,
  output logic expert_valid,
  input logic expert_ready,
  output logic [9:0] expert_id,
  output logic [15:0] weight_q15,
  output logic last
);
  typedef enum logic [3:0] {IDLE, SCAN, EXP, DIV_INIT, DIV_STEP, DIV_SAVE,
                            EMIT_BEGIN, EMIT} phase_t;
  phase_t phase;
  logic [NUM_EXPERTS*32-1:0] logits_hold;
  logic [4:0] k_hold;
  logic [4:0] shift_hold;
  logic [NUM_EXPERTS-1:0] selected;
  integer scan_index;
  integer rank_index;
  integer emit_index;
  logic signed [31:0] best_logit;
  logic [9:0] best_id;
  logic best_valid;
  logic [MAX_TOP_K*10-1:0] ids;
  logic [MAX_TOP_K*32-1:0] chosen_logits;
  logic [MAX_TOP_K*17-1:0] exps;
  logic [MAX_TOP_K*16-1:0] gates;
  logic [31:0] denominator;
  logic [31:0] gate_sum;
  logic [31:0] dividend;
  logic [31:0] quotient;
  logic [31:0] remainder;
  logic [5:0] div_bit;
  wire signed [31:0] scan_logit = $signed(logits_hold[scan_index*32 +: 32]);
  wire better = !selected[scan_index] && (!best_valid || scan_logit > best_logit);
  wire [9:0] winner_id = better ? 10'(scan_index) : best_id;
  wire signed [31:0] winner_logit = better ? scan_logit : best_logit;
  wire signed [32:0] difference = $signed({chosen_logits[31], chosen_logits[31:0]}) -
    $signed({chosen_logits[rank_index*32+31], chosen_logits[rank_index*32 +: 32]});
  wire [32:0] distance_q4 = $unsigned(difference) >> shift_hold;
  wire [16:0] exp_value = distance_q4 > 128 ? 17'd0 : exp_q16(distance_q4[7:0]);
  wire [32:0] trial = {remainder, dividend[31]};
  wire subtract = trial >= {1'b0, denominator};
  assign busy = phase != IDLE;
  assign begin_valid = phase == EMIT_BEGIN;
  assign expert_valid = phase == EMIT;
  assign expert_id = ids[emit_index*10 +: 10];
  assign weight_q15 = gates[emit_index*16 +: 16];
  assign last = emit_index == int'(k_hold)-1;

  function automatic logic [16:0] exp_q16(input logic [7:0] distance);
    case (distance)
      8'd0: exp_q16 = 17'd65536;
      8'd1: exp_q16 = 17'd61565;
      8'd2: exp_q16 = 17'd57835;
      8'd3: exp_q16 = 17'd54331;
      8'd4: exp_q16 = 17'd51039;
      8'd5: exp_q16 = 17'd47947;
      8'd6: exp_q16 = 17'd45042;
      8'd7: exp_q16 = 17'd42313;
      8'd8: exp_q16 = 17'd39750;
      8'd9: exp_q16 = 17'd37341;
      8'd10: exp_q16 = 17'd35079;
      8'd11: exp_q16 = 17'd32954;
      8'd12: exp_q16 = 17'd30957;
      8'd13: exp_q16 = 17'd29081;
      8'd14: exp_q16 = 17'd27319;
      8'd15: exp_q16 = 17'd25664;
      8'd16: exp_q16 = 17'd24109;
      8'd17: exp_q16 = 17'd22649;
      8'd18: exp_q16 = 17'd21276;
      8'd19: exp_q16 = 17'd19987;
      8'd20: exp_q16 = 17'd18776;
      8'd21: exp_q16 = 17'd17639;
      8'd22: exp_q16 = 17'd16570;
      8'd23: exp_q16 = 17'd15566;
      8'd24: exp_q16 = 17'd14623;
      8'd25: exp_q16 = 17'd13737;
      8'd26: exp_q16 = 17'd12905;
      8'd27: exp_q16 = 17'd12123;
      8'd28: exp_q16 = 17'd11388;
      8'd29: exp_q16 = 17'd10698;
      8'd30: exp_q16 = 17'd10050;
      8'd31: exp_q16 = 17'd9441;
      8'd32: exp_q16 = 17'd8869;
      8'd33: exp_q16 = 17'd8332;
      8'd34: exp_q16 = 17'd7827;
      8'd35: exp_q16 = 17'd7353;
      8'd36: exp_q16 = 17'd6907;
      8'd37: exp_q16 = 17'd6489;
      8'd38: exp_q16 = 17'd6096;
      8'd39: exp_q16 = 17'd5726;
      8'd40: exp_q16 = 17'd5380;
      8'd41: exp_q16 = 17'd5054;
      8'd42: exp_q16 = 17'd4747;
      8'd43: exp_q16 = 17'd4460;
      8'd44: exp_q16 = 17'd4190;
      8'd45: exp_q16 = 17'd3936;
      8'd46: exp_q16 = 17'd3697;
      8'd47: exp_q16 = 17'd3473;
      8'd48: exp_q16 = 17'd3263;
      8'd49: exp_q16 = 17'd3065;
      8'd50: exp_q16 = 17'd2879;
      8'd51: exp_q16 = 17'd2705;
      8'd52: exp_q16 = 17'd2541;
      8'd53: exp_q16 = 17'd2387;
      8'd54: exp_q16 = 17'd2243;
      8'd55: exp_q16 = 17'd2107;
      8'd56: exp_q16 = 17'd1979;
      8'd57: exp_q16 = 17'd1859;
      8'd58: exp_q16 = 17'd1746;
      8'd59: exp_q16 = 17'd1641;
      8'd60: exp_q16 = 17'd1541;
      8'd61: exp_q16 = 17'd1448;
      8'd62: exp_q16 = 17'd1360;
      8'd63: exp_q16 = 17'd1278;
      8'd64: exp_q16 = 17'd1200;
      8'd65: exp_q16 = 17'd1128;
      8'd66: exp_q16 = 17'd1059;
      8'd67: exp_q16 = 17'd995;
      8'd68: exp_q16 = 17'd935;
      8'd69: exp_q16 = 17'd878;
      8'd70: exp_q16 = 17'd825;
      8'd71: exp_q16 = 17'd775;
      8'd72: exp_q16 = 17'd728;
      8'd73: exp_q16 = 17'd684;
      8'd74: exp_q16 = 17'd642;
      8'd75: exp_q16 = 17'd604;
      8'd76: exp_q16 = 17'd567;
      8'd77: exp_q16 = 17'd533;
      8'd78: exp_q16 = 17'd500;
      8'd79: exp_q16 = 17'd470;
      8'd80: exp_q16 = 17'd442;
      8'd81: exp_q16 = 17'd415;
      8'd82: exp_q16 = 17'd390;
      8'd83: exp_q16 = 17'd366;
      8'd84: exp_q16 = 17'd344;
      8'd85: exp_q16 = 17'd323;
      8'd86: exp_q16 = 17'd303;
      8'd87: exp_q16 = 17'd285;
      8'd88: exp_q16 = 17'd268;
      8'd89: exp_q16 = 17'd252;
      8'd90: exp_q16 = 17'd236;
      8'd91: exp_q16 = 17'd222;
      8'd92: exp_q16 = 17'd209;
      8'd93: exp_q16 = 17'd196;
      8'd94: exp_q16 = 17'd184;
      8'd95: exp_q16 = 17'd173;
      8'd96: exp_q16 = 17'd162;
      8'd97: exp_q16 = 17'd153;
      8'd98: exp_q16 = 17'd143;
      8'd99: exp_q16 = 17'd135;
      8'd100: exp_q16 = 17'd127;
      8'd101: exp_q16 = 17'd119;
      8'd102: exp_q16 = 17'd112;
      8'd103: exp_q16 = 17'd105;
      8'd104: exp_q16 = 17'd99;
      8'd105: exp_q16 = 17'd93;
      8'd106: exp_q16 = 17'd87;
      8'd107: exp_q16 = 17'd82;
      8'd108: exp_q16 = 17'd77;
      8'd109: exp_q16 = 17'd72;
      8'd110: exp_q16 = 17'd68;
      8'd111: exp_q16 = 17'd64;
      8'd112: exp_q16 = 17'd60;
      8'd113: exp_q16 = 17'd56;
      8'd114: exp_q16 = 17'd53;
      8'd115: exp_q16 = 17'd50;
      8'd116: exp_q16 = 17'd47;
      8'd117: exp_q16 = 17'd44;
      8'd118: exp_q16 = 17'd41;
      8'd119: exp_q16 = 17'd39;
      8'd120: exp_q16 = 17'd36;
      8'd121: exp_q16 = 17'd34;
      8'd122: exp_q16 = 17'd32;
      8'd123: exp_q16 = 17'd30;
      8'd124: exp_q16 = 17'd28;
      8'd125: exp_q16 = 17'd27;
      8'd126: exp_q16 = 17'd25;
      8'd127: exp_q16 = 17'd23;
      8'd128: exp_q16 = 17'd22;
      default: exp_q16 = 0;
    endcase
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) phase <= IDLE;
    else if (flush) phase <= IDLE;
    else case (phase)
      IDLE: if (start && top_k > 0 && int'(top_k) <= MAX_TOP_K &&
                         int'(top_k) <= NUM_EXPERTS) phase <= SCAN;
      SCAN: if (scan_index == NUM_EXPERTS-1 && rank_index == int'(k_hold)-1) phase <= EXP;
      EXP: if (rank_index == int'(k_hold)-1) phase <= DIV_INIT;
      DIV_INIT: if (rank_index == int'(k_hold)-1) phase <= DIV_SAVE; else phase <= DIV_STEP;
      DIV_STEP: if (div_bit == 31) phase <= DIV_SAVE;
      DIV_SAVE: if (rank_index == int'(k_hold)-1) phase <= EMIT_BEGIN; else phase <= DIV_INIT;
      EMIT_BEGIN: if (begin_ready) phase <= EMIT;
      EMIT: if (expert_ready && last) phase <= IDLE;
      default: phase <= IDLE;
    endcase
  end
  always_ff @(posedge clk) if (phase == IDLE && start) logits_hold <= logits;
  always_ff @(posedge clk) if (phase == IDLE && start) k_hold <= top_k;
  always_ff @(posedge clk) if (phase == IDLE && start) shift_hold <= logit_shift;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) scan_index <= 0;
    else if (phase == IDLE || (phase == SCAN && scan_index == NUM_EXPERTS-1)) scan_index <= 0;
    else if (phase == SCAN) scan_index <= scan_index + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) rank_index <= 0;
    else if (phase == IDLE) rank_index <= 0;
    else if ((phase == SCAN && scan_index == NUM_EXPERTS-1) || phase == EXP || phase == DIV_SAVE)
      rank_index <= rank_index == int'(k_hold)-1 ? 0 : rank_index + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) selected <= 0;
    else if (phase == IDLE) selected <= 0;
    else if (phase == SCAN && scan_index == NUM_EXPERTS-1) selected[int'(winner_id)] <= 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) best_valid <= 0;
    else if (phase != SCAN || scan_index == NUM_EXPERTS-1) best_valid <= 0;
    else if (better) best_valid <= 1;
  end
  always_ff @(posedge clk) if (phase == SCAN && better) best_logit <= scan_logit;
  always_ff @(posedge clk) if (phase == SCAN && better) best_id <= 10'(scan_index);
  always_ff @(posedge clk) if (phase == SCAN && scan_index == NUM_EXPERTS-1) ids[rank_index*10 +: 10] <= winner_id;
  always_ff @(posedge clk) if (phase == SCAN && scan_index == NUM_EXPERTS-1) chosen_logits[rank_index*32 +: 32] <= winner_logit;
  always_ff @(posedge clk) if (phase == EXP) exps[rank_index*17 +: 17] <= exp_value;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) denominator <= 0;
    else if (phase == IDLE) denominator <= 0;
    else if (phase == EXP) denominator <= denominator + {15'd0, exp_value};
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) dividend <= 0;
    else if (phase == DIV_INIT) dividend <= {exps[rank_index*17 +: 17], 15'd0};
    else if (phase == DIV_STEP) dividend <= {dividend[30:0], 1'b0};
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) remainder <= 0;
    else if (phase == DIV_INIT) remainder <= 0;
    else if (phase == DIV_STEP) remainder <= 32'(subtract ? trial - {1'b0, denominator} : trial);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) quotient <= 0;
    else if (phase == DIV_INIT) quotient <= 0;
    else if (phase == DIV_STEP) quotient <= {quotient[30:0], subtract};
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) div_bit <= 0;
    else if (phase == DIV_INIT) div_bit <= 0;
    else if (phase == DIV_STEP) div_bit <= div_bit + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gate_sum <= 0;
    else if (phase == IDLE) gate_sum <= 0;
    else if (phase == DIV_SAVE && rank_index != int'(k_hold)-1) gate_sum <= gate_sum + quotient;
  end
  always_ff @(posedge clk) if (phase == DIV_SAVE)
    gates[rank_index*16 +: 16] <= rank_index == int'(k_hold)-1 ? 16'(32768-gate_sum) : quotient[15:0];
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) emit_index <= 0;
    else if (phase == IDLE) emit_index <= 0;
    else if (phase == EMIT && expert_ready && !last) emit_index <= emit_index + 1;
  end
  initial begin
    if (NUM_EXPERTS < 1 || NUM_EXPERTS > 1024 || MAX_TOP_K < 1 || MAX_TOP_K > 8)
      $fatal(1, "unsupported Router geometry");
  end
endmodule
