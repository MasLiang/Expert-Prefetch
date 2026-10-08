`timescale 1ns/1ps
/**
 * 16-lane signed int8 MAC，int32 wrap accumulation 的可选 SwiGLU/Switch F-tile。
 * H、F_TILE 均为 16 的倍数。GU payload 依次为 output-major G[F,H]、
 * U[F,H]；DOWN payload 为 output-major W[H,F]，每 beat 连续 16 个 byte。
 * GU requant shift 由 host 提供；G/U 为 Q4 int8，SiLU LUT 同为 Q4；
 * SiLU(G)*U 算术右移 4 并饱和至 int8，再进行 DOWN MAC。
 * SWITCH_RELU=1 时 GU 仅存 W_up[F,H]，执行 saturate_int8(sum >>> shift)
 * 后取 ReLU；不读取第二份矩阵，不执行 SiLU 或逐元素乘法。
 */
module int8_swiglu_tile_backend #(
  parameter integer H = 16,
  parameter integer F_TILE = 16,
  parameter bit SWITCH_RELU = 1'b0,
  parameter integer NUM_ROUTER_EXPERTS = 64
) (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic gu_start,
  input logic down_start,
  input logic router_start,
  output logic router_done,
  output logic router_read_valid,
  output logic [31:0] router_read_offset,
  input logic [127:0] router_read_data,
  output logic [NUM_ROUTER_EXPERTS*32-1:0] router_logits,
  input logic [H*8-1:0] activation,
  input logic [4:0] gu_requant_shift,
  input logic [15:0] gu_slot,
  input logic [15:0] down_slot,
  output logic read_valid,
  output logic [15:0] read_slot,
  output logic [31:0] read_offset,
  input logic [127:0] read_data,
  output logic gu_done,
  output logic nonlinear_done,
  output logic down_done,
  output logic [H*32-1:0] partial_y
);
  typedef enum logic [1:0] {IDLE, STREAM, NONLINEAR, WAIT_DOWN} phase_t;
  phase_t phase;
  logic [H*8-1:0] activation_hold;
  logic [4:0] shift_hold;
  logic [15:0] gu_slot_hold;
  logic [15:0] down_slot_hold;
  logic [31:0] row_index;
  logic [31:0] chunk_index;
  logic signed [31:0] accumulator;
  logic down_mode;
  logic router_mode;
  logic issue_active;
  wire issue = !clear && phase == STREAM && issue_active;
  wire router_fire = !clear && phase == IDLE && router_start === 1'b1;
  wire gu_fire = !clear && phase == IDLE && gu_start && router_start !== 1'b1;
  wire down_fire = !clear && phase == WAIT_DOWN && down_start;
  wire begin_operation = router_fire || gu_fire || down_fire;
  wire [31:0] chunks_per_row = down_mode ? F_TILE/16 : H/16;
  wire [31:0] rows_per_operation = router_mode ? NUM_ROUTER_EXPERTS : down_mode ? H :
    (SWITCH_RELU ? F_TILE : 2*F_TILE);
  wire issue_last = chunk_index == chunks_per_row-1;
  logic [6:0] valid_pipe;
  logic [31:0] row_pipe [0:6];
  logic first_pipe [0:6];
  logic last_pipe [0:6];
  logic [31:0] read_chunk;
  wire row_store = valid_pipe[6] && last_pipe[6] && !clear;
  wire operation_finished = row_store && row_pipe[6] == rows_per_operation-1;
  assign router_read_valid = issue && router_mode;
  assign router_read_offset = row_index * H + chunk_index * 16;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) router_mode <= 0;
    else if (clear) router_mode <= 0;
    else if (router_fire) router_mode <= 1;
    else if (gu_fire) router_mode <= 0;
  end
  logic signed [15:0] products [0:15];
  logic signed [16:0] pairs [0:7];
  logic signed [17:0] quads [0:3];
  logic signed [18:0] octets [0:1];
  logic signed [19:0] dot_sum;
  logic signed [31:0] final_sum;
  logic signed [7:0] gate_values [0:F_TILE-1];
  logic signed [7:0] up_values [0:F_TILE-1];
  logic signed [7:0] nonlinear_values [0:F_TILE-1];
  logic signed [15:0] nonlinear_product;

  function automatic logic signed [7:0] saturate_int8(input logic signed [31:0] value);
    if (value > 127) saturate_int8 = 8'sd127;
    else if (value < -128) saturate_int8 = -8'sd128;
    else saturate_int8 = value[7:0];
  endfunction

  function automatic logic signed [7:0] silu_q4(input logic [7:0] value);
    case (value)
      8'd0: silu_q4 = 8'h00;
      8'd1: silu_q4 = 8'h01;
      8'd2: silu_q4 = 8'h01;
      8'd3: silu_q4 = 8'h02;
      8'd4: silu_q4 = 8'h02;
      8'd5: silu_q4 = 8'h03;
      8'd6: silu_q4 = 8'h04;
      8'd7: silu_q4 = 8'h04;
      8'd8: silu_q4 = 8'h05;
      8'd9: silu_q4 = 8'h06;
      8'd10: silu_q4 = 8'h07;
      8'd11: silu_q4 = 8'h07;
      8'd12: silu_q4 = 8'h08;
      8'd13: silu_q4 = 8'h09;
      8'd14: silu_q4 = 8'h0a;
      8'd15: silu_q4 = 8'h0b;
      8'd16: silu_q4 = 8'h0c;
      8'd17: silu_q4 = 8'h0d;
      8'd18: silu_q4 = 8'h0e;
      8'd19: silu_q4 = 8'h0f;
      8'd20: silu_q4 = 8'h10;
      8'd21: silu_q4 = 8'h11;
      8'd22: silu_q4 = 8'h12;
      8'd23: silu_q4 = 8'h13;
      8'd24: silu_q4 = 8'h14;
      8'd25: silu_q4 = 8'h15;
      8'd26: silu_q4 = 8'h16;
      8'd27: silu_q4 = 8'h17;
      8'd28: silu_q4 = 8'h18;
      8'd29: silu_q4 = 8'h19;
      8'd30: silu_q4 = 8'h1a;
      8'd31: silu_q4 = 8'h1b;
      8'd32: silu_q4 = 8'h1c;
      8'd33: silu_q4 = 8'h1d;
      8'd34: silu_q4 = 8'h1e;
      8'd35: silu_q4 = 8'h1f;
      8'd36: silu_q4 = 8'h21;
      8'd37: silu_q4 = 8'h22;
      8'd38: silu_q4 = 8'h23;
      8'd39: silu_q4 = 8'h24;
      8'd40: silu_q4 = 8'h25;
      8'd41: silu_q4 = 8'h26;
      8'd42: silu_q4 = 8'h27;
      8'd43: silu_q4 = 8'h28;
      8'd44: silu_q4 = 8'h29;
      8'd45: silu_q4 = 8'h2a;
      8'd46: silu_q4 = 8'h2c;
      8'd47: silu_q4 = 8'h2d;
      8'd48: silu_q4 = 8'h2e;
      8'd49: silu_q4 = 8'h2f;
      8'd50: silu_q4 = 8'h30;
      8'd51: silu_q4 = 8'h31;
      8'd52: silu_q4 = 8'h32;
      8'd53: silu_q4 = 8'h33;
      8'd54: silu_q4 = 8'h34;
      8'd55: silu_q4 = 8'h35;
      8'd56: silu_q4 = 8'h36;
      8'd57: silu_q4 = 8'h37;
      8'd58: silu_q4 = 8'h38;
      8'd59: silu_q4 = 8'h3a;
      8'd60: silu_q4 = 8'h3b;
      8'd61: silu_q4 = 8'h3c;
      8'd62: silu_q4 = 8'h3d;
      8'd63: silu_q4 = 8'h3e;
      8'd64: silu_q4 = 8'h3f;
      8'd65: silu_q4 = 8'h40;
      8'd66: silu_q4 = 8'h41;
      8'd67: silu_q4 = 8'h42;
      8'd68: silu_q4 = 8'h43;
      8'd69: silu_q4 = 8'h44;
      8'd70: silu_q4 = 8'h45;
      8'd71: silu_q4 = 8'h46;
      8'd72: silu_q4 = 8'h47;
      8'd73: silu_q4 = 8'h48;
      8'd74: silu_q4 = 8'h49;
      8'd75: silu_q4 = 8'h4a;
      8'd76: silu_q4 = 8'h4b;
      8'd77: silu_q4 = 8'h4c;
      8'd78: silu_q4 = 8'h4d;
      8'd79: silu_q4 = 8'h4e;
      8'd80: silu_q4 = 8'h4f;
      8'd81: silu_q4 = 8'h50;
      8'd82: silu_q4 = 8'h52;
      8'd83: silu_q4 = 8'h53;
      8'd84: silu_q4 = 8'h54;
      8'd85: silu_q4 = 8'h55;
      8'd86: silu_q4 = 8'h56;
      8'd87: silu_q4 = 8'h57;
      8'd88: silu_q4 = 8'h58;
      8'd89: silu_q4 = 8'h59;
      8'd90: silu_q4 = 8'h5a;
      8'd91: silu_q4 = 8'h5b;
      8'd92: silu_q4 = 8'h5c;
      8'd93: silu_q4 = 8'h5d;
      8'd94: silu_q4 = 8'h5e;
      8'd95: silu_q4 = 8'h5f;
      8'd96: silu_q4 = 8'h60;
      8'd97: silu_q4 = 8'h61;
      8'd98: silu_q4 = 8'h62;
      8'd99: silu_q4 = 8'h63;
      8'd100: silu_q4 = 8'h64;
      8'd101: silu_q4 = 8'h65;
      8'd102: silu_q4 = 8'h66;
      8'd103: silu_q4 = 8'h67;
      8'd104: silu_q4 = 8'h68;
      8'd105: silu_q4 = 8'h69;
      8'd106: silu_q4 = 8'h6a;
      8'd107: silu_q4 = 8'h6b;
      8'd108: silu_q4 = 8'h6c;
      8'd109: silu_q4 = 8'h6d;
      8'd110: silu_q4 = 8'h6e;
      8'd111: silu_q4 = 8'h6f;
      8'd112: silu_q4 = 8'h70;
      8'd113: silu_q4 = 8'h71;
      8'd114: silu_q4 = 8'h72;
      8'd115: silu_q4 = 8'h73;
      8'd116: silu_q4 = 8'h74;
      8'd117: silu_q4 = 8'h75;
      8'd118: silu_q4 = 8'h76;
      8'd119: silu_q4 = 8'h77;
      8'd120: silu_q4 = 8'h78;
      8'd121: silu_q4 = 8'h79;
      8'd122: silu_q4 = 8'h7a;
      8'd123: silu_q4 = 8'h7b;
      8'd124: silu_q4 = 8'h7c;
      8'd125: silu_q4 = 8'h7d;
      8'd126: silu_q4 = 8'h7e;
      8'd127: silu_q4 = 8'h7f;
      8'd128: silu_q4 = 8'h00;
      8'd129: silu_q4 = 8'h00;
      8'd130: silu_q4 = 8'h00;
      8'd131: silu_q4 = 8'h00;
      8'd132: silu_q4 = 8'h00;
      8'd133: silu_q4 = 8'h00;
      8'd134: silu_q4 = 8'h00;
      8'd135: silu_q4 = 8'h00;
      8'd136: silu_q4 = 8'h00;
      8'd137: silu_q4 = 8'h00;
      8'd138: silu_q4 = 8'h00;
      8'd139: silu_q4 = 8'h00;
      8'd140: silu_q4 = 8'h00;
      8'd141: silu_q4 = 8'h00;
      8'd142: silu_q4 = 8'h00;
      8'd143: silu_q4 = 8'h00;
      8'd144: silu_q4 = 8'h00;
      8'd145: silu_q4 = 8'h00;
      8'd146: silu_q4 = 8'h00;
      8'd147: silu_q4 = 8'h00;
      8'd148: silu_q4 = 8'h00;
      8'd149: silu_q4 = 8'h00;
      8'd150: silu_q4 = 8'h00;
      8'd151: silu_q4 = 8'h00;
      8'd152: silu_q4 = 8'h00;
      8'd153: silu_q4 = 8'h00;
      8'd154: silu_q4 = 8'h00;
      8'd155: silu_q4 = 8'h00;
      8'd156: silu_q4 = 8'h00;
      8'd157: silu_q4 = 8'h00;
      8'd158: silu_q4 = 8'h00;
      8'd159: silu_q4 = 8'h00;
      8'd160: silu_q4 = 8'h00;
      8'd161: silu_q4 = 8'h00;
      8'd162: silu_q4 = 8'h00;
      8'd163: silu_q4 = 8'h00;
      8'd164: silu_q4 = 8'h00;
      8'd165: silu_q4 = 8'h00;
      8'd166: silu_q4 = 8'h00;
      8'd167: silu_q4 = 8'h00;
      8'd168: silu_q4 = 8'h00;
      8'd169: silu_q4 = 8'h00;
      8'd170: silu_q4 = 8'h00;
      8'd171: silu_q4 = 8'h00;
      8'd172: silu_q4 = 8'h00;
      8'd173: silu_q4 = 8'h00;
      8'd174: silu_q4 = 8'h00;
      8'd175: silu_q4 = 8'hff;
      8'd176: silu_q4 = 8'hff;
      8'd177: silu_q4 = 8'hff;
      8'd178: silu_q4 = 8'hff;
      8'd179: silu_q4 = 8'hff;
      8'd180: silu_q4 = 8'hff;
      8'd181: silu_q4 = 8'hff;
      8'd182: silu_q4 = 8'hff;
      8'd183: silu_q4 = 8'hff;
      8'd184: silu_q4 = 8'hff;
      8'd185: silu_q4 = 8'hff;
      8'd186: silu_q4 = 8'hff;
      8'd187: silu_q4 = 8'hff;
      8'd188: silu_q4 = 8'hff;
      8'd189: silu_q4 = 8'hff;
      8'd190: silu_q4 = 8'hff;
      8'd191: silu_q4 = 8'hff;
      8'd192: silu_q4 = 8'hff;
      8'd193: silu_q4 = 8'hff;
      8'd194: silu_q4 = 8'hff;
      8'd195: silu_q4 = 8'hff;
      8'd196: silu_q4 = 8'hff;
      8'd197: silu_q4 = 8'hff;
      8'd198: silu_q4 = 8'hfe;
      8'd199: silu_q4 = 8'hfe;
      8'd200: silu_q4 = 8'hfe;
      8'd201: silu_q4 = 8'hfe;
      8'd202: silu_q4 = 8'hfe;
      8'd203: silu_q4 = 8'hfe;
      8'd204: silu_q4 = 8'hfe;
      8'd205: silu_q4 = 8'hfe;
      8'd206: silu_q4 = 8'hfe;
      8'd207: silu_q4 = 8'hfe;
      8'd208: silu_q4 = 8'hfe;
      8'd209: silu_q4 = 8'hfe;
      8'd210: silu_q4 = 8'hfe;
      8'd211: silu_q4 = 8'hfd;
      8'd212: silu_q4 = 8'hfd;
      8'd213: silu_q4 = 8'hfd;
      8'd214: silu_q4 = 8'hfd;
      8'd215: silu_q4 = 8'hfd;
      8'd216: silu_q4 = 8'hfd;
      8'd217: silu_q4 = 8'hfd;
      8'd218: silu_q4 = 8'hfd;
      8'd219: silu_q4 = 8'hfd;
      8'd220: silu_q4 = 8'hfd;
      8'd221: silu_q4 = 8'hfc;
      8'd222: silu_q4 = 8'hfc;
      8'd223: silu_q4 = 8'hfc;
      8'd224: silu_q4 = 8'hfc;
      8'd225: silu_q4 = 8'hfc;
      8'd226: silu_q4 = 8'hfc;
      8'd227: silu_q4 = 8'hfc;
      8'd228: silu_q4 = 8'hfc;
      8'd229: silu_q4 = 8'hfc;
      8'd230: silu_q4 = 8'hfc;
      8'd231: silu_q4 = 8'hfc;
      8'd232: silu_q4 = 8'hfc;
      8'd233: silu_q4 = 8'hfc;
      8'd234: silu_q4 = 8'hfc;
      8'd235: silu_q4 = 8'hfc;
      8'd236: silu_q4 = 8'hfc;
      8'd237: silu_q4 = 8'hfc;
      8'd238: silu_q4 = 8'hfc;
      8'd239: silu_q4 = 8'hfc;
      8'd240: silu_q4 = 8'hfc;
      8'd241: silu_q4 = 8'hfc;
      8'd242: silu_q4 = 8'hfc;
      8'd243: silu_q4 = 8'hfc;
      8'd244: silu_q4 = 8'hfc;
      8'd245: silu_q4 = 8'hfc;
      8'd246: silu_q4 = 8'hfd;
      8'd247: silu_q4 = 8'hfd;
      8'd248: silu_q4 = 8'hfd;
      8'd249: silu_q4 = 8'hfd;
      8'd250: silu_q4 = 8'hfe;
      8'd251: silu_q4 = 8'hfe;
      8'd252: silu_q4 = 8'hfe;
      8'd253: silu_q4 = 8'hff;
      8'd254: silu_q4 = 8'hff;
      8'd255: silu_q4 = 8'h00;
    endcase
  endfunction

  initial begin
    if (H < 16 || H % 16 != 0 || F_TILE < 16 || F_TILE % 16 != 0)
      $fatal(1, "H and F_TILE must be positive multiples of 16");
  end

  assign read_valid = issue && !router_mode;
  assign read_slot = down_mode ? down_slot_hold : gu_slot_hold;
  assign read_offset = !down_mode ? row_index * H + chunk_index * 16 :
      row_index * F_TILE + chunk_index * 16;
  if (SWITCH_RELU) begin : relu_product_unused
    assign nonlinear_product = '0;
  end else begin : swiglu_product
    assign nonlinear_product = silu_q4(gate_values[row_index % F_TILE]) * up_values[row_index % F_TILE];
  end

  // 同步RAM读 -> 乘法 -> pair -> quad -> octet -> total -> accumulate -> store。
  // 每cycle可发一个chunk；row/first/last标签与数据一起流动，末chunk写回后才stage-done。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) valid_pipe <= '0;
    else if (clear) valid_pipe <= '0;
    else valid_pipe <= {valid_pipe[5:0],issue};
  end
  always_ff @(posedge clk) begin
    if (issue) read_chunk <= chunk_index;
  end
  for (genvar stage = 0; stage < 7; stage = stage + 1) begin : tags
    if (stage == 0) begin : read_tag
      always_ff @(posedge clk) begin
        if (issue) row_pipe[stage] <= row_index;
      end
      always_ff @(posedge clk) begin
        if (issue) first_pipe[stage] <= chunk_index == 0;
      end
      always_ff @(posedge clk) begin
        if (issue) last_pipe[stage] <= issue_last;
      end
    end else begin : shift_tag
      always_ff @(posedge clk) begin
        if (valid_pipe[stage-1]) row_pipe[stage] <= row_pipe[stage-1];
      end
      always_ff @(posedge clk) begin
        if (valid_pipe[stage-1]) first_pipe[stage] <= first_pipe[stage-1];
      end
      always_ff @(posedge clk) begin
        if (valid_pipe[stage-1]) last_pipe[stage] <= last_pipe[stage-1];
      end
    end
  end

  // BRAM -> int8 product -> 17/18/19/20-bit balanced reduction -> int32 accumulate。
  // 每一级只覆盖一层运算，且所有中间位宽足以无损容纳 16 路 signed int8 乘积。
  for (genvar lane = 0; lane < 16; lane = lane + 1) begin : multiply_stage
    wire signed [7:0] operand = down_mode ? nonlinear_values[read_chunk*16+lane] :
      $signed(activation_hold[(read_chunk*16+lane)*8 +: 8]);
    wire signed [7:0] weight = router_mode ? $signed(router_read_data[lane*8 +: 8]) :
      $signed(read_data[lane*8 +: 8]);
    always_ff @(posedge clk) begin
      if (valid_pipe[0] && !clear) products[lane] <= operand * weight;
    end
  end
  for (genvar lane = 0; lane < 8; lane = lane + 1) begin : pair_stage
    always_ff @(posedge clk) begin
      if (valid_pipe[1] && !clear) pairs[lane] <= 17'(products[2*lane]) + 17'(products[2*lane+1]);
    end
  end
  for (genvar lane = 0; lane < 4; lane = lane + 1) begin : quad_stage
    always_ff @(posedge clk) begin
      if (valid_pipe[2] && !clear) quads[lane] <= 18'(pairs[2*lane]) + 18'(pairs[2*lane+1]);
    end
  end
  for (genvar lane = 0; lane < 2; lane = lane + 1) begin : octet_stage
    always_ff @(posedge clk) begin
      if (valid_pipe[3] && !clear) octets[lane] <= 19'(quads[2*lane]) + 19'(quads[2*lane+1]);
    end
  end
  always_ff @(posedge clk) begin
    if (valid_pipe[4] && !clear) dot_sum <= 20'(octets[0]) + 20'(octets[1]);
  end
  always_ff @(posedge clk) begin
    if (valid_pipe[5] && !clear) final_sum <= first_pipe[5] ? 32'(dot_sum) : accumulator + 32'(dot_sum);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_mode <= 0;
    else if (clear) down_mode <= 0;
    else if (gu_fire || router_fire) down_mode <= 0;
    else if (down_fire) down_mode <= 1;
  end

  // Intermediate payload is fully overwritten before it is consumed. Reset
  // only invalidates the control phase; do not put these arrays on an
  // asynchronous-reset process or reset their contents.
  always_ff @(posedge clk) begin
    if (rst_n && row_store && !down_mode && !router_mode && row_pipe[6] < F_TILE)
      gate_values[row_pipe[6]] <= saturate_int8(final_sum >>> shift_hold);
  end

  always_ff @(posedge clk) begin
    if (!SWITCH_RELU && rst_n && row_store && !down_mode && !router_mode && row_pipe[6] >= F_TILE)
      up_values[row_pipe[6]-F_TILE] <= saturate_int8(final_sum >>> shift_hold);
  end

  always_ff @(posedge clk) begin
    if (rst_n && !clear && phase == NONLINEAR)
      nonlinear_values[row_index] <= SWITCH_RELU ?
        (gate_values[row_index] > 0 ? gate_values[row_index] : 8'sd0) :
        saturate_int8(32'($signed(nonlinear_product)) >>> 4);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) phase <= IDLE;
    else if (clear) phase <= IDLE;
    else if (begin_operation) phase <= STREAM;
    else if (operation_finished) begin
      if (!router_mode && !down_mode) phase <= NONLINEAR;
      else phase <= IDLE;
    end
    else if (phase == NONLINEAR && row_index == F_TILE-1) phase <= WAIT_DOWN;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) issue_active <= 0;
    else if (clear) issue_active <= 0;
    else if (begin_operation) issue_active <= 1;
    else if (issue && issue_last && row_index == rows_per_operation-1) issue_active <= 0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) activation_hold <= 0;
    else if (clear) activation_hold <= 0;
    else if (router_fire || gu_fire) activation_hold <= activation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) shift_hold <= 0;
    else if (clear) shift_hold <= 0;
    else if (gu_fire) shift_hold <= gu_requant_shift;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_slot_hold <= 0;
    else if (clear) gu_slot_hold <= 0;
    else if (gu_fire) gu_slot_hold <= gu_slot;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_slot_hold <= 0;
    else if (clear) down_slot_hold <= 0;
    else if (gu_fire) down_slot_hold <= down_slot;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) row_index <= 0;
    else if (clear) row_index <= 0;
    else if (begin_operation || operation_finished) row_index <= 0;
    else if (issue && issue_last) row_index <= row_index + 1;
    else if (phase == NONLINEAR && row_index != F_TILE-1) row_index <= row_index + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) chunk_index <= 0;
    else if (clear) chunk_index <= 0;
    else if (begin_operation || (issue && issue_last)) chunk_index <= 0;
    else if (issue) chunk_index <= chunk_index + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) accumulator <= 0;
    else if (clear) accumulator <= 0;
    else if (begin_operation) accumulator <= 0;
    else if (valid_pipe[5]) accumulator <= first_pipe[5] ? 32'(dot_sum) : accumulator + 32'(dot_sum);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) partial_y <= 0;
    else if (clear) partial_y <= 0;
    else if (row_store && down_mode) partial_y[row_pipe[6]*32 +: 32] <= final_sum;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) router_logits <= 0;
    else if (clear) router_logits <= 0;
    else if (row_store && router_mode) router_logits[row_pipe[6]*32 +: 32] <= final_sum;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_done <= 0;
    else if (clear) gu_done <= 0;
    else gu_done <= operation_finished && !down_mode && !router_mode;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_done <= 0;
    else if (clear) down_done <= 0;
    else down_done <= operation_finished && down_mode;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) router_done <= 0;
    else if (clear) router_done <= 0;
    else router_done <= operation_finished && router_mode;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) nonlinear_done <= 0;
    else if (clear || gu_fire) nonlinear_done <= 0;
    else if (phase == NONLINEAR && row_index == F_TILE-1) nonlinear_done <= 1;
  end
endmodule
