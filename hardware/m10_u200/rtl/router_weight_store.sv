`timescale 1ns/1ps
/** 一层常驻 Router int8 权重，output-major W[E,H]。host 顺序加载 128-bit words。
 * 仅 idle 时允许 reload/write；last 必须严格对应最后一个 word，避免未初始化读取。
 */
module router_weight_store #(
  parameter integer H = 16,
  parameter integer NUM_EXPERTS = 64
) (
  input logic clk,
  input logic rst_n,
  input logic host_enable,
  input logic reload_valid,
  output logic reload_ready,
  input logic [2:0] reload_layer,
  input logic write_valid,
  output logic write_ready,
  input logic [31:0] write_word_index,
  input logic [127:0] write_data,
  input logic write_last,
  output logic loaded,
  output logic [2:0] loaded_layer,
  output logic load_error,
  input logic read_valid,
  input logic [31:0] read_offset,
  output logic [127:0] read_data
);
  localparam integer WORDS = H*NUM_EXPERTS/16;
  (* ram_style = "block" *) logic [127:0] memory [0:WORDS-1];
  logic [31:0] next_word;
  wire accepted = write_valid && write_ready;
  wire legal_word = write_word_index == next_word && write_word_index < WORDS &&
    write_last == (write_word_index == WORDS-1);
  assign reload_ready = host_enable;
  assign write_ready = host_enable && !loaded && !load_error && !reload_valid;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) next_word <= 0;
    else if (reload_valid && reload_ready) next_word <= 0;
    else if (accepted && legal_word) next_word <= next_word + 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) loaded <= 0;
    else if (reload_valid && reload_ready) loaded <= 0;
    else if (accepted && legal_word && write_last) loaded <= 1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) loaded_layer <= 0;
    else if (reload_valid && reload_ready) loaded_layer <= reload_layer;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) load_error <= 0;
    else if (reload_valid && reload_ready) load_error <= 0;
    else if (accepted && !legal_word) load_error <= 1;
  end
  always_ff @(posedge clk) begin
    if (accepted && legal_word) memory[int'(write_word_index)] <= write_data;
  end
  always_ff @(posedge clk) begin
    if (read_valid && loaded && read_offset[3:0] == 0 && read_offset/16 < WORDS)
      read_data <= memory[int'(read_offset/16)];
  end
  initial begin
    if (H < 16 || H%16 != 0 || NUM_EXPERTS < 1) $fatal(1, "invalid Router weight geometry");
  end
endmodule
