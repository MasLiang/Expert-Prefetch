`timescale 1ns/1ps
/** Independent dense-matrix golden harness; input/expected files supplied by Python. */
module tb_swiglu_dense_golden #(
  parameter integer H = 16,
  parameter integer F_TILE = 16
);
  logic clk = 0;
  logic rst_n = 0;
  logic gu_start = 0;
  logic down_start = 0;
  logic [H*8-1:0] activation;
  logic [127:0] memory [0:(3*H*F_TILE/16)-1];
  logic [31:0] expected [0:H-1];
  logic read_valid;
  logic [15:0] read_slot;
  logic [31:0] read_offset;
  logic [127:0] read_data;
  logic nonlinear_done;
  logic down_done;
  logic [H*32-1:0] partial_y;
  string memory_path;
  string expected_path;
  always #5 clk = ~clk;
  always @(posedge clk) begin
    if (read_valid)
      read_data <= memory[(read_slot == 2 ? 2*H*F_TILE/16 : 0) + read_offset/16];
  end
  int8_swiglu_tile_backend #(
    .H(H),
    .F_TILE(F_TILE)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .gu_start(gu_start),
    .down_start(down_start),
    .activation(activation),
    .gu_requant_shift(5'd1),
    .gu_slot(16'd1),
    .down_slot(16'd2),
    .read_valid(read_valid),
    .read_slot(read_slot),
    .read_offset(read_offset),
    .read_data(read_data),
    .gu_done(),
    .nonlinear_done(nonlinear_done),
    .down_done(down_done),
    .partial_y(partial_y)
  );
  initial begin
    if (!$value$plusargs("MEM=%s", memory_path) || !$value$plusargs("EXPECTED=%s", expected_path) ||
        !$value$plusargs("ACT=%h", activation)) $fatal(1, "golden inputs missing");
    $readmemh(memory_path, memory);
    $readmemh(expected_path, expected);
    repeat (3) @(negedge clk);
    rst_n = 1;
    gu_start = 1;
    @(negedge clk);
    gu_start = 0;
    wait (nonlinear_done);
    @(negedge clk);
    down_start = 1;
    @(negedge clk);
    down_start = 0;
    wait (down_done);
    @(negedge clk);
    for (integer row = 0; row < H; row = row + 1)
      if (partial_y[row*32 +: 32] !== expected[row])
        $fatal(1, "dense numerical mismatch row=%0d expected=%h actual=%h", row, expected[row], partial_y[row*32 +: 32]);
    $display("PASS: dense numerical Python/RTL golden H=%0d F_TILE=%0d", H, F_TILE);
    $finish;
  end
  initial begin
    #100000;
    $fatal(1, "dense golden timeout");
  end
endmodule
