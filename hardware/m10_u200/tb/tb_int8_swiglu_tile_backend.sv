`timescale 1ns/1ps
/** Identity G/U/DOWN matrices verify real int8 MAC, Q4 SiLU and signed arithmetic. */
module tb_int8_swiglu_tile_backend;
  logic clk = 0;
  logic rst_n = 0;
  logic gu_start = 0;
  logic down_start = 0;
  logic [127:0] activation;
  logic read_valid;
  logic [15:0] read_slot;
  logic [31:0] read_offset;
  logic [127:0] read_data;
  logic gu_done;
  logic nonlinear_done;
  logic down_done;
  logic [511:0] partial_y;
  always #5 clk = ~clk;

  int8_swiglu_tile_backend dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .gu_start(gu_start),
    .down_start(down_start),
    .activation(activation),
    .gu_requant_shift(5'd0),
    .gu_slot(16'd1),
    .down_slot(16'd2),
    .read_valid(read_valid),
    .read_slot(read_slot),
    .read_offset(read_offset),
    .read_data(read_data),
    .gu_done(gu_done),
    .nonlinear_done(nonlinear_done),
    .down_done(down_done),
    .partial_y(partial_y)
  );

  always @(posedge clk) begin
    if (read_valid) read_data <= 128'd1 << (((read_offset/16)%16)*8);
  end

  task automatic check_identity(input logic [7:0] x, input integer expected);
    @(negedge clk);
    activation = {16{x}};
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
    for (integer row = 0; row < 16; row = row + 1) begin
      if ($signed(partial_y[row*32 +: 32]) != expected)
        $fatal(1, "row %0d expected %0d actual %0d", row, expected, $signed(partial_y[row*32 +: 32]));
    end
  endtask

  initial begin
    activation = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    check_identity(8'd16, 12);
    check_identity(8'hf0, 4);
    check_identity(8'd0, 0);
    $display("PASS: real signed int8 SwiGLU identity and negative-input numerical tests");
    $finish;
  end
  initial begin
    // 三次事务各约 400 cycles；保留有界 watchdog，覆盖新增 MAC 流水级。
    #20000;
    $fatal(1, "numeric backend timeout");
  end
endmodule
