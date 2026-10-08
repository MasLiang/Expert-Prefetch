`timescale 1ns/1ps
/** 独立 dense signed int8 参考：ReLU、饱和、配置 shift 和两矩阵 payload。 */
module tb_switch_relu_backend;
  logic clk=0;
  always #5 clk=~clk;
  logic rst_n=0;
  logic gu_start=0;
  logic down_start=0;
  logic [127:0] activation=0;
  logic [4:0] shift=0;
  wire read_valid, gu_done, nonlinear_done, down_done;
  wire [15:0] read_slot;
  wire [31:0] read_offset;
  logic [127:0] read_data;
  wire [511:0] partial_y;
  integer reads=0;
  integer transactions=0;

  function automatic integer weight(input integer slot, input integer row, input integer column);
    if(slot==1) begin
      case(row%4)
        0: return 127;
        1: return -128;
        2: return column%2==0 ? -3 : 2;
        default: return column%2==0 ? 2 : -3;
      endcase
    end
    return (row+column)%2==0 ? -128 : 127;
  endfunction
  function automatic logic [127:0] payload(input integer slot, input integer offset);
    logic [127:0] value;
    for(integer lane=0;lane<16;lane=lane+1)
      value[lane*8 +: 8]=8'(weight(slot,offset/16,lane));
    return value;
  endfunction
  always_ff @(posedge clk) begin
    if(read_valid) read_data<=payload(int'(read_slot),int'(read_offset));
  end
  always_ff @(posedge clk) begin
    if(gu_start || down_start) reads<=0;
    else if(read_valid) reads<=reads+1;
  end
  always @(posedge clk) begin
    if(read_valid && (read_offset!=32'(reads*16) || read_offset>=256))
      $fatal(1,"Switch consumed extra matrix or reordered payload slot=%0d offset=%0d",read_slot,read_offset);
  end
  int8_swiglu_tile_backend #(
    .SWITCH_RELU(1'b1),
    .NUM_ROUTER_EXPERTS(4)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .gu_start(gu_start),
    .down_start(down_start),
    .router_start(1'b0),
    .router_done(),
    .router_read_valid(),
    .router_read_offset(),
    .router_read_data(128'd0),
    .router_logits(),
    .activation(activation),
    .gu_requant_shift(shift),
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
  task automatic check(input integer mode, input integer quant_shift);
    integer x[0:15];
    integer intermediate[0:15];
    integer sum, expected;
    @(negedge clk);
    for(integer column=0;column<16;column=column+1) begin
      x[column]=mode==0 ? 127 : mode==1 ? -128 : mode==2 ? (column%2==0 ? -7 : 9) : 0;
      activation[column*8 +: 8]=8'(x[column]);
    end
    shift=5'(quant_shift);
    for(integer row=0;row<16;row=row+1) begin
      sum=0;
      for(integer column=0;column<16;column=column+1) sum=sum+x[column]*weight(1,row,column);
      sum=sum>>>quant_shift;
      intermediate[row]=sum<0 ? 0 : sum>127 ? 127 : sum;
    end
    gu_start=1;
    @(negedge clk); gu_start=0; activation='1; shift=0;
    wait(gu_done);
    if(reads!=16) $fatal(1,"Up payload count %0d",reads);
    wait(nonlinear_done);
    @(negedge clk); down_start=1;
    @(negedge clk); down_start=0;
    wait(down_done);
    if(reads!=16) $fatal(1,"Down payload count %0d",reads);
    @(negedge clk);
    for(integer row=0;row<16;row=row+1) begin
      expected=0;
      for(integer column=0;column<16;column=column+1)
        expected=expected+intermediate[column]*weight(2,row,column);
      if($signed(partial_y[row*32 +: 32])!==expected)
        $fatal(1,"Switch mismatch mode=%0d shift=%0d row=%0d expected=%0d got=%0d",
          mode,quant_shift,row,expected,$signed(partial_y[row*32 +: 32]));
    end
    transactions=transactions+1;
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;
    for(integer mode=0;mode<4;mode=mode+1) begin
      check(mode,0);
      check(mode,3);
      check(mode,31);
    end
    if(transactions!=12) $fatal(1,"missing test");
    $display("PASS: Switch ReLU 12 dense int8 saturation/shift transactions and exact Up/Down payload");
    $finish;
  end
  initial begin #100000; $fatal(1,"Switch backend watchdog"); end
endmodule
