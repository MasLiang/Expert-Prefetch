`timescale 1ns/1ps
/** 独立数值/协议参考：连续chunk、跨row、busy启动和全部流水年龄flush后重启。 */
module tb_mac_chunk_pipeline #(
  parameter integer H = 48,
  parameter integer F_TILE = 48,
  parameter bit SWITCH_RELU = 1'b0
);
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic gu_start = 0;
  logic down_start = 0;
  logic router_start = 0;
  logic [H*8-1:0] activation = 0;
  logic read_valid, router_read_valid, gu_done, nonlinear_done, down_done, router_done;
  logic [15:0] read_slot;
  logic [31:0] read_offset, router_read_offset;
  logic [127:0] read_data, router_read_data;
  logic [H*32-1:0] partial_y;
  logic [4*32-1:0] logits;
  integer reads = 0;
  integer router_reads = 0;
  integer previous_read_cycle = -1;
  integer previous_router_cycle = -1;
  integer cycle = 0;
  logic stream_open = 0;
  logic router_stream_open = 0;

  int8_swiglu_tile_backend #(
    .H(H),
    .F_TILE(F_TILE),
    .SWITCH_RELU(SWITCH_RELU),
    .NUM_ROUTER_EXPERTS(4)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .gu_start(gu_start),
    .down_start(down_start),
    .router_start(router_start),
    .router_done(router_done),
    .router_read_valid(router_read_valid),
    .router_read_offset(router_read_offset),
    .router_read_data(router_read_data),
    .router_logits(logits),
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

  function automatic integer weight(input integer row, input integer column);
    return (row+column)%2 == 0 ? -128 : 127;
  endfunction
  function automatic logic [127:0] payload(input integer slot, input integer offset);
    logic [127:0] word;
    integer row, column;
    word = 0;
    row = offset/(slot == 1 ? H : F_TILE);
    for (integer lane = 0; lane < 16; lane = lane + 1) begin
      column = offset%(slot == 1 ? H : F_TILE)+lane;
      if (slot == 1) word[lane*8 +: 8] = column == (row%F_TILE)%H ? 8'd1 : 8'd0;
      else word[lane*8 +: 8] = 8'(weight(row,column));
    end
    return word;
  endfunction
  always_ff @(posedge clk) begin
    if (read_valid) read_data <= payload(int'(read_slot),int'(read_offset));
  end
  always_ff @(posedge clk) begin
    if (router_read_valid) router_read_data <= {16{router_read_offset/H%2 == 0 ? 8'h80 : 8'h7f}};
  end
  always_ff @(posedge clk) begin
    cycle <= cycle+1;
  end
  always_ff @(posedge clk) begin
    if (clear || gu_start || down_start) reads <= 0;
    else if (read_valid) reads <= reads+1;
  end
  always_ff @(posedge clk) begin
    if (clear || gu_start || down_start) stream_open <= 0;
    else if (read_valid) stream_open <= 1;
  end
  always_ff @(posedge clk) begin
    if (read_valid) previous_read_cycle <= cycle;
  end
  always_ff @(posedge clk) begin
    if (clear || router_start) router_reads <= 0;
    else if (router_read_valid) router_reads <= router_reads+1;
  end
  always_ff @(posedge clk) begin
    if (clear || router_start) router_stream_open <= 0;
    else if (router_read_valid) router_stream_open <= 1;
  end
  always_ff @(posedge clk) begin
    if (router_read_valid) previous_router_cycle <= cycle;
  end
  always @(posedge clk) begin
    if (read_valid && !clear && !gu_start && !down_start) begin
      if (router_read_valid || read_offset != reads*16 ||
          (stream_open && cycle != previous_read_cycle+1))
        $fatal(1,"payload stream ordering/II failed offset=%0d reads=%0d",read_offset,reads);
    end
    if (router_read_valid && !clear && !router_start) begin
      if (router_read_offset != router_reads*16 ||
          (router_stream_open && cycle != previous_router_cycle+1))
        $fatal(1,"Router stream ordering/II failed");
    end
  end
  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask
  task automatic start_gu;
    @(negedge clk);
    for (integer lane = 0; lane < H; lane = lane+1)
      activation[lane*8 +: 8] = lane%3 == 0 ? 8'd16 : lane%3 == 1 ? 8'hf0 : 8'd0;
    gu_start = 1;
    tick();
    @(negedge clk);
    gu_start = 0;
    activation = '1; // The backend must use the accepted activation, not this update.
  endtask
  task automatic start_down;
    wait(nonlinear_done);
    @(negedge clk);
    down_start = 1;
    tick();
    @(negedge clk);
    down_start = 0;
  endtask
  task automatic check_transaction;
    integer expected, n;
    start_gu();
    wait(gu_done);
    if (reads != (SWITCH_RELU ? 1 : 2)*H*F_TILE/16) $fatal(1,"GU missing payload");
    start_down();
    wait(down_done);
    if (reads != H*F_TILE/16) $fatal(1,"DOWN missing payload");
    for (integer row = 0; row < H; row = row+1) begin
      expected = 0;
      for (integer column = 0; column < F_TILE; column = column+1) begin
        n = SWITCH_RELU ? ((column%H)%3 == 0 ? 16 : 0) :
          ((column%H)%3 == 0 ? 12 : (column%H)%3 == 1 ? 4 : 0);
        expected = expected+n*weight(row,column);
      end
      if ($signed(partial_y[row*32 +: 32]) !== expected)
        $fatal(1,"chunk accumulation mismatch row=%0d got=%0d expected=%0d",row,$signed(partial_y[row*32 +: 32]),expected);
    end
    @(negedge clk);
  endtask
  task automatic check_router;
    @(negedge clk);
    activation = {H{8'h80}};
    router_start = 1;
    tick();
    @(negedge clk);
    router_start = 0;
    // Simultaneous GU start while Router is busy must not overwrite its context.
    activation = 0;
    gu_start = 1;
    tick();
    @(negedge clk);
    gu_start = 0;
    wait(router_done);
    if (router_reads != 4*H/16) $fatal(1,"Router missing payload");
    for (integer row = 0; row < 4; row = row+1)
      if ($signed(logits[row*32 +: 32]) !== (row%2 == 0 ? 16384*H : -16256*H))
        $fatal(1,"signed int8 extreme Router mismatch");
    @(negedge clk);
  endtask
  initial begin
    repeat(3) @(negedge clk);
    rst_n = 1;
    check_router();
    check_transaction();
    for (integer mode = 0; mode < 3; mode = mode+1) begin
      for (integer scenario = 0; scenario < 11; scenario = scenario+1) begin
        integer age, chunks;
        chunks = mode == 0 ? 4*H/16 : mode == 1 ? (SWITCH_RELU ? 1 : 2)*H*F_TILE/16 : H*F_TILE/16;
        age = scenario < 8 ? scenario : scenario == 8 ? chunks-1 : scenario == 9 ? chunks+5 : chunks+7;
        if (mode == 0) begin
          @(negedge clk);
          activation = {H{8'h80}};
          router_start = 1;
          tick();
          @(negedge clk);
          router_start = 0;
        end else begin
          start_gu();
          if (mode == 2) start_down();
        end
        repeat(age) tick();
        @(negedge clk);
        clear = 1;
        tick();
        if (read_valid || router_read_valid || gu_done || nonlinear_done || down_done || router_done)
          $fatal(1,"flush leaked stage/read mode=%0d age=%0d",mode,age);
        @(negedge clk);
        clear = 0;
        repeat(10) begin
          tick();
          if (gu_done || nonlinear_done || down_done || router_done || read_valid || router_read_valid)
            $fatal(1,"stale pipeline completed after flush");
        end
        check_router();
        check_transaction();
      end
    end
    $display("PASS: MAC II=1 H=%0d F=%0d row/chunk accumulation, int8 extremes, context hold and 33 fill/drain/finish flush/restarts",H,F_TILE);
    $finish;
  end
  initial begin
    #500000;
    $fatal(1,"chunk pipeline watchdog");
  end
endmodule
