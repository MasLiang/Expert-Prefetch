`timescale 1ns/1ps
/** 独立数学参考：32-input signed int8 Router；顺序 RAM 加载与错误恢复。 */
module tb_router_gemv;
  logic clk=0;
  always #5 clk=~clk;
  logic rst_n=0, start=0, reload=0, valid=0, last=0;
  logic ready, loaded, load_error, done, read_valid;
  logic [31:0] index=0, offset;
  logic [127:0] data=0, read_data;
  logic [255:0] activation=0;
  logic [127:0] logits;
  integer expected [0:3];
  integer row, lane, test_id;
  router_weight_store #(
    .H(32),
    .NUM_EXPERTS(4)
  ) weights (
    .clk(clk),
    .rst_n(rst_n),
    .host_enable(!backend.router_mode || done),
    .reload_valid(reload),
    .reload_ready(),
    .reload_layer(3'd0),
    .write_valid(valid),
    .write_ready(ready),
    .write_word_index(index),
    .write_data(data),
    .write_last(last),
    .loaded(loaded),
    .loaded_layer(),
    .load_error(load_error),
    .read_valid(read_valid),
    .read_offset(offset),
    .read_data(read_data)
  );
  int8_swiglu_tile_backend #(
    .H(32),
    .F_TILE(16),
    .NUM_ROUTER_EXPERTS(4)
  ) backend (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .router_start(start),
    .router_done(done),
    .router_read_valid(read_valid),
    .router_read_offset(offset),
    .router_read_data(read_data),
    .router_logits(logits),
    .gu_start(1'b0),
    .down_start(1'b0),
    .activation(activation),
    .gu_requant_shift(5'd0),
    .gu_slot(16'd0),
    .down_slot(16'd0),
    .read_valid(),
    .read_slot(),
    .read_offset(),
    .read_data(128'd0),
    .gu_done(),
    .nonlinear_done(),
    .down_done(),
    .partial_y()
  );
  initial begin
    repeat(3) @(negedge clk);
    rst_n=1;
    // 错序 word 不应使未初始化 RAM 成为 loaded。
    index=1; valid=1; @(negedge clk); valid=0;
    if (!load_error || loaded) $fatal(1,"bad load accepted");
    for (test_id=0; test_id<12; test_id=test_id+1) begin
      // 重置 backend 来退休前一事务；Router RAM 只通过显式 reload 更新。
      rst_n=0; @(negedge clk); rst_n=1;
      reload=1; @(negedge clk); reload=0;
      for (lane=0;lane<32;lane=lane+1)
        activation[lane*8 +: 8]=8'((lane*7+test_id*11)%256-128);
      for (row=0;row<4;row=row+1) expected[row]=0;
      for (index=0; index<8; index=index+1) begin
        for (lane=0;lane<16;lane=lane+1) begin
          data[lane*8 +: 8]=8'((int'(index)*19+lane*13+test_id*5)%256-128);
          expected[index/2]=expected[index/2]+
            $signed(data[lane*8 +: 8])*$signed(activation[((index%2)*16+lane)*8 +: 8]);
        end
        valid=1; last=index==7; @(negedge clk);
      end
      valid=0; last=0;
      if (!loaded || load_error) $fatal(1,"weights not resident");
      start=1; @(negedge clk); start=0;
      wait(done); @(negedge clk);
      for(row=0;row<4;row=row+1)
        if ($signed(logits[row*32 +: 32]) != expected[row])
          $fatal(1,"Router GEMV test=%0d expert=%0d got=%0d expected=%0d",
            test_id,row,$signed(logits[row*32 +: 32]),expected[row]);
    end
    $display("PASS: 12 dense signed Router GEMVs, 48 independent dot products, load recovery");
    $finish;
  end
  initial begin #1000000; $fatal(1,"timeout"); end
endmodule
