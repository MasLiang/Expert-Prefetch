`timescale 1ns/1ps
/** Top-K / selected softmax 的独立预计算参考，包含 signed 极值与背压。 */
module tb_router_topk_softmax;
  logic clk=0;
  always #5 clk=~clk;
  logic rst_n=0, start=0, flush=0, begin_ready=0, expert_ready=0;
  logic begin_valid, expert_valid, busy, last;
  logic [191:0] logits;
  logic [4:0] top_k, shift;
  logic [9:0] id;
  logic [15:0] weight;
  integer expected_ids [0:7], expected_weights [0:7];
  integer i, total;
  router_topk_softmax #(
    .NUM_EXPERTS(6)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .flush(flush),
    .start(start),
    .logits(logits),
    .top_k(top_k),
    .logit_shift(shift),
    .busy(busy),
    .begin_valid(begin_valid),
    .begin_ready(begin_ready),
    .expert_valid(expert_valid),
    .expert_ready(expert_ready),
    .expert_id(id),
    .weight_q15(weight),
    .last(last)
  );
  task automatic run_case(input integer case_id);
    begin
      start=1; @(negedge clk); start=0;
      wait(begin_valid); @(negedge clk);
      repeat(3) begin @(negedge clk); if(!begin_valid || expert_valid) $fatal(1,"begin backpressure"); end
      begin_ready=1; @(negedge clk); begin_ready=0;
      total=0;
      for(i=0;i<int'(top_k);i=i+1) begin
        repeat(2) begin
          @(negedge clk);
          if(!expert_valid || int'(id)!=expected_ids[i] || int'(weight)!=expected_weights[i] ||
            last!=(i==int'(top_k)-1)) $fatal(1,"softmax case=%0d index=%0d got=(%0d,%0d) expected=(%0d,%0d)",
              case_id,i,id,weight,expected_ids[i],expected_weights[i]);
        end
        total=total+int'(weight);
        expert_ready=1; @(negedge clk); expert_ready=0;
      end
      if(total!=32768 || busy) $fatal(1,"normalization/retirement");
    end
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;

    logits={32'sd32,-32'sd9,32'sd16,32'sd0,32'sd16,-32'sd128};
    top_k=5'd3; shift=5'd0;
    expected_ids[0]=5; expected_weights[0]=18878;
    expected_ids[1]=1; expected_weights[1]=6944;
    expected_ids[2]=3; expected_weights[2]=6946;
    run_case(0);

    logits={32'sd32,-32'sd9,32'sd16,32'sd0,32'sd16,-32'sd128};
    top_k=5'd1; shift=5'd0;
    expected_ids[0]=5; expected_weights[0]=32768;
    run_case(1);

    logits={32'sd0,32'sd80,-32'sd8,32'sd0,32'sd2147483647,-32'sd2147483648};
    top_k=5'd6; shift=5'd0;
    expected_ids[0]=1; expected_weights[0]=32768;
    expected_ids[1]=4; expected_weights[1]=0;
    expected_ids[2]=2; expected_weights[2]=0;
    expected_ids[3]=5; expected_weights[3]=0;
    expected_ids[4]=3; expected_weights[4]=0;
    expected_ids[5]=0; expected_weights[5]=0;
    run_case(2);

    logits={32'sd32,32'sd0,32'sd64,32'sd128,-32'sd256,32'sd256};
    top_k=5'd4; shift=5'd3;
    expected_ids[0]=0; expected_weights[0]=18567;
    expected_ids[1]=2; expected_weights[1]=6830;
    expected_ids[2]=3; expected_weights[2]=4143;
    expected_ids[3]=5; expected_weights[3]=3228;
    run_case(3);

    logits={-32'sd16,-32'sd16,-32'sd16,-32'sd16,-32'sd16,-32'sd16};
    top_k=5'd6; shift=5'd0;
    expected_ids[0]=0; expected_weights[0]=5461;
    expected_ids[1]=1; expected_weights[1]=5461;
    expected_ids[2]=2; expected_weights[2]=5461;
    expected_ids[3]=3; expected_weights[3]=5461;
    expected_ids[4]=4; expected_weights[4]=5461;
    expected_ids[5]=5; expected_weights[5]=5463;
    run_case(4);
    top_k=0; start=1; @(negedge clk); start=0; if(busy) $fatal(1,"invalid K");
    top_k=2; start=1; @(negedge clk); start=0;
    repeat(4) @(negedge clk); flush=1; @(negedge clk); flush=0;
    if(busy || begin_valid || expert_valid) $fatal(1,"flush");
    $display("PASS: Top-K signed ties, configurable quantization, softmax Q15, stalls and flush");
    $finish;
  end
  initial begin #1000000; $fatal(1,"timeout"); end
endmodule
