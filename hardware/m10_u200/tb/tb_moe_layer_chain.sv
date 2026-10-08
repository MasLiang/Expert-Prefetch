`timescale 1ns/1ps
/** Controller独立contract：配置锁存、signed饱和、排空、abort和重新开始。 */
module tb_moe_layer_chain;
  logic clk=0;
  always #2.5 clk=~clk;
  logic rst_n=0, start=0, abort=0, accept=0, result_valid=0, drained=1;
  logic ready, start_ready, busy, done, flush;
  logic [2:0] layer;
  logic [4:0] shift=0;
  logic [63:0] result_value=0;
  logic [15:0] activation;
  moe_layer_chain #(
    .LAYER_COUNT(3),
    .H(2)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .start_ready(start_ready),
    .abort(abort),
    .layer_accept(accept),
    .layer_result_valid(result_valid),
    .layer_result(result_value),
    .transaction_drained(drained),
    .initial_activation(16'hf010),
    .output_requant_shift(shift),
    .activation(activation),
    .layer_ready(ready),
    .current_layer(layer),
    .busy(busy),
    .done(done),
    .context_flush(flush)
  );
  task automatic tick;
    @(posedge clk); #0.1;
  endtask
  task automatic launch;
    @(negedge clk); start=1;
    tick();
    @(negedge clk); start=0;
    if(!busy || !ready || layer!=0 || activation!=16'hf010) $fatal(1,"initial chain context");
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;
    launch();
    for(integer index=0;index<3;index=index+1) begin
      @(negedge clk); shift=2; accept=1;
      tick();
      @(negedge clk); accept=0; shift=31; // Must not replace accepted requant config.
      drained=0;
      result_value={32'h80000000,32'h7fffffff}; result_valid=1;
      tick();
      @(negedge clk); result_valid=0;
      repeat(5) begin
        tick();
        if(flush || ready || done || layer!=index || activation!=16'h807f) $fatal(1,"requant or premature drain");
      end
      @(negedge clk); drained=1;
      tick();
      if(!flush) $fatal(1,"missing context cleanup");
      tick();
      tick();
      if(index<2 && (!ready || layer!=index+1 || done)) $fatal(1,"next layer dependency");
      if(index==2 && (!done || busy || !start_ready)) $fatal(1,"final sequence boundary");
    end
    launch();
    @(negedge clk); accept=1;
    tick();
    @(negedge clk); accept=0; result_valid=1; abort=1;
    tick();
    if(done || flush || busy || ready) $fatal(1,"abort/result collision leaked advance");
    @(negedge clk); result_valid=0; abort=0; drained=0; start=1;
    tick();
    if(busy || start_ready) $fatal(1,"restart admitted while old DMA pending");
    @(negedge clk); start=0; drained=1;
    launch();
    $display("PASS: chain signed requant/config hold, delayed drain, final result boundary and abort/restart");
    $finish;
  end
  initial begin
    #20000; $fatal(1,"layer chain watchdog");
  end
endmodule
