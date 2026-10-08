`timescale 1ns/1ps
/** 无前层权重屏障的2-layer lead、credit/backpressure、epoch隔离与abort sweep。 */
module tb_cross_layer_prefetch;
  import tile_prefetch_package::*;
  logic clk=0;
  always #2.5 clk=~clk;
  logic rst_n=0, clear_sequence=0, abort=0, current_begin=0;
  logic [2:0] current_layer=0, target=2;
  logic [15:0] epoch=3;
  logic begin_valid=0, begin_ready, candidate_valid=0, candidate_ready;
  logic [4:0] count=1, rank=0;
  logic [9:0] expert=1;
  logic candidate_last=1;
  logic logical_valid, logical_ready=0, context_ok, flush_valid, flush_ready=0, error;
  logic [15:0] flush_epoch;
  tile_descriptor_t gu, down, completion='0, saved;
  logic route_accept=0, route_last=0, dma_done=0;
  logic [2:0] route_layer=0;
  wire [3:0] weights_done;
  integer issued=0, flushes=0;
  layer_weight_barrier #(
    .LAYERS(4),
    .EXPERTS(4),
    .TILES(2)
  ) barrier (
    .clk(clk),
    .rst_n(rst_n),
    .clear_sequence(clear_sequence),
    .route_accept(route_accept),
    .route_layer(route_layer),
    .route_expert(10'd1),
    .route_last(route_last),
    .dma_done(dma_done),
    .dma_descriptor(completion),
    .weights_done(weights_done)
  );
  cross_layer_prefetch #(
    .LAYERS(4),
    .MAX_CANDIDATES(2),
    .EXPERTS(4),
    .TILES(2),
    .CHANNELS(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear_sequence(clear_sequence),
    .sequence_busy(1'b1),
    .abort(abort),
    .current_layer(current_layer),
    .current_begin(current_begin),
    .current_epoch(16'd1),
    .current_sequence(16'd9),
    .sequence_id(16'd9),
    .begin_valid(begin_valid),
    .begin_ready(begin_ready),
    .target_layer(target),
    .target_epoch(epoch),
    .target_sequence(16'd9),
    .candidate_count(count),
    .budget_bytes(32'd768),
    .channel_credit_bytes(32'd768),
    .gu_base_addr(64'h10000),
    .down_base_addr(64'h20000),
    .expert_stride_bytes(32'd4096),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(expert),
    .candidate_rank(rank),
    .candidate_confidence(12'd3000),
    .candidate_last(candidate_last),
    .logical_valid(logical_valid),
    .logical_ready(logical_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .current_context_ok(context_ok),
    .flush_valid(flush_valid),
    .flush_epoch(flush_epoch),
    .flush_ready(flush_ready),
    .error(error)
  );
  always @(posedge clk) begin
    if(logical_valid && logical_ready) issued=issued+1;
    if(flush_valid && flush_ready) flushes=flushes+1;
  end
  task automatic tick;
    @(posedge clk); #0.1;
  endtask
  task automatic submit(input logic [2:0] layer, input logic [15:0] tag);
    @(negedge clk); target=layer; epoch=tag; begin_valid=1;
    if(!begin_ready) #0.1;
    wait(begin_ready); tick();
    @(negedge clk); begin_valid=0; candidate_valid=1;
    wait(candidate_ready); tick();
    @(negedge clk); candidate_valid=0;
  endtask
  task automatic resolve(input logic [2:0] layer);
    @(negedge clk); route_layer=layer; route_accept=1; route_last=1;
    tick();
    @(negedge clk); route_accept=0; route_last=0;
  endtask
  task automatic complete(input logic [2:0] layer, input integer tile, input matrix_kind_t matrix, input integer expert_id);
    @(negedge clk);
    completion='0; completion.layer_id={13'd0,layer}; completion.expert_id=10'(expert_id);
    completion.tile_id=8'(tile); completion.matrix_kind=matrix; dma_done=1;
    tick();
    @(negedge clk); dma_done=0;
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;
    current_begin=1; tick();
    @(negedge clk); current_begin=0;
    submit(1,2);
    submit(2,3); // 前层 truth 尚未解析，两层 lead 也必须能参与预取。
    @(negedge clk); target=3; epoch=4; count=0; begin_valid=1;
    wait(begin_ready); tick();
    @(negedge clk); begin_valid=0; count=1;
    target=3; epoch=2; begin_valid=1; #0.1;
    if(begin_ready) $fatal(1,"duplicate epoch accepted");
    begin_valid=0;
    repeat(15) tick();
    if(!logical_valid || weights_done!=0) $fatal(1,"prefetch waited for predecessor truth/weights");
    resolve(0);
    complete(0,0,MATRIX_GU,1);
    complete(0,0,MATRIX_DOWN,1);
    complete(0,1,MATRIX_GU,1);
    complete(0,1,MATRIX_DOWN,2); // 错误expert不能满足真实权重屏障。
    if(weights_done[0] || !logical_valid) $fatal(1,"weight telemetry gated prefetch or classified false expert");
    complete(0,1,MATRIX_DOWN,1);
    wait(logical_valid);
    if(gu.layer_id!=1 || gu.epoch_id!=2 || gu.byte_count!=512 || down.byte_count!=256)
      $fatal(1,"wrong released target or byte geometry");
    saved=gu;
    repeat(10) begin tick(); if(!logical_valid || gu!=saved || issued!=0) $fatal(1,"backpressure lost plan"); end
    @(negedge clk); logical_ready=1;
    tick();
    @(negedge clk); logical_ready=0;
    repeat(30) tick();
    if(issued!=1) $fatal(1,"byte/channel credit exceeded");
    if(!logical_valid || gu.layer_id!=2 || weights_done[1])
      $fatal(1,"two-layer prefetch waited for unresolved intermediate layer");
    resolve(1);
    for(integer tile=0;tile<2;tile=tile+1) begin
      complete(1,tile,MATRIX_GU,1);
      complete(1,tile,MATRIX_DOWN,1);
    end
    wait(logical_valid);
    if(gu.layer_id!=2 || gu.epoch_id!=3) $fatal(1,"two-layer lead lost or wrong ordering");
    @(negedge clk); abort=1;
    tick();
    @(negedge clk); abort=0;
    wait(flush_valid);
    repeat(3) begin tick(); if(!flush_valid || flush_epoch!=1 || logical_valid) $fatal(1,"abort sweep backpressure"); end
    @(negedge clk); flush_ready=1;
    repeat(8) tick();
    if(flushes!=4 || flush_valid || logical_valid) $fatal(1,"missing cross-context abort cleanup");
    @(negedge clk); clear_sequence=1;
    tick();
    @(negedge clk); clear_sequence=0; count=2; candidate_last=0;
    submit(1,2);
    repeat(15) tick();
    if(logical_valid) $fatal(1,"incomplete prediction launched");
    rank=1; candidate_last=1; candidate_valid=1; #0.1;
    if(candidate_ready) $fatal(1,"duplicate candidate accepted");
    tick();
    if(!error) $fatal(1,"malformed candidate not reported");
    $display("PASS: barrier-free cross-layer prefetch before truth/weights, two-layer lead, complete prediction, stable OCM backpressure, credit, epoch and abort sweep");
    $finish;
  end
  initial begin
    #20000; $fatal(1,"cross-layer controller watchdog");
  end
endmodule
