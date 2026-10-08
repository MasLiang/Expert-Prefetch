`timescale 1ns/1ps
/** GU ready 在最后 truth 前两拍、同拍、后两拍；只检查接口因果，不 force DUT。 */
module tb_boundary_readiness #(
  parameter integer READY_OFFSET=0
);
  import tile_prefetch_package::*;
  logic clk=0;
  always #2.5 clk=~clk;
  logic rst_n=0;
  logic capture=0;
  logic dma_ready=0;
  logic truth=0;
  tile_descriptor_t incoming;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  logic request;
  integer cycle=0;
  integer requests=0;
  integer gu_ready_cycle=-1;
  integer truth_cycle=-1;
  integer request_cycle=-1;
  always @(posedge clk) cycle<=cycle+1;
  always @(posedge clk) if(rst_n && request) requests<=requests+1;
  always @(posedge clk) if(rst_n && dma_ready) gu_ready_cycle<=cycle;
  always @(posedge clk) if(rst_n && truth) truth_cycle<=cycle;
  always @(posedge clk) if(rst_n && request) request_cycle<=cycle;
  always @(posedge clk) begin
    if(rst_n && request && (gu_ready_cycle<0 || truth_cycle<0 || gu.expert_id!=3 || down.ocm_slot!=1))
      $fatal(1,"compute before ready/truth/ownership");
  end
  tile_execution_scheduler #(
    .SLOT_COUNT(2),
    .TOP_K_MAX(1),
    .TILES_PER_EXPERT(1),
    .H(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .context_epoch(16'd1),
    .context_layer(3'd0),
    .flush_valid(1'b0),
    .flush_epoch(16'd0),
    .capture_valid(capture),
    .capture_descriptor(incoming),
    .capture_generation(8'd1),
    .dma_done(dma_ready),
    .dma_descriptor(incoming),
    .route_accept(truth),
    .route_expert(10'd3),
    .route_weight_q15(16'd32768),
    .route_last(1'b1),
    .compute_req_valid(request),
    .compute_req_ready(1'b1),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .gu_generation(),
    .down_generation(),
    .compute_done(1'b0),
    .gu_release_valid(1'b0),
    .partial_y(32'd0),
    .layer_done(),
    .layer_y()
  );
  initial begin
    incoming='0; incoming.expert_id=3; incoming.epoch_id=1;
    repeat(2) @(negedge clk);rst_n=1;
    capture=1;incoming.matrix_kind=MATRIX_GU;incoming.ocm_slot=0;
    @(negedge clk);incoming.matrix_kind=MATRIX_DOWN;incoming.ocm_slot=1;
    @(negedge clk);capture=0;incoming.matrix_kind=MATRIX_GU;incoming.ocm_slot=0;
    for(integer step=0;step<5;step=step+1) begin
      truth=step==2;dma_ready=step==2+READY_OFFSET;
      @(negedge clk);
    end
    truth=0;dma_ready=0;
    repeat(8) @(negedge clk);
    if(requests!=1 || gu_ready_cycle-truth_cycle!=READY_OFFSET || request_cycle<=gu_ready_cycle || request_cycle<=truth_cycle)
      $fatal(1,"readiness case missing/duplicated");
    $display("PASS: boundary readiness offset=%0d truth=%0d ready=%0d accept=%0d",READY_OFFSET,truth_cycle,gu_ready_cycle,request_cycle);
    $finish;
  end
  initial begin #10000;$fatal(1,"boundary readiness watchdog");end
endmodule
