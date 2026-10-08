`timescale 1ns/1ps
module tb_descriptor_manager;
  import tile_prefetch_package::*;
  logic clk=0,rst_n=0,logical_valid,logical_ready,live_pair,live_spec;
  tile_descriptor_t gu_in,down_in,gu_out,down_out;
  logic gu_slot_ready,down_slot_ready,table_ready,queue_ready,promotion_ready;
  logic reserve_valid,promotion_valid,gu_valid,gu_ready,down_valid,down_ready;
  always #5 clk=~clk;
  descriptor_manager dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .logical_valid(logical_valid),
    .logical_gu_descriptor(gu_in),
    .logical_down_descriptor(down_in),
    .existing_pair_live(live_pair),
    .existing_pair_is_speculative(live_spec),
    .logical_ready(logical_ready),
    .gu_slot_reserve_ready(gu_slot_ready),
    .down_slot_reserve_ready(down_slot_ready),
    .descriptor_pair_reserve_ready(table_ready),
    .queue_pair_reserve_ready(queue_ready),
    .promotion_ready(promotion_ready),
    .reserve_pair_valid(reserve_valid),
    .promotion_valid(promotion_valid),
    .gu_valid(gu_valid),
    .gu_descriptor(gu_out),
    .gu_ready(gu_ready),
    .down_valid(down_valid),
    .down_descriptor(down_out),
    .down_ready);
  function automatic tile_descriptor_t desc(input matrix_kind_t kind,input request_class_t cls);
    tile_descriptor_t x; begin x='0;
      x.epoch_id=2;
      x.expert_id=7;
      x.tile_id=3;
      x.matrix_kind=kind;
      x.byte_count=128;
      x.request_class=cls;
      desc=x; end
  endfunction
  initial begin
    logical_valid=0;
      gu_in=desc(MATRIX_GU,REQUEST_SPECULATIVE);
      down_in=desc(MATRIX_DOWN,REQUEST_SPECULATIVE);
    live_pair=0;
      live_spec=0;
      gu_slot_ready=1;
      down_slot_ready=0;
      table_ready=1;
      queue_ready=1;
      promotion_ready=1;
      gu_ready=0;
      down_ready=0;
    #17 rst_n=1;
    /* A missing DOWN slot blocks the whole logical allocation. */
    @(negedge clk);
      logical_valid=1;
    #1; if(logical_ready || reserve_valid || gu_valid || down_valid) $fatal(1,"partial reservation escaped admission");
    down_slot_ready=1; #1; if(!logical_ready || !reserve_valid) $fatal(1,"atomic reservation not offered");
    @(negedge clk);
      logical_valid=0;
    #1; if(!gu_valid || gu_out.matrix_kind!=MATRIX_GU) $fatal(1,"GU was not emitted first");
    @(negedge clk);
      gu_ready=1;
    @(negedge clk);
      gu_ready=0;
    #1; if(!down_valid || down_out.matrix_kind!=MATRIX_DOWN) $fatal(1,"DOWN did not follow GU");
    @(negedge clk);
      down_ready=1;
    @(negedge clk);
      down_ready=0;
    /* Demand on a live speculative pair only promotes, and never emits a copy. */
    live_pair=1;
      live_spec=1;
      gu_in=desc(MATRIX_GU,REQUEST_DEMAND);
      down_in=desc(MATRIX_DOWN,REQUEST_DEMAND);
    @(negedge clk);
      logical_valid=1; #1;
    if(!logical_ready || !promotion_valid || reserve_valid || gu_valid || down_valid) $fatal(1,"promotion duplicated a live pair");
    @(negedge clk);
      logical_valid=0;
    $display("PASS: atomic GU/DOWN descriptor admission and promotion");
    $finish;
  end
endmodule
