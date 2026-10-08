`timescale 1ns/1ps
/** 独立 flat-array 参考模型验证 OCM occupancy、pair admission 与最低 free slot。 */
module tb_ocm_admission_tree #(
  parameter integer BANKS = 4,
  parameter integer SLOTS = 32,
  parameter bit PAIR_PRIMARY = 1'b0
);
  import tile_prefetch_package::*;
  localparam integer COUNT = BANKS*SLOTS;
  logic clk=0;
  logic rst_n=0;
  logic allocate_valid=0;
  logic allocate_ready;
  logic pair_ready;
  logic flush_valid=0;
  logic retire_valid=0;
  logic [15:0] retire_slot=0;
  logic [31:0] occupancy;
  tile_descriptor_t descriptor, pair_descriptor, primary_descriptor, allocated;
  logic model_used [0:COUNT-1];
  integer captured_slot;
  integer expected_occupancy;
  integer free_a;
  integer free_b;
  integer first_a;
  integer used_primary;
  always #5 clk=~clk;

  ocm_slot_ownership #(
    .NUM_OCM_BANKS(BANKS),
    .SLOTS_PER_BANK(SLOTS),
    .USE_PAIR_PRIMARY(PAIR_PRIMARY)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(allocate_valid),
    .allocate_descriptor(descriptor),
    .pair_check_descriptor(pair_descriptor),
    .pair_primary_descriptor(primary_descriptor),
    .allocate_ready(allocate_ready),
    .pair_reserve_ready(pair_ready),
    .allocated_descriptor(allocated),
    .allocated_slot_generation(),
    .dma_start_valid(1'b0),
    .dma_start_slot(16'b0),
    .dma_start_epoch(16'b0),
    .dma_start_slot_generation(8'b0),
    .dma_write_valid(1'b0),
    .dma_write_last(1'b0),
    .dma_write_slot(16'b0),
    .dma_write_epoch(16'b0),
    .dma_write_slot_generation(8'b0),
    .dma_write_accept(),
    .commit_valid(1'b0),
    .commit_slot(16'b0),
    .commit_epoch(16'b0),
    .commit_slot_generation(8'b0),
    .cancel_valid(1'b0),
    .cancel_slot(16'b0),
    .cancel_epoch(16'b0),
    .cancel_slot_generation(8'b0),
    .soft_flush_valid(flush_valid),
    .soft_flush_epoch(16'd1),
    .retire_cancelled_valid(retire_valid),
    .retire_cancelled_slot(retire_slot),
    .retire_cancelled_epoch(16'd1),
    .retire_cancelled_slot_generation(8'd1),
    .consumer_release_valid(1'b0),
    .consumer_release_slot(16'b0),
    .consumer_release_epoch(16'b0),
    .consumer_release_slot_generation(8'b0),
    .consumer_release2_valid(1'b0),
    .consumer_release2_slot(16'b0),
    .consumer_release2_epoch(16'b0),
    .consumer_release2_slot_generation(8'b0),
    .compute_start_valid(1'b0),
    .compute_start_slot(16'b0),
    .compute_start_epoch(16'b0),
    .compute_start_slot_generation(8'b0),
    .compute_start2_valid(1'b0),
    .compute_start2_slot(16'b0),
    .compute_start2_epoch(16'b0),
    .compute_start2_slot_generation(8'b0),
    .query_slot(16'b0),
    .query_state(),
    .query_epoch(),
    .query_slot_generation(),
    .query_quarantined(),
    .query_payload_complete(),
    .query2_slot(16'b0),
    .query2_state(),
    .query2_epoch(),
    .query2_slot_generation(),
    .query2_quarantined(),
    .query2_payload_complete(),
    .stale_event_count(),
    .discarded_dma_write_count(),
    .occupancy(occupancy)
  );

  /** 构造确定性散列到指定 bank 的合法 descriptor。 */
  function automatic tile_descriptor_t bank_descriptor(input integer bank);
    tile_descriptor_t value;
    value='0;
    value.epoch_id=1;
    value.expert_id=10'(bank ^ 1);
    value.matrix_kind=MATRIX_GU;
    value.request_class=REQUEST_SPECULATIVE;
    value.byte_count=64;
    return value;
  endfunction

  /** 每次状态改变后遍历 bank 组合，与串行参考扫描逐项比较。 */
  task automatic check_reference;
    expected_occupancy=0;
    for(integer slot=0;slot<COUNT;slot=slot+1)
      if(model_used[slot]) expected_occupancy=expected_occupancy+1;
    for(integer a=0;a<BANKS;a=a+1) begin
      for(integer p=0;p<BANKS;p=p+1) begin
        for(integer b=0;b<BANKS;b=b+1) begin
          descriptor=bank_descriptor(a);
          pair_descriptor=bank_descriptor(b);
          primary_descriptor=bank_descriptor(p);
          free_a=0;
          free_b=0;
          first_a=-1;
          used_primary=PAIR_PRIMARY ? p : a;
          for(integer local_slot=0;local_slot<SLOTS;local_slot=local_slot+1) begin
            if(!model_used[a*SLOTS+local_slot] && first_a<0) first_a=local_slot;
            if(!model_used[used_primary*SLOTS+local_slot]) free_a=free_a+1;
            if(!model_used[b*SLOTS+local_slot]) free_b=free_b+1;
          end
          #1;
          if(occupancy != 32'(expected_occupancy)) $fatal(1,"occupancy differs from flat reference");
          if(allocate_ready != (first_a>=0)) $fatal(1,"bank free admission mismatch");
          if(allocated.ocm_slot != 16'(a*SLOTS + ((first_a>=0) ? first_a : 0)))
            $fatal(1,"lowest-local-free index mismatch");
          if(pair_ready != ((used_primary==b) ? (free_a>=2) : (free_a>=1 && free_b>=1)))
            $fatal(1,"pair free threshold mismatch");
        end
      end
    end
  endtask

  initial begin
    descriptor=bank_descriptor(0);
    pair_descriptor=descriptor;
    primary_descriptor=descriptor;
    for(integer slot=0;slot<COUNT;slot=slot+1) model_used[slot]=0;
    #17;
    rst_n=1;
    check_reference();
    for(integer round=0;round<SLOTS;round=round+1) begin
      for(integer bank=0;bank<BANKS;bank=bank+1) begin
        @(negedge clk);
        descriptor=bank_descriptor(bank);
        allocate_valid=1;
        #1;
        captured_slot=32'(allocated.ocm_slot);
        if(!allocate_ready) $fatal(1,"reference free slot not admitted");
        @(posedge clk);
        model_used[captured_slot]=1;
        @(negedge clk);
        allocate_valid=0;
        check_reference();
      end
    end
    @(negedge clk);
    flush_valid=1;
    @(negedge clk);
    flush_valid=0;
    check_reference();
    for(integer slot=COUNT-1;slot>=0;slot=slot-1) begin
      @(negedge clk);
      retire_slot=16'(slot);
      retire_valid=1;
      @(posedge clk);
      model_used[slot]=0;
      @(negedge clk);
      retire_valid=0;
      check_reference();
    end
    $display("PASS: independent OCM admission/popcount reference banks=%0d slots=%0d primary=%0d",BANKS,SLOTS,PAIR_PRIMARY);
    $finish;
  end
endmodule
