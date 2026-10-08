`timescale 1ns/1ps
module tb_ocm_slot_ownership;
  import tile_prefetch_package::*;
  logic clk=0, rst_n=0;
  logic allocate_valid, allocate_ready;
  tile_descriptor_t allocate_descriptor, allocated_descriptor;
  logic [7:0] allocated_generation;
  logic dma_start_valid, dma_write_valid, dma_write_last, soft_flush_valid;
  logic retire_cancelled_valid, consumer_release_valid, write_accept;
  logic commit_valid, compute_start_valid;
  logic cancel_valid, compute_start2_valid, consumer_release2_valid;
  logic [15:0] commit_slot, commit_epoch, compute_start_slot, compute_start_epoch;
  logic [7:0] commit_generation, compute_start_generation;
  logic [15:0] dma_start_slot, dma_start_epoch, dma_write_slot, dma_write_epoch;
  logic [7:0] dma_start_generation, dma_write_generation;
  logic [15:0] flush_epoch, retire_slot, retire_epoch, release_slot, release_epoch, query_slot;
  logic [7:0] retire_generation, release_generation, query_generation;
  logic [15:0] captured_slot;
  logic [7:0] captured_generation;
  logic [15:0] query_epoch;
  logic query_quarantined; metadata_state_t query_state;
  metadata_state_t query2_state;
  logic [15:0] query2_epoch;
  logic [7:0] query2_generation;
  logic query2_quarantined;
  logic [31:0] stale_count, discarded_count;
  always #5 clk=~clk;

  ocm_slot_ownership #(.NUM_OCM_BANKS(4),
    .SLOTS_PER_BANK(4)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(allocate_valid),
    .allocate_descriptor(allocate_descriptor),
    .allocate_ready(allocate_ready),
    .allocated_descriptor(allocated_descriptor),
    .allocated_slot_generation(allocated_generation),
    .dma_start_valid(dma_start_valid),
    .dma_start_slot(dma_start_slot),
    .dma_start_epoch(dma_start_epoch),
    .dma_start_slot_generation(dma_start_generation),
    .dma_write_valid(dma_write_valid),
    .dma_write_last(dma_write_last),
    .dma_write_slot(dma_write_slot),
    .dma_write_epoch(dma_write_epoch),
    .dma_write_slot_generation(dma_write_generation),
    .dma_write_accept(write_accept),
    .commit_valid(commit_valid),
    .commit_slot(commit_slot),
    .commit_epoch(commit_epoch),
    .commit_slot_generation(commit_generation),
    .cancel_valid(cancel_valid),
    .cancel_slot(commit_slot),
    .cancel_epoch(commit_epoch),
    .cancel_slot_generation(commit_generation),
    .compute_start_valid(compute_start_valid),
    .compute_start_slot(compute_start_slot),
    .compute_start_epoch(compute_start_epoch),
    .compute_start_slot_generation(compute_start_generation),
    .compute_start2_valid(compute_start2_valid),
    .compute_start2_slot(compute_start_slot),
    .compute_start2_epoch(compute_start_epoch),
    .compute_start2_slot_generation(compute_start_generation),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(flush_epoch),
    .retire_cancelled_valid(retire_cancelled_valid),
    .retire_cancelled_slot(retire_slot),
    .retire_cancelled_epoch(retire_epoch),
    .retire_cancelled_slot_generation(retire_generation),
    .consumer_release_valid(consumer_release_valid),
    .consumer_release_slot(release_slot),
    .consumer_release_epoch(release_epoch),
    .consumer_release_slot_generation(release_generation),
    .consumer_release2_valid(consumer_release2_valid),
    .consumer_release2_slot(release_slot),
    .consumer_release2_epoch(release_epoch),
    .consumer_release2_slot_generation(release_generation),
    .query_slot(query_slot),
    .query_state(query_state),
    .query_epoch(query_epoch),
    .query_slot_generation(query_generation),
    .query_quarantined(query_quarantined),
    .query2_slot(query_slot),
    .query2_state(query2_state),
    .query2_epoch(query2_epoch),
    .query2_slot_generation(query2_generation),
    .query2_quarantined(query2_quarantined),
    .stale_event_count(stale_count),
    .discarded_dma_write_count(discarded_count));

  function automatic tile_descriptor_t descriptor(input logic [15:0] epoch, input logic [9:0] expert, input request_class_t request_class);
    tile_descriptor_t x;
    begin
      x='0;
      x.epoch_id=epoch;
      x.layer_id=1;
      x.expert_id=expert;
      x.tile_id=2;
      x.matrix_kind=MATRIX_GU;
      x.byte_count=64;
      x.request_class=request_class;
      descriptor=x;
    end
  endfunction
  task automatic allocate(input tile_descriptor_t value);
    begin
      @(negedge clk); while(!allocate_ready) @(negedge clk);
      allocate_descriptor=value;
      allocate_valid=1; #1;
      captured_slot=allocated_descriptor.ocm_slot;
      captured_generation=allocated_generation;
      @(posedge clk);
      @(negedge clk);
      allocate_valid=0;
    end
  endtask
  initial begin
    allocate_valid=0;
      allocate_descriptor='0;
      dma_start_valid=0;
      dma_start_slot=0;
      dma_start_epoch=0;
      dma_start_generation=0;
    dma_write_valid=0;
      dma_write_last=0;
      dma_write_slot=0;
      dma_write_epoch=0;
      dma_write_generation=0;
      soft_flush_valid=0;
      flush_epoch=0;
    retire_cancelled_valid=0;
      retire_slot=0;
      retire_epoch=0;
      retire_generation=0;
      consumer_release_valid=0;
      release_slot=0;
      release_epoch=0;
      release_generation=0;
      query_slot=0;
    commit_valid=0;
    commit_slot=0;
    commit_epoch=0;
    commit_generation=0;
    compute_start_valid=0;
    compute_start2_valid=0;
    consumer_release2_valid=0;
    cancel_valid=0;
    compute_start_slot=0;
    compute_start_epoch=0;
    compute_start_generation=0;
    #17 rst_n=1;
    allocate(descriptor(16'd9,10'd3,REQUEST_SPECULATIVE));
    query_slot=captured_slot;
    #1;
    if(query_state != META_QUEUED_SPEC || query_epoch != 9 || query_generation != captured_generation) begin
      $display("DEBUG allocation slot=%0d state=%0d epoch=%0d qgen=%0d agen=%0d",query_slot,query_state,query_epoch,query_generation,captured_generation);
      $fatal(1,"allocation tag/state incorrect");
    end
    @(negedge clk);
      dma_start_slot=query_slot;
      dma_start_epoch=9;
      dma_start_generation=captured_generation;
      dma_start_valid=1;
    @(negedge clk);
      dma_start_valid=0;
    if(query_state != META_INFLIGHT_SPEC) $fatal(1,"DMA start did not acquire slot");
    @(negedge clk);
      flush_epoch=9;
      soft_flush_valid=1;
    @(negedge clk);
      soft_flush_valid=0;
    if(query_state != META_DROP_ON_COMPLETE || !query_quarantined) $fatal(1,"flush released or failed to quarantine inflight slot");
    /* Wrong generation must not write or free the reused address. */
    @(negedge clk);
      dma_write_slot=query_slot;
      dma_write_epoch=9;
      dma_write_generation=captured_generation-1'b1;
      dma_write_last=1;
      dma_write_valid=1;
    @(negedge clk);
      dma_write_valid=0;
    if(discarded_count != 1 || query_state != META_DROP_ON_COMPLETE) $fatal(1,"stale write guard failure");
    @(negedge clk);
      dma_write_generation=captured_generation;
      dma_write_valid=1;
    @(negedge clk);
      dma_write_valid=0;
    if(query_state != META_FREE || query_quarantined) $fatal(1,"last response did not safely release quarantine");
    /* epoch/expert are selected to hash to the same bank as the released slot. */
    allocate(descriptor(16'd10,10'd4,REQUEST_DEMAND));
    if(captured_slot != query_slot || captured_generation == 1) $fatal(1,"slot generation did not advance on reuse");
    /* Flush cancellation must be visible to tombstone retirement on this edge. */
    @(negedge clk);
    soft_flush_valid=1;
    flush_epoch=10;
    retire_cancelled_valid=1;
    retire_slot=captured_slot;
    retire_epoch=10;
    retire_generation=captured_generation;
    @(negedge clk);
    soft_flush_valid=0;
    retire_cancelled_valid=0;
    #1;
    if(query_state != META_FREE) $fatal(1,"same-cycle flush and queued retirement did not free slot");

    /* Flush overrides promotion of an inflight request without losing ownership. */
    allocate(descriptor(16'd11,10'd5,REQUEST_SPECULATIVE));
    query_slot=captured_slot;
    dma_start_slot=captured_slot;
    dma_start_epoch=11;
    dma_start_generation=captured_generation;
    dma_start_valid=1;
    @(negedge clk);
    dma_start_valid=0;
    commit_valid=1;
    commit_slot=captured_slot;
    commit_epoch=11;
    commit_generation=captured_generation;
    soft_flush_valid=1;
    flush_epoch=11;
    @(negedge clk);
    commit_valid=0;
    soft_flush_valid=0;
    #1;
    if(query_state != META_DROP_ON_COMPLETE || !query_quarantined)
      $fatal(1,"flush and promotion resurrected an inflight slot");
    dma_write_valid=1;
    dma_write_last=1;
    dma_write_slot=captured_slot;
    dma_write_epoch=11;
    dma_write_generation=captured_generation;
    soft_flush_valid=1;
    @(negedge clk);
    dma_write_valid=0;
    soft_flush_valid=0;
    #1;
    if(query_state != META_FREE || query_quarantined)
      $fatal(1,"same-cycle final DMA and flush failed safe retirement");

    /* Complete a demanded payload; flush must beat both compute-start ports. */
    allocate(descriptor(16'd12,10'd6,REQUEST_DEMAND));
    query_slot=captured_slot;
    dma_start_slot=captured_slot;
    dma_start_epoch=12;
    dma_start_generation=captured_generation;
    dma_start_valid=1;
    @(negedge clk);
    dma_start_valid=0;
    dma_write_valid=1;
    dma_write_slot=captured_slot;
    dma_write_epoch=12;
    dma_write_generation=captured_generation;
    @(negedge clk);
    dma_write_valid=0;
    #1;
    if(query_state != META_COMMITTED) $fatal(1,"demand completion did not commit");
    compute_start_valid=1;
    compute_start2_valid=1;
    compute_start_slot=captured_slot;
    compute_start_epoch=12;
    compute_start_generation=captured_generation;
    soft_flush_valid=1;
    flush_epoch=12;
    @(negedge clk);
    compute_start_valid=0;
    compute_start2_valid=0;
    soft_flush_valid=0;
    #1;
    if(query_state != META_FREE) $fatal(1,"flush and compute-start resurrected a resident slot");

    /* First beat ownership acquired on the flush edge stays quarantined. */
    allocate(descriptor(16'd13,10'd7,REQUEST_SPECULATIVE));
    query_slot=captured_slot;
    dma_start_slot=captured_slot;
    dma_start_epoch=13;
    dma_start_generation=captured_generation;
    dma_start_valid=1;
    soft_flush_valid=1;
    flush_epoch=13;
    @(negedge clk);
    dma_start_valid=0;
    soft_flush_valid=0;
    #1;
    if(query_state != META_DROP_ON_COMPLETE || !query_quarantined)
      $fatal(1,"same-cycle start and flush failed quarantine");
    dma_write_slot=captured_slot;
    dma_write_epoch=13;
    dma_write_generation=captured_generation;
    dma_write_valid=1;
    dma_write_last=0;
    @(negedge clk);
    #1;
    if(query_state != META_DROP_ON_COMPLETE || !query_quarantined)
      $fatal(1,"non-final DMA beat released flush ownership");
    dma_write_last=1;
    @(negedge clk);
    dma_write_valid=0;
    #1;
    if(query_state != META_FREE || query_quarantined)
      $fatal(1,"valid start and flush did not retire after final DMA");

    /* Neither wrong generation nor wrong epoch grants a new DMA owner. */
    for(integer stale_case=0; stale_case<2; stale_case=stale_case+1) begin
      allocate(descriptor(16'(14+stale_case),10'(8+stale_case),REQUEST_SPECULATIVE));
      query_slot=captured_slot;
      dma_start_slot=captured_slot;
      dma_start_epoch=16'(14+stale_case);
      dma_start_generation=captured_generation;
      if(stale_case==0) dma_start_generation=captured_generation-1'b1;
      else dma_start_epoch=dma_start_epoch+1'b1;
      dma_start_valid=1;
      soft_flush_valid=1;
      flush_epoch=16'(14+stale_case);
      @(negedge clk);
      dma_start_valid=0;
      soft_flush_valid=0;
      #1;
      if(query_state != META_QUEUED_SPEC || query_quarantined)
        $fatal(1,"stale ownership tag quarantined an unstarted request");
      retire_cancelled_valid=1;
      retire_slot=captured_slot;
      retire_epoch=flush_epoch;
      retire_generation=captured_generation;
      @(negedge clk);
      retire_cancelled_valid=0;
      #1;
      if(query_state != META_FREE) $fatal(1,"stale-start flush tombstone did not retire");
    end

    /* Full 16-bit slot bounds must be checked before truncating RAM indices. */
    allocate(descriptor(16'd16,10'd10,REQUEST_SPECULATIVE));
    query_slot=captured_slot;
    dma_start_slot=captured_slot+16'd16;
    dma_start_epoch=16;
    dma_start_generation=captured_generation;
    commit_slot=dma_start_slot;
    commit_epoch=16;
    commit_generation=captured_generation;
    dma_start_valid=1;
    commit_valid=1;
    cancel_valid=1;
    @(negedge clk);
    dma_start_valid=0;
    commit_valid=0;
    cancel_valid=0;
    #1;
    if(query_state != META_QUEUED_SPEC) $fatal(1,"invalid slot aliased a queued owner");
    /* High-bit start with valid low-bit alias cannot turn flush into quarantine. */
    dma_start_valid=1;
    soft_flush_valid=1;
    flush_epoch=16;
    @(negedge clk);
    dma_start_valid=0;
    soft_flush_valid=0;
    #1;
    if(query_state != META_QUEUED_SPEC || query_quarantined)
      $fatal(1,"invalid slot start quarantined an unstarted request");
    retire_cancelled_valid=1;
    retire_slot=captured_slot+16'd16;
    retire_epoch=16;
    retire_generation=captured_generation;
    @(negedge clk);
    #1;
    if(query_state != META_QUEUED_SPEC) $fatal(1,"invalid retirement slot aliased an owner");
    retire_slot=captured_slot;
    @(negedge clk);
    retire_cancelled_valid=0;

    allocate(descriptor(16'd17,10'd11,REQUEST_SPECULATIVE));
    query_slot=captured_slot;
    dma_start_slot=captured_slot;
    dma_start_epoch=17;
    dma_start_generation=captured_generation;
    dma_start_valid=1;
    @(negedge clk);
    dma_start_valid=0;
    dma_write_slot=captured_slot+16'd16;
    dma_write_epoch=17;
    dma_write_generation=captured_generation;
    dma_write_valid=1;
    dma_write_last=1;
    #1;
    if(write_accept) $fatal(1,"invalid DMA write slot accepted through alias");
    @(negedge clk);
    #1;
    if(query_state != META_INFLIGHT_SPEC) $fatal(1,"invalid DMA write changed owner state");
    dma_write_slot=captured_slot;
    commit_slot=captured_slot+16'd16;
    commit_epoch=17;
    commit_generation=captured_generation;
    commit_valid=1;
    cancel_valid=1;
    @(negedge clk);
    dma_write_valid=0;
    commit_valid=0;
    cancel_valid=0;
    #1;
    if(query_state != META_RESIDENT_SPEC)
      $fatal(1,"invalid cancellation/promotion aliased a valid final DMA beat");
    commit_valid=1;
    commit_slot=captured_slot;
    @(negedge clk);
    commit_valid=0;
    compute_start_valid=1;
    compute_start2_valid=1;
    compute_start_slot=captured_slot+16'd16;
    compute_start_epoch=17;
    compute_start_generation=captured_generation;
    consumer_release_valid=1;
    consumer_release2_valid=1;
    release_slot=captured_slot+16'd16;
    release_epoch=17;
    release_generation=captured_generation;
    @(negedge clk);
    compute_start_valid=0;
    compute_start2_valid=0;
    consumer_release_valid=0;
    consumer_release2_valid=0;
    #1;
    if(query_state != META_COMMITTED) $fatal(1,"invalid compute/release port aliased a resident owner");
    query_slot=captured_slot+16'd16;
    #1;
    if(query_state != META_FREE || query_epoch != 0 || query_generation != 0 || query_quarantined)
      $fatal(1,"invalid query port leaked aliased ownership");
    if(query2_state != META_FREE || query2_epoch != 0 || query2_generation != 0 || query2_quarantined)
      $fatal(1,"invalid second query port leaked aliased ownership");
    $display("PASS: OCM slot ownership and stale-write guard");
    $finish;
  end
endmodule
