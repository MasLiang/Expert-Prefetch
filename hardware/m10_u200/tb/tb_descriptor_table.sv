`timescale 1ns/1ps
module tb_descriptor_table;
  import tile_prefetch_package::*;
  logic clk=0, rst_n=0, alloc_valid, alloc_ready, promote_valid, cancel_valid;
  logic dma_start_valid, dma_done_valid, retire_valid, stale_event;
  logic [1:0] alloc_id, id, query_id; logic [7:0] alloc_gen, generation, query_gen;
  tile_descriptor_t alloc_desc, query_desc;
  logic qvalid, qcancelled, qpromoted, qinflight, qdrop;
  always #5 clk=~clk;
  descriptor_table #(.ENTRY_COUNT(4)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(alloc_valid),
    .allocate_descriptor(alloc_desc),
    .allocate_ready(alloc_ready),
    .allocate_request_id(alloc_id),
    .allocate_generation(alloc_gen),
    .mark_promoted_valid(promote_valid),
    .mark_promoted_id(id),
    .mark_promoted_generation(generation),
    .mark_cancelled_valid(cancel_valid),
    .mark_cancelled_id(id),
    .mark_cancelled_generation(generation),
    .mark_dma_start_valid(dma_start_valid),
    .mark_dma_start_id(id),
    .mark_dma_start_generation(generation),
    .mark_dma_done_valid(dma_done_valid),
    .mark_dma_done_id(id),
    .mark_dma_done_generation(generation),
    .retire_refs_valid(retire_valid),
    .retire_refs_id(id),
    .retire_refs_generation(generation),
    .stale_event(stale_event),
    .query_id(query_id),
    .query_valid(qvalid),
    .query_generation(query_gen),
    .query_cancelled(qcancelled),
    .query_promoted(qpromoted),
    .query_inflight(qinflight),
    .query_drop_on_complete(qdrop),
    .query_descriptor(query_desc));
  initial begin
    alloc_valid=0;
      promote_valid=0;
      cancel_valid=0;
      dma_start_valid=0;
      dma_done_valid=0;
      retire_valid=0;
    alloc_desc='0;
      id=0;
      generation=0;
      query_id=0;
      alloc_desc.epoch_id=3;
      alloc_desc.byte_count=64;
    #17 rst_n=1;
    @(negedge clk);
      id=alloc_id;
      generation=alloc_gen;
      query_id=alloc_id;
      alloc_valid=1;
    @(negedge clk);
      alloc_valid=0;
    if(!qvalid || query_gen!=generation) $fatal(1,"allocation or generation failed");
    @(negedge clk);
      promote_valid=1; @(negedge clk);
      promote_valid=0;
    if(!qpromoted) $fatal(1,"promotion failed");
    @(negedge clk);
      dma_start_valid=1; @(negedge clk);
      dma_start_valid=0;
    @(negedge clk);
      cancel_valid=1; @(negedge clk);
      cancel_valid=0;
    if(!qdrop || !qinflight) $fatal(1,"inflight cancel must quarantine entry");
    @(negedge clk);
      dma_done_valid=1; @(negedge clk);
      dma_done_valid=0;
    if(!qvalid) $fatal(1,"entry released before references retired");
    @(negedge clk);
      retire_valid=1; @(negedge clk);
      retire_valid=0;
    if(qvalid) $fatal(1,"entry did not retire after DMA and references retired");
    generation=8'hff; @(negedge clk);
      cancel_valid=1; @(negedge clk);
      cancel_valid=0;
    if(!stale_event) $fatal(1,"stale generation was accepted");
    /* Allocation and DMA start on different entries must both take effect. */
    @(negedge clk);
    id = alloc_id;
    query_id = alloc_id;
    generation = alloc_gen;
    alloc_valid = 1;
    @(negedge clk);
    dma_start_valid = 1;
    @(negedge clk);
    alloc_valid = 0;
    dma_start_valid = 0;
    if (!qinflight || !qvalid) $fatal(1, "allocation suppressed independent DMA start");
    @(negedge clk);
    alloc_valid = 1;
    dma_done_valid = 1;
    @(negedge clk);
    alloc_valid = 0;
    dma_done_valid = 0;
    if (qinflight || !qvalid) $fatal(1, "allocation suppressed independent DMA completion");
    $display("PASS: descriptor table retirement and stale protection"); $finish;
  end
endmodule
