`timescale 1ns/1ps
/** DescriptorTable 与 ID-token burst DMA ABI 的端到端定向测试。 */
module tb_descriptor_dma_path;
  import tile_prefetch_package::*;
  localparam integer ID_W=2;
  logic clk=0,rst_n=0;
  logic alloc_valid,alloc_ready; tile_descriptor_t alloc_desc; logic [ID_W-1:0] alloc_id; logic [7:0] alloc_gen;
  logic cancelled_valid; logic [ID_W-1:0] cancelled_id; logic [7:0] cancelled_gen;
  logic stale; logic [ID_W-1:0] query_id; logic query_valid; logic [7:0] query_gen; logic query_cancelled,query_promoted,query_inflight,query_drop; tile_descriptor_t query_desc;
  logic demand_valid,demand_ready,promoted_valid,promoted_ready,spec_valid,spec_ready; descriptor_token_t demand_token,promoted_token,spec_token;
  logic lookup_valid; descriptor_token_t lookup_token,claim_token,active_token; logic claim_valid,burst_start,burst_done,dma_done,active;
  logic [31:0] burst_bytes,remaining,stall,dpeak,ppeak,speak,tombstones; tile_descriptor_t active_desc;
  logic [15:0] claim_log[0:2]; integer claims;
  always #5 clk=~clk;
  descriptor_table #(.ENTRY_COUNT(4),
    .ID_W(ID_W)) descriptor_table_inst (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(alloc_valid),
    .allocate_descriptor(alloc_desc),
    .allocate_ready(alloc_ready),
    .allocate_request_id(alloc_id),
    .allocate_generation(alloc_gen),
    .mark_promoted_valid(1'b0),
    .mark_promoted_id('0),
    .mark_promoted_generation('0),
    .mark_cancelled_valid(cancelled_valid),
    .mark_cancelled_id(cancelled_id),
    .mark_cancelled_generation(cancelled_gen),
    .mark_dma_start_valid(claim_valid),
    .mark_dma_start_id(claim_token.request_id[ID_W-1:0]),
    .mark_dma_start_generation(claim_token.generation),
    .mark_dma_done_valid(dma_done),
    .mark_dma_done_id(active_token.request_id[ID_W-1:0]),
    .mark_dma_done_generation(active_token.generation),
    .retire_refs_valid(1'b0),
    .retire_refs_id('0),
    .retire_refs_generation('0),
    .stale_event(stale),
    .query_id(query_id),
    .query_valid(query_valid),
    .query_generation(query_gen),
    .query_cancelled(query_cancelled),
    .query_promoted(query_promoted),
    .query_inflight(query_inflight),
    .query_drop_on_complete(query_drop),
    .query_descriptor(query_desc));
  assign query_id=lookup_token.request_id[ID_W-1:0];
  burst_dma_channel #(.FIFO_DEPTH(4),
    .MAX_BURST_BYTES(128)) dma (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(0),
    .bytes_per_cycle(64),
    .demand_enqueue_valid(demand_valid),
    .demand_enqueue_token(demand_token),
    .demand_enqueue_ready(demand_ready),
    .promoted_enqueue_valid(promoted_valid),
    .promoted_enqueue_token(promoted_token),
    .promoted_enqueue_ready(promoted_ready),
    .speculative_enqueue_valid(spec_valid),
    .speculative_enqueue_token(spec_token),
    .speculative_enqueue_ready(spec_ready),
    .descriptor_lookup_valid(lookup_valid),
    .descriptor_lookup_token(lookup_token),
    .descriptor_entry_valid(query_valid && query_gen==lookup_token.generation),
    .descriptor_entry_cancelled(query_cancelled),
    .descriptor_lookup_descriptor(query_desc),
    .dma_claim_valid(claim_valid),
    .dma_claim_token(claim_token),
    .dma_burst_start(burst_start),
    .dma_burst_done(burst_done),
    .dma_burst_bytes(burst_bytes),
    .dma_done(dma_done),
    .active(active),
    .active_token(active_token),
    .active_descriptor(active_desc),
    .active_remaining_bytes(remaining),
    .demand_stall_cycles(stall),
    .demand_fifo_peak(dpeak),
    .promoted_fifo_peak(ppeak),
    .speculative_fifo_peak(speak),
    .tombstone_skip_count(tombstones));
  always @(posedge clk) if(claim_valid) begin claim_log[claims] <= claim_token.request_id;
      claims <= claims+1; end
  function automatic tile_descriptor_t desc(input logic [9:0] expert,input logic [31:0] bytes,input request_class_t cls);
    tile_descriptor_t x; begin x='0;
      x.epoch_id=1;
      x.layer_id=1;
      x.expert_id=expert;
      x.byte_count=bytes;
      x.request_class=cls;
      desc=x; end
  endfunction
  task automatic create(input tile_descriptor_t value,output descriptor_token_t token);
    begin
      @(negedge clk); while(!alloc_ready) @(negedge clk);
      alloc_desc=value;
      alloc_valid=1; #1;
      token.request_id=alloc_id;
      token.generation=alloc_gen;
      @(negedge clk);
      alloc_valid=0;
    end
  endtask
  task automatic enqueue_spec(input descriptor_token_t token);
    begin @(negedge clk); while(!spec_ready) @(negedge clk);
      spec_token=token;
      spec_valid=1; @(negedge clk);
      spec_valid=0; end
  endtask
  task automatic enqueue_demand(input descriptor_token_t token);
    begin @(negedge clk); while(!demand_ready) @(negedge clk);
      demand_token=token;
      demand_valid=1; @(negedge clk);
      demand_valid=0; end
  endtask
  descriptor_token_t bad_spec, long_spec, demand;
  initial begin
    alloc_valid=0;
      alloc_desc='0;
      cancelled_valid=0;
      cancelled_id='0;
      cancelled_gen='0;
      demand_valid=0;
      promoted_valid=0;
      spec_valid=0;
      demand_token='0;
      promoted_token='0;
      spec_token='0;
      claims=0;
    #17 rst_n=1;
    create(desc(20,64,REQUEST_SPECULATIVE),bad_spec);
    @(negedge clk);
      cancelled_id=bad_spec.request_id[ID_W-1:0];
      cancelled_gen=bad_spec.generation;
      cancelled_valid=1;
    @(negedge clk);
      cancelled_valid=0;
    enqueue_spec(bad_spec); wait(tombstones==1);
    create(desc(10,512,REQUEST_SPECULATIVE),long_spec);
    create(desc(1,64,REQUEST_DEMAND),demand);
    enqueue_spec(long_spec); wait(burst_start); enqueue_demand(demand);
    wait(claims>=2);
    if(claim_log[0] != long_spec.request_id || claim_log[1] != demand.request_id) $fatal(1,"table-backed demand priority failed");
    wait(dma_done && active_token.request_id==demand.request_id);
    if(stale) $fatal(1,"completion generated a stale DescriptorTable event");
    wait(dma_done && active_token.request_id==long_spec.request_id);
    $display("PASS: DescriptorTable to token DMA burst path");
    $finish;
  end
endmodule
