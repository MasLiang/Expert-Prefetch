`timescale 1ns/1ps
module tb_burst_dma_channel;
  import tile_prefetch_package::*;
  logic clk=0, rst_n=0, bandwidth_enable=1;
  logic demand_valid, demand_ready, promoted_valid, promoted_ready, spec_valid, spec_ready;
  descriptor_token_t demand_token, promoted_token, spec_token, lookup_token, claim_token, active_token;
  logic entry_valid, entry_cancelled, claim_valid, burst_start, burst_done, dma_done, active;
  tile_descriptor_t lookup_descriptor, active_descriptor;
  logic [31:0] burst_bytes, remaining, stall, demand_peak, promoted_peak, spec_peak, tombstones;
  logic lookup_valid;
  logic [15:0] claim_log [0:3]; integer claim_count;
  integer transferred_spec = 0;
  always @(posedge clk) begin
    if (rst_n && burst_done && active_descriptor.expert_id == 1)
      transferred_spec <= transferred_spec + burst_bytes;
  end
  always #5 clk=~clk;

  function automatic tile_descriptor_t descriptor(input logic [15:0] id, input logic [31:0] bytes, input request_class_t cls);
    tile_descriptor_t x;
    begin
      x='0;
      x.epoch_id=1;
      x.expert_id=id[9:0];
      x.byte_count=bytes;
      x.request_class=cls;
      descriptor=x;
    end
  endfunction
  always_comb begin
    lookup_descriptor='0;
      entry_valid=lookup_valid;
      entry_cancelled=0;
    case (lookup_token.request_id)
      16'd1: lookup_descriptor=descriptor(1,512,REQUEST_SPECULATIVE);
      16'd2: lookup_descriptor=descriptor(2,64,REQUEST_DEMAND);
      16'd3: begin lookup_descriptor=descriptor(3,64,REQUEST_SPECULATIVE);
      entry_cancelled=1; end
      default: entry_valid=0;
    endcase
  end
  always @(posedge clk) if(claim_valid) begin claim_log[claim_count] <= claim_token.request_id;
      claim_count <= claim_count + 1; end
  burst_dma_channel #(.FIFO_DEPTH(4),
    .MAX_BURST_BYTES(128)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
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
    .descriptor_entry_valid(entry_valid),
    .descriptor_entry_cancelled(entry_cancelled),
    .descriptor_lookup_descriptor(lookup_descriptor),
    .dma_claim_valid(claim_valid),
    .dma_claim_token(claim_token),
    .dma_burst_start(burst_start),
    .dma_burst_done(burst_done),
    .dma_burst_bytes(burst_bytes),
    .dma_done(dma_done),
    .active(active),
    .active_token(active_token),
    .active_descriptor(active_descriptor),
    .active_remaining_bytes(remaining),
    .demand_stall_cycles(stall),
    .demand_fifo_peak(demand_peak),
    .promoted_fifo_peak(promoted_peak),
    .speculative_fifo_peak(spec_peak),
    .tombstone_skip_count(tombstones));
  task automatic enqueue_spec(input logic [15:0] id);
    begin @(negedge clk); while(!spec_ready) @(negedge clk);
      spec_token.request_id=id;
      spec_token.generation=1;
      spec_valid=1; @(negedge clk);
      spec_valid=0; end
  endtask
  task automatic enqueue_demand(input logic [15:0] id);
    begin @(negedge clk); while(!demand_ready) @(negedge clk);
      demand_token.request_id=id;
      demand_token.generation=1;
      demand_valid=1; @(negedge clk);
      demand_valid=0; end
  endtask
  initial begin
    demand_valid=0;
      promoted_valid=0;
      spec_valid=0;
      demand_token='0;
      promoted_token='0;
      spec_token='0;
      claim_count=0;
    #17 rst_n=1;
    enqueue_spec(3); /* Tombstone must be skipped without reconstructing a FIFO. */
    wait(tombstones==1);
    enqueue_spec(1);
    wait(burst_start);
    enqueue_demand(2); /* Arrives during an unpreemptible speculative burst. */
    wait(burst_done);
    wait(claim_count >= 2);
    if(claim_log[0] != 1 || claim_log[1] != 2) $fatal(1,"demand did not win at next burst boundary");
    wait(dma_done && active_descriptor.expert_id==2);
    wait(dma_done && active_descriptor.expert_id==1);
    @(posedge clk);
    @(negedge clk);
    if (transferred_spec != 512) $fatal(1, "preempted DMA restarted payload: transferred=%0d", transferred_spec);
    if(demand_peak != 1 || tombstones != 1) $fatal(1,"FIFO accounting or tombstone skip incorrect");
    $display("PASS: burst arbitration, priority and tombstone skipping");
    $finish;
  end
endmodule
