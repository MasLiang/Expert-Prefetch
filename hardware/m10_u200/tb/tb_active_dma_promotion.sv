`timescale 1ns/1ps
/** Active hit promotion in setup, mid-burst and exactly at burst boundary. */
module tb_active_dma_promotion #(
  parameter integer PHASE=0
);
  import tile_prefetch_package::*;
  logic clk=0;
  always #5 clk=~clk;
  logic rst_n=0;
  logic spec_valid=0, demand_valid=0, notify_valid=0;
  descriptor_token_t spec_token='0, demand_token='0, notify_token='0;
  descriptor_token_t lookup_token, claim_token, active_token;
  tile_descriptor_t lookup_descriptor, active_descriptor;
  wire spec_ready, demand_ready, lookup_valid, claim_valid, burst_start, burst_done, dma_done, active;
  wire [31:0] burst_bytes;
  logic [15:0] claim_log [0:7];
  integer claims=0;
  integer transferred=0;
  logic hit_promoted=0;
  always @(posedge clk) if(claim_valid) claims<=claims+1;
  always @(posedge clk) if(claim_valid) claim_log[claims]<=claim_token.request_id;
  always @(posedge clk) if(burst_done) transferred<=transferred+int'(burst_bytes);
  always @(posedge clk) if(notify_valid && notify_token.request_id==0 && notify_token.generation==1) hit_promoted<=1;
  always_comb begin
    lookup_descriptor='0;
    lookup_descriptor.expert_id=10'(lookup_token.request_id);
    lookup_descriptor.byte_count=lookup_token.request_id==0 ? 256 : 64;
    lookup_descriptor.request_class=REQUEST_SPECULATIVE;
    if(lookup_token.request_id==2) lookup_descriptor.request_class=REQUEST_DEMAND;
  end
  burst_dma_channel #(
    .FIFO_DEPTH(4),
    .MAX_BURST_BYTES(64),
    .PROGRESS_ENTRIES(4)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd4),
    .bytes_per_cycle(32'd8),
    .external_burst_done(1'b0),
    .promotion_notify_valid(notify_valid),
    .promotion_notify_token(notify_token),
    .demand_enqueue_valid(demand_valid),
    .demand_enqueue_token(demand_token),
    .demand_enqueue_ready(demand_ready),
    .promoted_enqueue_valid(1'b0),
    .promoted_enqueue_token('0),
    .promoted_enqueue_ready(),
    .speculative_enqueue_valid(spec_valid),
    .speculative_enqueue_token(spec_token),
    .speculative_enqueue_ready(spec_ready),
    .demand_pair_reserve_ready(),
    .speculative_pair_reserve_ready(),
    .descriptor_lookup_valid(lookup_valid),
    .descriptor_lookup_token(lookup_token),
    .descriptor_entry_valid(lookup_valid),
    .descriptor_entry_cancelled(1'b0),
    .descriptor_entry_promoted(hit_promoted && lookup_token.request_id==0),
    .descriptor_entry_inflight(1'b1),
    .descriptor_lookup_slot_generation(8'd1),
    .descriptor_lookup_descriptor(lookup_descriptor),
    .dma_claim_valid(claim_valid),
    .dma_claim_token(claim_token),
    .dma_burst_start(burst_start),
    .dma_burst_done(burst_done),
    .dma_burst_bytes(burst_bytes),
    .dma_done(dma_done),
    .active(active),
    .drained(),
    .active_token(active_token),
    .active_descriptor(active_descriptor),
    .active_slot_generation(),
    .active_remaining_bytes(),
    .demand_stall_cycles(),
    .demand_fifo_peak(),
    .promoted_fifo_peak(),
    .speculative_fifo_peak(),
    .tombstone_skip_count(),
    .tombstone_retired_valid(),
    .tombstone_retired_cancelled(),
    .tombstone_retired_token(),
    .tombstone_retired_descriptor(),
    .tombstone_retired_slot_generation()
  );
  task automatic enqueue_spec(input integer id);
    @(negedge clk);
    wait(spec_ready);
    spec_token.request_id=16'(id); spec_token.generation=1; spec_valid=1;
    @(negedge clk); spec_valid=0;
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;
    enqueue_spec(0);
    wait(active);
    enqueue_spec(1);
    @(negedge clk); demand_token.request_id=2; demand_token.generation=1; demand_valid=1;
    @(negedge clk); demand_valid=0;
    if(PHASE==1) begin wait(burst_start); repeat(2) @(negedge clk); end
    if(PHASE==2) begin wait(dut.state==dut.BURST && dut.burst_remaining_cycles==1); @(negedge clk); end
    notify_token.request_id=0; notify_token.generation=PHASE==3 ? 2 : 1; notify_valid=1;
    @(negedge clk); notify_valid=0;
    wait(dma_done && active_descriptor.expert_id==1);
    if(PHASE==3) wait(dma_done && active_descriptor.expert_id==0);
    repeat(3) @(negedge clk);
    if(transferred!=384) $fatal(1,"payload restarted or lost %0d",transferred);
    if(PHASE!=3 && (claims!=3 || claim_log[0]!=0 || claim_log[1]!=2 || claim_log[2]!=1))
      $fatal(1,"active hit fell behind future speculation claims=%0d order=%0d,%0d,%0d",claims,claim_log[0],claim_log[1],claim_log[2]);
    if(PHASE==3 && (claims!=4 || claim_log[1]!=2 || claim_log[2]!=1 || claim_log[3]!=0))
      $fatal(1,"stale generation incorrectly promoted active owner");
    $display("PASS: active promotion phase=%0d priority, generation and 384-byte conservation",PHASE);
    $finish;
  end
  initial begin #20000; $fatal(1,"promotion watchdog"); end
endmodule
