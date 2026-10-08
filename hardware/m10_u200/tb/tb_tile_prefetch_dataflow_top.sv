`timescale 1ns/1ps
/** v0.9 顶层：logical tile -> table/OCM/token DMA -> route commit/cancel。 */
module tb_tile_prefetch_dataflow_top;
  import tile_prefetch_package::*;
  logic clk=0,rst_n=0,bandwidth_enable=0;
  logic logical_valid,logical_ready,existing_live,existing_spec,logical_promote;
  tile_descriptor_t gu,down,accepted_desc,dma_active_desc,compute_gu_desc,compute_down_desc;
  logic accepted_valid; descriptor_token_t accepted_token; logic [7:0] accepted_slot_generation,compute_gu_generation,compute_down_generation;
  logic compute_req_valid,compute_req_ready,compute_busy,compute_done,compute_wait_gu,compute_wait_down;
  logic sequence_active=1, layer_done=0, scheduler_active=1;
  logic [63:0] telemetry_cycles, telemetry_spec_alloc, telemetry_commit;
  logic route_valid,route_truth,route_ready; descriptor_token_t route_token;
  logic soft_flush_valid; logic [15:0] soft_flush_epoch;
  logic burst_start,burst_done,dma_done; logic [31:0] stall,stale,discarded,tombstones;
  descriptor_token_t gu_token,down_token;
  tile_descriptor_t capture_desc;
  always #5 clk=~clk;
  tile_prefetch_top #(.DESCRIPTOR_TABLE_ENTRIES(8),
    .NUM_OCM_BANKS(4),
    .SLOTS_PER_BANK(4),
    .DMA_FIFO_DEPTH(4),
    .MAX_BURST_BYTES(64)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(bandwidth_enable),
    .setup_cycles(0),
    .bytes_per_cycle(32),
    .sequence_active(sequence_active),
    .layer_done(layer_done),
    .scheduler_active(scheduler_active),
    .logical_valid(logical_valid),
    .logical_gu_descriptor(gu),
    .logical_down_descriptor(down),
    .existing_pair_live(existing_live),
    .existing_pair_is_speculative(existing_spec),
    .logical_ready(logical_ready),
    .logical_promotion_valid(logical_promote),
    .physical_accept_valid(accepted_valid),
    .physical_accept_token(accepted_token),
    .physical_accept_descriptor(accepted_desc),
    .physical_accept_slot_generation(accepted_slot_generation),
    .route_valid(route_valid),
    .route_token(route_token),
    .route_truth(route_truth),
    .route_ready(route_ready),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(soft_flush_epoch),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(compute_req_ready),
    .compute_gu_descriptor(compute_gu_desc),
    .compute_gu_slot_generation(compute_gu_generation),
    .compute_down_descriptor(compute_down_desc),
    .compute_down_slot_generation(compute_down_generation),
    .compute_gu_cycles(1),
    .compute_nonlinear_cycles(1),
    .compute_down_cycles(1),
    .compute_busy(compute_busy),
    .compute_done(compute_done),
    .compute_waiting_gu(compute_wait_gu),
    .compute_waiting_down(compute_wait_down),
    .dma_burst_start(burst_start),
    .dma_burst_done(burst_done),
    .dma_done(dma_done),
    .dma_active_descriptor(dma_active_desc),
    .demand_stall_cycles(stall),
    .stale_event_count(stale),
    .discarded_dma_write_count(discarded),
    .tombstone_skip_count(tombstones),
    .telemetry_total_cycles(telemetry_cycles),
    .telemetry_speculative_allocated_bytes(telemetry_spec_alloc),
    .telemetry_speculative_issued_bytes(),
    .telemetry_demand_issued_bytes(),
    .telemetry_compute_wait_gu_cycles(),
    .telemetry_compute_wait_down_cycles(),
    .telemetry_ocm_peak_occupancy(),
    .telemetry_commit_count(telemetry_commit),
    .telemetry_cancel_count());
  function automatic tile_descriptor_t descriptor(input matrix_kind_t matrix,input logic [15:0] epoch,input logic [9:0] expert,input request_class_t cls);
    tile_descriptor_t x;
    begin
      x='0;
      x.epoch_id=epoch;
      x.layer_id=1;
      x.expert_id=expert;
      x.tile_id=0;
      x.matrix_kind=matrix;
      x.byte_count=128;
      x.request_class=cls;
      descriptor=x;
    end
  endfunction
  task automatic issue_logical(input logic [15:0] epoch,input logic [9:0] expert,input request_class_t cls);
    begin
      @(negedge clk); while(!logical_ready) @(negedge clk);
      gu=descriptor(MATRIX_GU,epoch,expert,cls);
      down=descriptor(MATRIX_DOWN,epoch,expert,cls);
      logical_valid=1;
      @(negedge clk);
      logical_valid=0;
    end
  endtask
  task automatic capture_accept(output descriptor_token_t token,output tile_descriptor_t value,output logic [7:0] slot_generation);
    begin
      while(!accepted_valid) @(negedge clk);
      token=accepted_token;
      value=accepted_desc;
      slot_generation=accepted_slot_generation;
      @(negedge clk);
    end
  endtask
  task automatic resolve(input descriptor_token_t token,input logic truth);
    begin
      @(negedge clk);
      route_token=token;
      route_truth=truth;
      route_valid=1; #1;
      while(!route_ready) @(negedge clk);
      @(negedge clk);
      route_valid=0;
    end
  endtask
  initial begin
    logical_valid=0;
      gu='0;
      down='0;
      existing_live=0;
      existing_spec=0;
      route_valid=0;
      route_token='0;
      route_truth=0;
      soft_flush_valid=0;
      soft_flush_epoch=0;
      compute_req_valid=0;
      compute_gu_desc='0;
      compute_down_desc='0;
      compute_gu_generation=0;
      compute_down_generation=0;
    #17 rst_n=1;
    /* Keep bandwidth closed: both physical requests must queue before route. */
    issue_logical(1,10,REQUEST_SPECULATIVE);
    capture_accept(gu_token,compute_gu_desc,compute_gu_generation); if(compute_gu_desc.matrix_kind!=MATRIX_GU) $fatal(1,"GU admission order broken");
    capture_accept(down_token,compute_down_desc,compute_down_generation); if(compute_down_desc.matrix_kind!=MATRIX_DOWN) $fatal(1,"DOWN admission order broken");
    resolve(gu_token,1'b1); resolve(down_token,1'b1);
    bandwidth_enable=1;
    wait(dma_done && dma_active_desc.matrix_kind==MATRIX_GU);
    wait(dma_done && dma_active_desc.matrix_kind==MATRIX_DOWN);
    @(negedge clk); while(!compute_req_ready) @(negedge clk);
      compute_req_valid=1;
    @(negedge clk);
      compute_req_valid=0;
    wait(compute_done);
    if(stale != 0 || discarded != 0) $fatal(1,"integrated route path produced stale writes/events");
    if(telemetry_commit != 2 || telemetry_cycles == 0) $fatal(1,"telemetry did not observe integrated transactions");
    $display("PASS: v0.9 integrated logical-to-burst data path");
    $finish;
  end
endmodule
