`timescale 1ns/1ps
/** Four-slot pressure: reuse old GU as successor DOWN, reject old generation. */
module tb_gu_slot_reuse;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic allocate_valid = 0;
  logic allocate_ready, pair_ready;
  tile_descriptor_t incoming, allocated, saved, gu, down, next_gu, next_down;
  logic [7:0] allocated_generation, saved_generation, gu_generation, down_generation;
  logic dma_start = 0;
  logic dma_write = 0;
  logic [15:0] dma_slot = 0;
  logic [7:0] dma_generation = 0;
  logic [127:0] dma_data = 0;
  logic write_accept;
  logic route_accept = 0;
  logic enable_issue = 1;
  logic controller_ready;
  logic request_valid, request_ready, done, gu_start, down_start, gu_release, down_release;
  logic busy, layer_done;
  logic [31:0] layer_y, occupancy, stale_count, discarded_count;
  logic gu_done = 0;
  logic nonlinear_done = 0;
  logic down_done = 0;
  logic replay_old_gu_release = 0;
  wire release_gu = gu_release || replay_old_gu_release;
  logic [15:0] release_gu_slot, release_down_slot, release_epoch;
  logic [7:0] release_gu_generation, release_down_generation;
  metadata_state_t gu_state, down_state;
  logic [15:0] gu_epoch, down_epoch;
  logic [7:0] gu_tag, down_tag;
  logic gu_payload, down_payload;
  wire gu_resident;
  wire down_resident;
  assign gu_resident = gu_state == tile_prefetch_package::META_COMMITTED && gu_payload && gu_epoch == gu.epoch_id && gu_tag == gu_generation;
  assign down_resident = down_state == tile_prefetch_package::META_COMMITTED && down_payload && down_epoch == down.epoch_id && down_tag == down_generation;
  logic ram_read = 0;
  logic [1:0] ram_read_slot = 0;
  logic [127:0] ram_read_data;
  logic [31:0] ram_discards;
  logic [15:0] old_gu_slot;
  logic [7:0] old_gu_generation;
  assign request_ready = controller_ready && enable_issue;

  tile_execution_scheduler #(
    .SLOT_COUNT(4),
    .TOP_K_MAX(1),
    .TILES_PER_EXPERT(2),
    .H(1)
  ) scheduler (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .capture_valid(allocate_valid && allocate_ready),
    .capture_descriptor(allocated),
    .capture_generation(allocated_generation),
    .dma_done(dma_write),
    .dma_descriptor(saved),
    .route_accept(route_accept),
    .route_expert(10'd3),
    .route_weight_q15(16'd32768),
    .route_last(1'b1),
    .compute_req_valid(request_valid),
    .compute_req_ready(request_ready),
    .gu_descriptor(gu),
    .down_descriptor(down),
    .gu_generation(gu_generation),
    .down_generation(down_generation),
    .compute_done(done),
    .gu_release_valid(release_gu),
    .partial_y(32'd6),
    .layer_done(layer_done),
    .layer_y(layer_y)
  );
  tile_compute_controller #(
    .EXTERNAL_COMPUTE(1'b1)
  ) controller (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .compute_req_valid(request_valid && enable_issue),
    .compute_req_ready(controller_ready),
    .compute_gu_descriptor(gu),
    .compute_gu_slot_generation(gu_generation),
    .compute_down_descriptor(down),
    .compute_down_slot_generation(down_generation),
    .gu_resident(gu_resident),
    .down_resident(down_resident),
    .gu_cycles(32'd1),
    .nonlinear_cycles(32'd1),
    .down_cycles(32'd1),
    .backend_gu_done(gu_done),
    .backend_nonlinear_done(nonlinear_done),
    .backend_down_done(down_done),
    .busy(busy),
    .done(done),
    .waiting_gu(),
    .waiting_down(),
    .gu_compute_start(gu_start),
    .down_compute_start(down_start),
    .consumer_gu_release_valid(gu_release),
    .consumer_release_valid(down_release),
    .consumer_release_gu_slot(release_gu_slot),
    .consumer_release_down_slot(release_down_slot),
    .consumer_release_epoch(release_epoch),
    .consumer_release_gu_generation(release_gu_generation),
    .consumer_release_down_generation(release_down_generation)
  );
  ocm_slot_ownership #(
    .NUM_OCM_BANKS(2),
    .SLOTS_PER_BANK(2),
    .USE_PAIR_PRIMARY(1'b1)
  ) ocm (
    .clk(clk),
    .rst_n(rst_n),
    .allocate_valid(allocate_valid),
    .allocate_descriptor(incoming),
    .pair_check_descriptor(next_down),
    .pair_primary_descriptor(next_gu),
    .allocate_ready(allocate_ready),
    .pair_reserve_ready(pair_ready),
    .allocated_descriptor(allocated),
    .allocated_slot_generation(allocated_generation),
    .dma_start_valid(dma_start),
    .dma_start_slot(dma_slot),
    .dma_start_epoch(16'd1),
    .dma_start_slot_generation(dma_generation),
    .dma_write_valid(dma_write),
    .dma_write_last(1'b1),
    .dma_write_slot(dma_slot),
    .dma_write_epoch(16'd1),
    .dma_write_slot_generation(dma_generation),
    .dma_write_accept(write_accept),
    .commit_valid(1'b0),
    .commit_slot(16'd0),
    .commit_epoch(16'd0),
    .commit_slot_generation(8'd0),
    .cancel_valid(1'b0),
    .cancel_slot(16'd0),
    .cancel_epoch(16'd0),
    .cancel_slot_generation(8'd0),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch(16'd1),
    .retire_cancelled_valid(1'b0),
    .retire_cancelled_slot(16'd0),
    .retire_cancelled_epoch(16'd0),
    .retire_cancelled_slot_generation(8'd0),
    .consumer_release_valid(release_gu),
    .consumer_release_slot(release_gu_slot),
    .consumer_release_epoch(release_epoch),
    .consumer_release_slot_generation(release_gu_generation),
    .consumer_release2_valid(down_release),
    .consumer_release2_slot(release_down_slot),
    .consumer_release2_epoch(release_epoch),
    .consumer_release2_slot_generation(release_down_generation),
    .compute_start_valid(gu_start),
    .compute_start_slot(gu.ocm_slot),
    .compute_start_epoch(gu.epoch_id),
    .compute_start_slot_generation(gu_generation),
    .compute_start2_valid(down_start),
    .compute_start2_slot(down.ocm_slot),
    .compute_start2_epoch(down.epoch_id),
    .compute_start2_slot_generation(down_generation),
    .query_slot(gu.ocm_slot),
    .query_state(gu_state),
    .query_epoch(gu_epoch),
    .query_slot_generation(gu_tag),
    .query_quarantined(),
    .query_payload_complete(gu_payload),
    .query2_slot(down.ocm_slot),
    .query2_state(down_state),
    .query2_epoch(down_epoch),
    .query2_slot_generation(down_tag),
    .query2_quarantined(),
    .query2_payload_complete(down_payload),
    .stale_event_count(stale_count),
    .discarded_dma_write_count(discarded_count),
    .occupancy(occupancy)
  );
  ocm_payload_ram #(
    .SLOT_COUNT(4),
    .BYTES_PER_SLOT(64)
  ) payload (
    .clk(clk),
    .rst_n(rst_n),
    .slot_bind_valid(allocate_valid && allocate_ready),
    .slot_bind_slot(allocated.ocm_slot[1:0]),
    .slot_bind_epoch(allocated.epoch_id),
    .slot_bind_generation(allocated_generation),
    .slot_release_valid(release_gu),
    .slot_release_slot(release_gu_slot[1:0]),
    .slot_release_epoch(release_epoch),
    .slot_release_generation(release_gu_generation),
    .slot_release2_valid(down_release),
    .slot_release2_slot(release_down_slot[1:0]),
    .slot_release2_epoch(release_epoch),
    .slot_release2_generation(release_down_generation),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch(16'd1),
    .dma_write_valid(dma_write),
    .dma_write_slot(dma_slot[1:0]),
    .dma_write_offset(6'd0),
    .dma_write_data(dma_data),
    .dma_write_strobe(16'hffff),
    .dma_write_epoch(16'd1),
    .dma_write_generation(dma_generation),
    .dma_write_accept(),
    .read_valid(ram_read),
    .read_slot(ram_read_slot),
    .read_offset(6'd0),
    .read_data(ram_read_data),
    .discarded_write_count(ram_discards)
  );
  function automatic tile_descriptor_t descriptor(input integer expert, input integer tile, input matrix_kind_t matrix);
    tile_descriptor_t value;
    value = '0;
    value.epoch_id = 1;
    value.expert_id = 10'(expert);
    value.tile_id = 16'(tile);
    value.matrix_kind = matrix;
    value.request_class = REQUEST_DEMAND;
    value.byte_count = 16;
    return value;
  endfunction
  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask
  task automatic allocate(input tile_descriptor_t value, input bit complete);
    @(negedge clk);
    incoming = value;
    #0.1;
    if (!allocate_ready) $fatal(1,"pressure allocation was not available");
    saved = allocated;
    saved_generation = allocated_generation;
    allocate_valid = 1;
    tick();
    @(negedge clk);
    allocate_valid = 0;
    if (complete) begin
      dma_slot = saved.ocm_slot;
      dma_generation = saved_generation;
      dma_start = 1;
      tick();
      @(negedge clk);
      dma_start = 0;
      dma_data = {16{8'hc3}};
      dma_write = 1;
      #0.1;
      if (!write_accept) $fatal(1,"DMA payload was rejected");
      tick();
      @(negedge clk);
      dma_write = 0;
    end
  endtask
  task automatic finish_gu;
    @(negedge clk);
    gu_done = 1;
    tick();
    if (!gu_release || down_release || done) $fatal(1,"GU did not retire independently");
    @(negedge clk);
    gu_done = 0;
    tick();
  endtask
  task automatic finish_down;
    @(negedge clk);
    nonlinear_done = 1;
    tick();
    if (!down_start) $fatal(1,"resident DOWN did not start");
    @(negedge clk);
    nonlinear_done = 0;
    tick();
    @(negedge clk);
    down_done = 1;
    tick();
    if (!done || !down_release || gu_release) $fatal(1,"tile end repeated GU release");
    @(negedge clk);
    down_done = 0;
    tick();
  endtask
  initial begin
    incoming = '0;
    saved = '0;
    next_gu = descriptor(3,1,MATRIX_GU);
    next_down = descriptor(3,1,MATRIX_DOWN);
    repeat (3) @(negedge clk);
    rst_n = 1;
    allocate(descriptor(3,0,MATRIX_GU),1);
    old_gu_slot = saved.ocm_slot;
    old_gu_generation = saved_generation;
    allocate(descriptor(3,0,MATRIX_DOWN),1);
    allocate(descriptor(5,0,MATRIX_GU),0); // unrelated live owner creates bank pressure
    if (occupancy != 3 || pair_ready) $fatal(1,"next pair was not capacity blocked");
    @(negedge clk);
    route_accept = 1;
    @(negedge clk);
    route_accept = 0;
    wait (gu_start);
    tick();
    finish_gu();
    if (occupancy != 2 || !pair_ready || payload.valid_mem[old_gu_slot] || scheduler.valid_mem[old_gu_slot])
      $fatal(1,"GU release did not unlock pair capacity before DOWN");
    allocate(next_gu,1);
    allocate(next_down,1);
    if (saved.ocm_slot != old_gu_slot || saved_generation == old_gu_generation || !busy || done)
      $fatal(1,"GU slot was not reused as next DOWN while old tile remained active");
    @(negedge clk);
    replay_old_gu_release = 1;
    tick();
    @(negedge clk);
    replay_old_gu_release = 0;
    if (!payload.valid_mem[old_gu_slot] || !scheduler.valid_mem[old_gu_slot] ||
        ocm.state_mem[old_gu_slot] != META_COMMITTED || stale_count != 1)
      $fatal(1,"old GU generation release damaged new DOWN owner");
    enable_issue = 0; // Hold the next handshake while checking reused payload.
    finish_down();
    if (!payload.valid_mem[old_gu_slot] || !scheduler.valid_mem[old_gu_slot] ||
        payload.gen_mem[old_gu_slot] != saved_generation || stale_count != 1)
      $fatal(1,"old tile retirement invalidated reused GU slot");
    @(negedge clk);
    ram_read = 1;
    ram_read_slot = old_gu_slot[1:0];
    tick();
    if (ram_read_data !== {16{8'hc3}}) $fatal(1,"reused slot payload was corrupted");
    @(negedge clk);
    ram_read = 0;
    enable_issue = 1;
    wait (gu_start);
    tick();
    finish_gu();
    finish_down();
    wait (layer_done);
    if (layer_y != 12 || occupancy != 1 || ram_discards || discarded_count || stale_count != 1)
      $fatal(1,"new owner computation/retirement or unrelated ownership failed");
    $display("PASS: GU early capacity release, same-slot DOWN reuse, stale generation protection and payload preservation");
    $finish;
  end
  initial begin
    #20000;
    $display("GU slot=%0d state=%0d payload=%0b epoch=%0d/%0d gen=%0d/%0d allocated=%0d",gu.ocm_slot,gu_state,gu_payload,gu_epoch,gu.epoch_id,gu_tag,gu_generation,allocated.ocm_slot);
    $display("resident=%b eq=%b/%b/%b/%b",controller.gu_resident,gu_state==META_COMMITTED,gu_payload,gu_epoch==gu.epoch_id,gu_tag==gu_generation);
    $fatal(1,"GU slot reuse watchdog phase=%0d running=%0b truth=%0b issue=%0b gu_found=%0b occupancy=%0d tile=%0d", controller.phase, scheduler.running, scheduler.truth_ready, scheduler.issue_pending, scheduler.gu_found, occupancy, scheduler.next_tiles[0]);
  end
endmodule
