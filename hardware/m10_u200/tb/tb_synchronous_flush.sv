`timescale 1ns/1ps
/** 检查同步 abort、同周期握手抑制和下一事务重启，不复位 payload RAM。 */
module tb_synchronous_flush;
  import tile_prefetch_package::*;
  logic clk = 0;
  logic rst_n = 0;
  logic clear = 0;
  always #2.5 clk = ~clk;

  logic prediction_begin_valid = 0;
  logic prediction_begin_ready;
  logic candidate_valid = 0;
  logic candidate_ready;
  logic candidate_accept;
  logic route_begin_valid = 0;
  logic route_begin_ready;
  logic route_expert_valid = 0;
  logic route_expert_ready;
  logic route_accept;
  control_transaction_parser #(
    .MAX_EXPERTS(4),
    .EXPERT_W(2)
  ) parser (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .prediction_begin_valid(prediction_begin_valid),
    .prediction_begin_ready(prediction_begin_ready),
    .prediction_epoch(16'd1),
    .prediction_layer(3'd0),
    .prediction_count(5'd1),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(2'd1),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd1000),
    .candidate_last(1'b1),
    .route_begin_valid(route_begin_valid),
    .route_begin_ready(route_begin_ready),
    .route_epoch(16'd1),
    .route_layer(3'd0),
    .route_top_k(5'd1),
    .route_expert_valid(route_expert_valid),
    .route_expert_ready(route_expert_ready),
    .route_expert(2'd1),
    .route_last(1'b1),
    .candidate_accept(candidate_accept),
    .route_accept(route_accept)
  );

  tile_descriptor_t descriptor;
  logic logical_valid = 0;
  logic logical_ready;
  logic gu_valid;
  logic down_valid;
  logic reserve_pair_valid;
  logic promotion_valid;
  descriptor_manager manager (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .logical_valid(logical_valid),
    .logical_gu_descriptor(descriptor),
    .logical_down_descriptor(descriptor),
    .existing_pair_live(1'b0),
    .existing_pair_is_speculative(1'b0),
    .logical_ready(logical_ready),
    .gu_slot_reserve_ready(1'b1),
    .down_slot_reserve_ready(1'b1),
    .descriptor_pair_reserve_ready(1'b1),
    .queue_pair_reserve_ready(1'b1),
    .promotion_ready(1'b1),
    .reserve_pair_valid(reserve_pair_valid),
    .promotion_valid(promotion_valid),
    .gu_valid(gu_valid),
    .gu_ready(1'b0),
    .down_valid(down_valid),
    .down_ready(1'b0)
  );

  logic compute_req_valid = 0;
  logic compute_req_ready;
  logic compute_busy;
  logic compute_done;
  logic compute_release;
  logic gu_resident = 0;
  tile_compute_controller controller (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .compute_req_valid(compute_req_valid),
    .compute_req_ready(compute_req_ready),
    .compute_gu_descriptor(descriptor),
    .compute_gu_slot_generation(8'd1),
    .compute_down_descriptor(descriptor),
    .compute_down_slot_generation(8'd1),
    .gu_resident(gu_resident),
    .down_resident(1'b1),
    .gu_cycles(32'd2),
    .nonlinear_cycles(32'd2),
    .down_cycles(32'd2),
    .backend_gu_done(1'b0),
    .backend_nonlinear_done(1'b0),
    .backend_down_done(1'b0),
    .busy(compute_busy),
    .done(compute_done),
    .consumer_release_valid(compute_release)
  );

  logic gu_start = 0;
  logic router_start = 0;
  logic read_valid;
  logic router_read_valid;
  logic router_done;
  logic gu_done;
  logic nonlinear_done;
  logic down_done;
  logic [127:0] router_logits;
  int8_swiglu_tile_backend #(
    .NUM_ROUTER_EXPERTS(4)
  ) backend (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .gu_start(gu_start),
    .down_start(1'b0),
    .router_start(router_start),
    .router_done(router_done),
    .router_read_valid(router_read_valid),
    .router_read_data(128'd0),
    .router_logits(router_logits),
    .activation(128'd0),
    .gu_requant_shift(5'd0),
    .gu_slot(16'd0),
    .down_slot(16'd1),
    .read_valid(read_valid),
    .read_data(128'd0),
    .gu_done(gu_done),
    .nonlinear_done(nonlinear_done),
    .down_done(down_done)
  );

  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask

  initial begin
    descriptor = '0;
    repeat (2) tick();
    @(negedge clk);
    rst_n = 1;
    prediction_begin_valid = 1;
    route_begin_valid = 1;
    logical_valid = 1;
    compute_req_valid = 1;
    gu_start = 1;
    tick();
    @(negedge clk);
    prediction_begin_valid = 0;
    route_begin_valid = 0;
    logical_valid = 0;
    compute_req_valid = 0;
    gu_start = 0;
    if (!candidate_ready || !route_expert_ready || !gu_valid || !compute_busy)
      $fatal(1, "setup transactions did not become active");
    clear = 1;
    candidate_valid = 1;
    route_expert_valid = 1;
    logical_valid = 1;
    compute_req_valid = 1;
    #0.1;
    if (candidate_ready || route_expert_ready || prediction_begin_ready ||
        route_begin_ready || gu_valid || down_valid || reserve_pair_valid ||
        promotion_valid || logical_ready || compute_req_ready || read_valid || router_read_valid)
      $fatal(1, "abort must inhibit all new handshakes and payload reads");
    tick();
    if (candidate_accept || route_accept || compute_busy || compute_done ||
        compute_release || gu_done || nonlinear_done || down_done || router_done)
      $fatal(1, "aborted operation produced completion or retained active state");
    @(negedge clk);
    clear = 0;
    candidate_valid = 0;
    route_expert_valid = 0;
    logical_valid = 0;
    compute_req_valid = 0;
    #0.1;
    if (!prediction_begin_ready || !route_begin_ready || !logical_ready ||
        !compute_req_ready || gu_valid || down_valid)
      $fatal(1, "synchronous abort failed to retire transaction state");
    prediction_begin_valid = 1;
    route_begin_valid = 1;
    compute_req_valid = 1;
    gu_resident = 1;
    router_start = 1;
    tick();
    @(negedge clk);
    prediction_begin_valid = 0;
    route_begin_valid = 0;
    compute_req_valid = 0;
    router_start = 0;
    candidate_valid = 1;
    route_expert_valid = 1;
    tick();
    if (!candidate_accept || !route_accept)
      $fatal(1, "fresh parser transaction failed after abort");
    @(negedge clk);
    candidate_valid = 0;
    route_expert_valid = 0;
    wait (compute_done);
    #0.1;
    if (!compute_release) $fatal(1, "fresh compute did not release slots");
    wait (router_done);
    #0.1;
    if (router_logits !== 128'd0) $fatal(1, "fresh Router computation wrong after abort");
    $display("PASS: synchronous flush handshake suppression, abort retirement and restart");
    $finish;
  end

  initial begin
    #10000;
    $fatal(1, "synchronous flush test timeout");
  end
endmodule
