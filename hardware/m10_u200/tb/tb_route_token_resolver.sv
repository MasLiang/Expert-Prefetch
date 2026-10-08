`timescale 1ns/1ps

/** Self-checking Router truth to descriptor-token resolution test. */
module tb_route_token_resolver;
  import tile_prefetch_package::*;

  logic clk;
  logic rst_n;
  logic capture_valid;
  descriptor_token_t capture_token;
  tile_descriptor_t capture_descriptor;
  logic route_accept;
  logic [15:0] route_epoch;
  logic [2:0] route_layer;
  logic [9:0] route_expert;
  logic route_last;
  logic route_valid;
  descriptor_token_t route_token;
  logic route_truth;
  logic route_ready;
  logic resolver_full;
  logic [31:0] dropped_capture_count;
  integer observed_count;

  route_token_resolver #(
    .MAX_LIVE(4),
    .TOP_K_MAX(2),
    .EXPERT_W(10)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .capture_valid(capture_valid),
    .capture_token(capture_token),
    .capture_descriptor(capture_descriptor),
    .route_accept(route_accept),
    .route_epoch(route_epoch),
    .route_layer(route_layer),
    .route_expert(route_expert),
    .route_last(route_last),
    .route_valid(route_valid),
    .route_token(route_token),
    .route_truth(route_truth),
    .route_ready(route_ready),
    .resolver_full(resolver_full),
    .dropped_capture_count(dropped_capture_count)
  );

  always #5 clk = ~clk;

  task automatic capture_descriptor_token(
    input logic [15:0] request_id,
    input logic [7:0] generation,
    input logic [9:0] expert_id
  );
    begin
      @(negedge clk);
      capture_valid = 1'b1;
      capture_token.request_id = request_id;
      capture_token.generation = generation;
      capture_descriptor = '0;
      capture_descriptor.epoch_id = 16'd9;
      capture_descriptor.layer_id = 16'd3;
      capture_descriptor.expert_id = expert_id;
      @(negedge clk);
      capture_valid = 1'b0;
    end
  endtask

  task automatic accept_route_expert(
    input logic [9:0] expert_id,
    input logic last
  );
    begin
      @(negedge clk);
      route_accept = 1'b1;
      route_epoch = 16'd9;
      route_layer = 3'd3;
      route_expert = expert_id;
      route_last = last;
      @(negedge clk);
      route_accept = 1'b0;
      route_last = 1'b0;
    end
  endtask

  always @(posedge clk) begin
    if (rst_n && route_valid && route_ready) begin
      if (observed_count == 0) begin
        if (route_token.request_id != 16'd1 || !route_truth) begin
          $fatal(1, "first token must commit expert 11");
        end
      end else if (observed_count == 1) begin
        if (route_token.request_id != 16'd2 || route_truth) begin
          $fatal(1, "second token must cancel expert 12");
        end
      end else begin
        $fatal(1, "unexpected extra resolved token");
      end
      observed_count = observed_count + 1;
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    capture_valid = 1'b0;
    capture_token = '0;
    capture_descriptor = '0;
    route_accept = 1'b0;
    route_epoch = '0;
    route_layer = '0;
    route_expert = '0;
    route_last = 1'b0;
    route_ready = 1'b1;
    observed_count = 0;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    capture_descriptor_token(16'd1, 8'd4, 10'd11);
    capture_descriptor_token(16'd2, 8'd7, 10'd12);
    accept_route_expert(10'd11, 1'b1);
    repeat (16) @(negedge clk);
    if (observed_count != 2) begin
      $fatal(1, "expected two resolved tokens, got %0d", observed_count);
    end
    if (dropped_capture_count != 0 || resolver_full) begin
      $fatal(1, "resolver unexpectedly overflowed");
    end
    $display("PASS: Router truth resolves captured descriptor tokens deterministically");
    $finish;
  end
endmodule
