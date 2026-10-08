`timescale 1ns/1ps
/** Six-layer permission gates and restart after a completed sequence. */
module tb_sequence_controller;
  logic clk = 0;
  logic rst_n = 0;
  logic start = 0;
  logic layer_done = 0;
  logic sequence_active;
  logic sequence_done;
  logic [2:0] current_layer;
  logic compute_ready;
  logic bandwidth_ready;
  always #5 clk = ~clk;
  sequence_controller dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .layer_done(layer_done),
    .compute_block_cycles(32'd2),
    .bandwidth_block_cycles(32'd4),
    .sequence_active(sequence_active),
    .sequence_done(sequence_done),
    .current_layer(current_layer),
    .layer_start_pulse(),
    .compute_ready(compute_ready),
    .bandwidth_ready(bandwidth_ready)
  );
  initial begin
    repeat (3) @(negedge clk);
    rst_n = 1;
    for (integer sequence_index = 0; sequence_index < 2; sequence_index = sequence_index + 1) begin
      start = 1;
      @(negedge clk);
      start = 0;
      if (!sequence_active || sequence_done || current_layer != 0) $fatal(1, "sequence start/restart failed");
      for (integer layer = 0; layer < 6; layer = layer + 1) begin
        if (current_layer != layer) $fatal(1, "layer rollover failed");
        layer_done = 1;
        @(negedge clk);
        layer_done = 0;
        if (layer < 5) begin
          if (compute_ready || bandwidth_ready) $fatal(1, "transition permission opened early");
          repeat (2) @(negedge clk);
          if (!compute_ready || bandwidth_ready) $fatal(1, "independent compute/bandwidth gate timing failed");
          wait (current_layer == layer+1);
          @(negedge clk);
        end
      end
      if (!sequence_done || sequence_active) $fatal(1, "sequence completion failed");
    end
    $display("PASS: six-layer sequence gates and second-sequence restart");
    $finish;
  end
  initial begin
    #10000;
    $fatal(1, "sequence test timeout");
  end
endmodule
