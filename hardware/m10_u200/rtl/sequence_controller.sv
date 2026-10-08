`timescale 1ns/1ps
/**
 * Shared six-layer sequence control. After a layer completes, compute and DMA
 * permissions for the next layer open at independently configured delays.
 */
module sequence_controller #(
  parameter integer LAYER_COUNT = 6,
  parameter integer LAYER_BITS = (LAYER_COUNT <= 1) ? 1 : $clog2(LAYER_COUNT)
) (
  input logic clk,
  input logic rst_n,
  input logic start,
  input logic layer_done,
  input logic [31:0] compute_block_cycles,
  input logic [31:0] bandwidth_block_cycles,
  output logic sequence_active,
  output logic sequence_done,
  output logic [LAYER_BITS-1:0] current_layer,
  output logic layer_start_pulse,
  output logic compute_ready,
  output logic bandwidth_ready
);
  logic [31:0] compute_count, bandwidth_count;
  logic waiting_transition;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sequence_active <= 0;
      sequence_done <= 0;
      current_layer <= 0;
      layer_start_pulse <= 0;
      compute_ready <= 0;
      bandwidth_ready <= 0;
      compute_count <= 0;
      bandwidth_count <= 0;
      waiting_transition <= 0;
    end else begin
      layer_start_pulse <= 0;
      if (start && !sequence_active) begin
        sequence_active <= 1;
        sequence_done <= 0;
        current_layer <= 0;
        compute_ready <= 1;
        bandwidth_ready <= 1;
        layer_start_pulse <= 1;
      end else if (sequence_active && !waiting_transition && layer_done) begin
        if (current_layer == LAYER_COUNT - 1) begin
          sequence_active <= 0;
          sequence_done <= 1;
          compute_ready <= 0;
          bandwidth_ready <= 0;
        end else begin
          waiting_transition <= 1;
          compute_ready <= (compute_block_cycles == 0);
          bandwidth_ready <= (bandwidth_block_cycles == 0);
          compute_count <= compute_block_cycles;
          bandwidth_count <= bandwidth_block_cycles;
        end
      end else if (sequence_active && waiting_transition) begin
        if (!compute_ready) begin
          if (compute_count <= 1) begin
            compute_ready <= 1;
          end else begin
            compute_count <= compute_count - 1;
          end
        end
        if (!bandwidth_ready) begin
          if (bandwidth_count <= 1) begin
            bandwidth_ready <= 1;
          end else begin
            bandwidth_count <= bandwidth_count - 1;
          end
        end
        if ((compute_ready || compute_count <= 1) && (bandwidth_ready || bandwidth_count <= 1)) begin
          waiting_transition <= 0;
          current_layer <= current_layer + 1'b1;
          layer_start_pulse <= 1;
        end
      end
    end
  end
endmodule
