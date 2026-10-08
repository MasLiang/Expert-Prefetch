`timescale 1ns/1ps
/** Prefix/any-order tile completion scoreboard with exactly-once accumulation. */
module partial_output_accumulator #(
  parameter integer MAX_TILES = 16,
  parameter integer TILE_W = (MAX_TILES <= 1) ? 1 : $clog2(MAX_TILES)
) (
  input logic clk,input logic rst_n,input logic start_valid,input logic [TILE_W:0] required_tile_count,
  input logic tile_done_valid,input logic [TILE_W-1:0] tile_id,input logic [31:0] tile_partial_sum,
  input logic [31:0] combine_cycles,output logic tile_done_ready,output logic combine_busy,
  output logic combine_done,output logic duplicate_tile,output logic [31:0] accumulated_sum,
  output logic [MAX_TILES-1:0] complete_bitmap
);
  logic active;
  logic [TILE_W:0] required_count;
  logic [TILE_W:0] complete_count;
  logic [31:0] remain;

  assign tile_done_ready = active && !combine_busy;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active <= 0;
      combine_busy <= 0;
      combine_done <= 0;
      duplicate_tile <= 0;
      accumulated_sum <= 0;
      complete_bitmap <= 0;
      required_count <= 0;
      complete_count <= 0;
      remain <= 0;
    end
    else begin
      combine_done <= 0;
      duplicate_tile <= 0;
      if (start_valid && !active && !combine_busy) begin
        active <= 1;
        required_count <= required_tile_count;
        complete_count <= 0;
        accumulated_sum <= 0;
        complete_bitmap <= 0;
      end
      if (tile_done_valid && tile_done_ready) begin
        if (complete_bitmap[tile_id]) begin
          duplicate_tile <= 1;
        end else begin
          complete_bitmap[tile_id] <= 1;
          accumulated_sum <= accumulated_sum + tile_partial_sum;
          complete_count <= complete_count + 1;
          if (complete_count + 1 >= required_count) begin
            active <= 0;
            combine_busy <= 1;
            remain <= combine_cycles;
          end
        end
      end
      if (combine_busy) begin
        if (remain <= 1) begin
          combine_busy <= 0;
          combine_done <= 1;
        end else begin
          remain <= remain - 1;
        end
      end
    end
  end
endmodule
