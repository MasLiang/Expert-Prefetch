`timescale 1ns/1ps
/**
 * 带 slot ownership tag 的可综合 OCM payload RAM。
 * DMA 写回必须匹配 live `(epoch_id, slot_generation)`；不匹配的 beat 被丢弃。
 */
module ocm_payload_ram #(
  parameter integer SLOT_COUNT = 32,
  parameter integer BYTES_PER_SLOT = 1024,
  parameter integer DATA_W = 128,
  parameter integer READ_DATA_W = 128,
  parameter integer GENERATION_W = 8,
  parameter integer SLOT_W = (SLOT_COUNT <= 1) ? 1 : $clog2(SLOT_COUNT),
  parameter integer OFFSET_W = (BYTES_PER_SLOT <= 1) ? 1 : $clog2(BYTES_PER_SLOT)
) (
  input logic clk, input logic rst_n,
  input logic slot_bind_valid, input logic [SLOT_W-1:0] slot_bind_slot,
  input logic [15:0] slot_bind_epoch, input logic [GENERATION_W-1:0] slot_bind_generation,
  input logic slot_release_valid, input logic [SLOT_W-1:0] slot_release_slot,
  input logic [15:0] slot_release_epoch, input logic [GENERATION_W-1:0] slot_release_generation,
  input logic slot_release2_valid,
  input logic [SLOT_W-1:0] slot_release2_slot,
  input logic [15:0] slot_release2_epoch,
  input logic [GENERATION_W-1:0] slot_release2_generation,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic dma_write_valid, input logic [SLOT_W-1:0] dma_write_slot,
  input logic [OFFSET_W-1:0] dma_write_offset, input logic [DATA_W-1:0] dma_write_data,
  input logic [DATA_W/8-1:0] dma_write_strobe, input logic [15:0] dma_write_epoch,
  input logic [GENERATION_W-1:0] dma_write_generation, output logic dma_write_accept,
  input logic read_valid, input logic [SLOT_W-1:0] read_slot,
  input logic [OFFSET_W-1:0] read_offset, output logic [READ_DATA_W-1:0] read_data,
  output logic [31:0] discarded_write_count
);
  logic valid_mem [0:SLOT_COUNT-1];
  logic [15:0] epoch_mem [0:SLOT_COUNT-1];
  logic [GENERATION_W-1:0] gen_mem [0:SLOT_COUNT-1];
  localparam integer LANES = DATA_W/8;
  localparam integer WORDS_PER_SLOT = BYTES_PER_SLOT/LANES;
  localparam integer WORD_COUNT = SLOT_COUNT*WORDS_PER_SLOT;
  localparam integer READ_GROUPS = DATA_W/READ_DATA_W;
  localparam integer READ_GROUP_W = READ_GROUPS <= 1 ? 1 : $clog2(READ_GROUPS);
  logic [DATA_W-1:0] read_wide;
  logic [READ_GROUP_W-1:0] read_group_hold;
  assign read_data = read_wide[int'(read_group_hold)*READ_DATA_W +: READ_DATA_W];
  always_ff @(posedge clk) begin
    if (read_valid) read_group_hold <= READ_GROUP_W'((32'(read_offset) % LANES)/(READ_DATA_W/8));
  end
  integer i;
  wire [31:0] write_word = dma_write_slot * WORDS_PER_SLOT + 32'(dma_write_offset)/LANES;
  wire [31:0] read_word = read_slot * WORDS_PER_SLOT + 32'(read_offset)/LANES;
  assign dma_write_accept = dma_write_valid && valid_mem[dma_write_slot] &&
    epoch_mem[dma_write_slot] == dma_write_epoch && gen_mem[dma_write_slot] == dma_write_generation &&
    32'(dma_write_offset) + DATA_W/8 - 1 < BYTES_PER_SLOT && 32'(dma_write_offset) % LANES == 0;
  initial begin
    if (BYTES_PER_SLOT % LANES != 0 || DATA_W % READ_DATA_W != 0 || READ_DATA_W < 8)
      $fatal(1, "OCM slot and read width must contain whole payload groups");
  end
  generate for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : payload_lane
    (* ram_style = "block" *) logic [7:0] memory [0:WORD_COUNT-1];
    always_ff @(posedge clk) begin
      if (dma_write_accept && dma_write_strobe[lane]) memory[write_word] <= dma_write_data[lane*8 +: 8];
    end
    always_ff @(posedge clk) begin
      if (read_valid && valid_mem[read_slot] && 32'(read_offset) + READ_DATA_W/8 <= BYTES_PER_SLOT)
        read_wide[lane*8 +: 8] <= memory[read_word];
    end
  end endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      discarded_write_count <= 0;
      for (i = 0; i < SLOT_COUNT; i = i + 1) begin
        valid_mem[i] <= 0;
        epoch_mem[i] <= 0;
        gen_mem[i] <= 0;
      end
    end else begin
      if (slot_bind_valid) begin
        valid_mem[slot_bind_slot] <= 1;
        epoch_mem[slot_bind_slot] <= slot_bind_epoch;
        gen_mem[slot_bind_slot] <= slot_bind_generation;
      end
      if (slot_release_valid && valid_mem[slot_release_slot] &&
          epoch_mem[slot_release_slot] == slot_release_epoch &&
          gen_mem[slot_release_slot] == slot_release_generation) begin
        valid_mem[slot_release_slot] <= 0;
      end
      if (slot_release2_valid === 1'b1 && valid_mem[slot_release2_slot] &&
          epoch_mem[slot_release2_slot] == slot_release2_epoch &&
          gen_mem[slot_release2_slot] == slot_release2_generation)
        valid_mem[slot_release2_slot] <= 0;
      if (soft_flush_valid === 1'b1) begin
        for (i = 0; i < SLOT_COUNT; i = i + 1)
          if (epoch_mem[i] == soft_flush_epoch) valid_mem[i] <= 0;
      end
      if (dma_write_valid) begin
        if (dma_write_accept) begin
          /* Payload lane memories implement the accepted byte strobes. */
        end else begin
          discarded_write_count <= discarded_write_count + 1;
        end
      end
    end
  end
endmodule
