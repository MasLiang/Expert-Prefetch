`timescale 1ns/1ps
/**
 * 单 outstanding AXI4 read adapter。
 *
 * 本模块把 DMA arbiter 的一个 burst 变成 AXI4 read address 与 read-data
 * transaction；每个返回 beat 都带回当前 descriptor 的 OCM ownership tag。
 * 它不假设 U200 的时钟、DDR channel 数或实现带宽，只有 AXI ready/valid
 * backpressure 决定实际传输时序。
 */
module axi_read_adapter #(
  parameter integer AXI_ADDR_W = 64,
  parameter integer AXI_DATA_W = 128,
  parameter integer GENERATION_W = 8
) (
  input logic clk,
  input logic rst_n,
  input logic burst_start,
  input tile_prefetch_package::tile_descriptor_t active_descriptor,
  input logic [GENERATION_W-1:0] active_slot_generation,
  input logic [31:0] burst_bytes,
  input logic [31:0] active_remaining_bytes,
  output logic burst_complete,
  output logic busy,
  output logic payload_valid,
  input logic payload_ready,
  output logic [15:0] payload_slot,
  output logic [15:0] payload_epoch,
  output logic [GENERATION_W-1:0] payload_slot_generation,
  output logic [31:0] payload_offset,
  output logic [AXI_DATA_W-1:0] payload_data,
  output logic [AXI_DATA_W/8-1:0] payload_keep,
  output logic payload_last,
  output logic [31:0] response_error_count,
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0] m_axi_arlen,
  output logic [2:0] m_axi_arsize,
  output logic [1:0] m_axi_arburst,
  output logic m_axi_arvalid,
  input logic m_axi_arready,
  input logic [AXI_DATA_W-1:0] m_axi_rdata,
  input logic [1:0] m_axi_rresp,
  input logic m_axi_rlast,
  input logic m_axi_rvalid,
  output logic m_axi_rready
);
  localparam integer DATA_BYTES = AXI_DATA_W / 8;
  localparam integer BYTE_SHIFT = $clog2(DATA_BYTES);

  typedef enum logic [1:0] {IDLE, SEND_AR, RECEIVE_R} state_t;
  state_t state;
  logic [AXI_ADDR_W-1:0] address_hold;
  logic [15:0] slot_hold;
  logic [15:0] epoch_hold;
  logic [GENERATION_W-1:0] generation_hold;
  logic [31:0] offset_hold;
  logic [31:0] bytes_left;
  logic [8:0] beats_left;
  logic [7:0] arlen_hold;
  logic [AXI_DATA_W/8-1:0] final_keep;

  initial begin
    if (AXI_DATA_W != 128) begin
      $error("M10.1 payload ABI fixes AXI_DATA_W to 128 bits");
    end
  end

  always_comb begin
    busy = state != IDLE;
    m_axi_araddr = address_hold;
    m_axi_arlen = arlen_hold;
    m_axi_arsize = BYTE_SHIFT[2:0];
    m_axi_arburst = 2'b01;
    m_axi_arvalid = state == SEND_AR;
    m_axi_rready = state == RECEIVE_R && payload_ready;
    payload_valid = state == RECEIVE_R && m_axi_rvalid;
    payload_slot = slot_hold;
    payload_epoch = epoch_hold;
    payload_slot_generation = generation_hold;
    payload_offset = offset_hold;
    payload_data = m_axi_rdata;
    payload_last = beats_left == 1;
    if (bytes_left >= DATA_BYTES) begin
      payload_keep = {DATA_BYTES{1'b1}};
    end else begin
      payload_keep = final_keep;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      address_hold <= '0;
      slot_hold <= '0;
      epoch_hold <= '0;
      generation_hold <= '0;
      offset_hold <= '0;
      bytes_left <= '0;
      beats_left <= '0;
      arlen_hold <= '0;
      final_keep <= '0;
      burst_complete <= 1'b0;
      response_error_count <= '0;
    end else begin
      burst_complete <= 1'b0;
      case (state)
        IDLE: begin
          if (burst_start) begin
            if (burst_bytes == 0 || burst_bytes > 4096 ||
                ((active_descriptor.external_addr + 64'(active_descriptor.byte_count) -
                  64'(active_remaining_bytes)) % 64'(DATA_BYTES)) != 0)
              $error("AXI read command must be nonzero, aligned and at most 256 beats");
            address_hold <= active_descriptor.external_addr +
              (64'(active_descriptor.byte_count) - 64'(active_remaining_bytes));
            slot_hold <= active_descriptor.ocm_slot;
            epoch_hold <= active_descriptor.epoch_id;
            generation_hold <= active_slot_generation;
            offset_hold <= active_descriptor.byte_count - active_remaining_bytes;
            bytes_left <= burst_bytes;
            beats_left <= 9'((burst_bytes + DATA_BYTES - 1) >> BYTE_SHIFT);
            arlen_hold <= 8'(((burst_bytes + DATA_BYTES - 1) >> BYTE_SHIFT) - 1'b1);
            if (burst_bytes % DATA_BYTES == 0) begin
              final_keep <= {DATA_BYTES{1'b1}};
            end else begin
              final_keep <= ({DATA_BYTES{1'b1}} >> (DATA_BYTES - (burst_bytes % DATA_BYTES)));
            end
            state <= SEND_AR;
          end
        end
        SEND_AR: begin
          if (m_axi_arvalid && m_axi_arready) begin
            state <= RECEIVE_R;
          end
        end
        RECEIVE_R: begin
          if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rresp != 2'b00 || m_axi_rlast != (beats_left == 1)) begin
              response_error_count <= response_error_count + 1'b1;
            end
            if (beats_left == 1) begin
              burst_complete <= 1'b1;
              state <= IDLE;
            end else begin
              beats_left <= beats_left - 1'b1;
              bytes_left <= bytes_left - DATA_BYTES;
              offset_hold <= offset_hold + DATA_BYTES;
            end
          end
        end
        default: begin
          state <= IDLE;
        end
      endcase
    end
  end
endmodule
