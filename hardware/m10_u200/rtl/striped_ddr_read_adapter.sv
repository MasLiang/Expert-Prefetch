`timescale 1ns/1ps
/** 四 bank 并行读：虚拟连续地址以 16 byte stripe 分布到四个 DDR。
 * bank=(virtual_address/16)%4，local_address=(virtual_address/64)*16。
 * 一组四个 128-bit beats 合并成 512-bit OCM 写回。各 bank 独立 AR/R
 * handshake 和一级弹性缓冲；最多四个 burst 同时 outstanding。
 * 本接口是 core-clock AXI，并非已绑定的 U200 DDR controller/CDC。
 */
module striped_ddr_read_adapter #(
  parameter integer AXI_ADDR_W = 64,
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
  output logic [511:0] payload_data,
  output logic [63:0] payload_keep,
  output logic payload_last,
  output logic [31:0] response_error_count,
  output logic [4*AXI_ADDR_W-1:0] m_ddr_araddr,
  output logic [31:0] m_ddr_arlen,
  output logic [11:0] m_ddr_arsize,
  output logic [7:0] m_ddr_arburst,
  output logic [3:0] m_ddr_arvalid,
  input logic [3:0] m_ddr_arready,
  input logic [511:0] m_ddr_rdata,
  input logic [7:0] m_ddr_rresp,
  input logic [3:0] m_ddr_rlast,
  input logic [3:0] m_ddr_rvalid,
  output logic [3:0] m_ddr_rready
);
  logic [3:0] ar_pending;
  logic [3:0] buffer_full;
  logic [511:0] buffer_data;
  logic [8:0] beats_left [0:3];
  logic [AXI_ADDR_W-1:0] local_address;
  logic [7:0] arlen_hold;
  logic [31:0] bytes_left;
  logic [31:0] offset_hold;
  logic [15:0] slot_hold;
  logic [15:0] epoch_hold;
  logic [GENERATION_W-1:0] generation_hold;
  logic [31:0] errors_this_cycle;
  wire take_payload = payload_valid && payload_ready;
  // 首拍直通 AR；未握手的 bank 转为 registered pending，地址保持不变。
  wire launch = rst_n && burst_start && !busy;
  wire [63:0] virtual_address = active_descriptor.external_addr +
    64'(active_descriptor.byte_count) - 64'(active_remaining_bytes);
  assign payload_valid = busy && &buffer_full;
  assign payload_data = buffer_data;
  assign payload_keep = '1;
  assign payload_last = bytes_left == 64;
  assign payload_offset = offset_hold;
  assign payload_slot = slot_hold;
  assign payload_epoch = epoch_hold;
  assign payload_slot_generation = generation_hold;

  always_comb begin
    errors_this_cycle = 0;
    for (integer bank = 0; bank < 4; bank = bank + 1) begin
      if (m_ddr_rvalid[bank] && m_ddr_rready[bank] &&
          (m_ddr_rresp[bank*2 +: 2] != 0 || m_ddr_rlast[bank] != (beats_left[bank] == 1)))
        errors_this_cycle = errors_this_cycle + 1;
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) response_error_count <= 0;
    else response_error_count <= response_error_count + errors_this_cycle;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) busy <= 0;
    else if (burst_start && !busy) busy <= 1;
    else if (take_payload && payload_last) busy <= 0;
  end
  // 最后 OCM 写握手就是完成边界；不额外注册一拍再通知 DMA。
  assign burst_complete = rst_n && take_payload && payload_last;
  always_ff @(posedge clk) begin
    if (burst_start && !busy) local_address <= AXI_ADDR_W'(virtual_address >> 2);
  end
  always_ff @(posedge clk) begin
    if (burst_start && !busy) arlen_hold <= 8'((burst_bytes >> 6) - 1);
  end
  always_ff @(posedge clk) begin
    if (burst_start && !busy) slot_hold <= active_descriptor.ocm_slot;
  end
  always_ff @(posedge clk) begin
    if (burst_start && !busy) epoch_hold <= active_descriptor.epoch_id;
  end
  always_ff @(posedge clk) begin
    if (burst_start && !busy) generation_hold <= active_slot_generation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) bytes_left <= 0;
    else if (burst_start && !busy) bytes_left <= burst_bytes;
    else if (take_payload) bytes_left <= bytes_left - 64;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) offset_hold <= 0;
    else if (burst_start && !busy) offset_hold <= active_descriptor.byte_count - active_remaining_bytes;
    else if (take_payload) offset_hold <= offset_hold + 64;
  end
  for (genvar bank = 0; bank < 4; bank = bank + 1) begin : banks
    assign m_ddr_araddr[bank*AXI_ADDR_W +: AXI_ADDR_W] = launch ? AXI_ADDR_W'(virtual_address >> 2) : local_address;
    assign m_ddr_arlen[bank*8 +: 8] = launch ? 8'((burst_bytes >> 6) - 1) : arlen_hold;
    assign m_ddr_arsize[bank*3 +: 3] = 3'd4;
    assign m_ddr_arburst[bank*2 +: 2] = 2'b01;
    assign m_ddr_arvalid[bank] = launch || ar_pending[bank];
    assign m_ddr_rready[bank] = busy && !ar_pending[bank] && beats_left[bank] != 0 &&
      (!buffer_full[bank] || take_payload);
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) ar_pending[bank] <= 0;
      else if (launch) ar_pending[bank] <= !m_ddr_arready[bank];
      else if (m_ddr_arvalid[bank] && m_ddr_arready[bank]) ar_pending[bank] <= 0;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) beats_left[bank] <= 0;
      else if (burst_start && !busy) beats_left[bank] <= 9'(burst_bytes >> 6);
      else if (m_ddr_rvalid[bank] && m_ddr_rready[bank]) beats_left[bank] <= beats_left[bank] - 1'b1;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) buffer_full[bank] <= 0;
      else if (burst_start && !busy) buffer_full[bank] <= 0;
      else if (m_ddr_rvalid[bank] && m_ddr_rready[bank]) buffer_full[bank] <= 1;
      else if (take_payload) buffer_full[bank] <= 0;
    end
    always_ff @(posedge clk) begin
      if (m_ddr_rvalid[bank] && m_ddr_rready[bank])
        buffer_data[bank*128 +: 128] <= m_ddr_rdata[bank*128 +: 128];
    end
  end
  // 输入是整组 stripes；flush 必须 drain，不复位正在访问 DDR 的 ownership。
  always @(posedge clk) begin
    if (rst_n && burst_start &&
        (busy || burst_bytes == 0 || burst_bytes > 4096 || burst_bytes[5:0] != 0 ||
         virtual_address[5:0] != 0)) $error("invalid striped DDR command");
  end
endmodule
