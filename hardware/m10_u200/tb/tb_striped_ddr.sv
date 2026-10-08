`timescale 1ns/1ps
/** 四 DDR -> 512-bit join -> OCM -> 128-bit synchronous reads。
 * 独立 AR/R stalls，payload 背压，full/partial burst，RRESP 错误计数。
 */
module tb_striped_ddr;
  import tile_prefetch_package::*;
  logic clk=0;
  always #5 clk=~clk;
  logic rst_n=0, start=0, complete, busy, payload_valid, payload_ready;
  tile_descriptor_t descriptor;
  logic [31:0] remaining=256, burst_bytes=256, errors, offset, discarded;
  logic [511:0] payload, rdata;
  logic [63:0] keep;
  logic [15:0] slot, epoch;
  logic [7:0] generation;
  logic last;
  logic [255:0] araddr;
  logic [31:0] arlen;
  logic [11:0] arsize;
  logic [7:0] arburst, rresp=0;
  logic [3:0] arvalid, arready, rvalid=0, rready, rlast=0;
  logic [3:0] active=0;
  integer cursor[0:3], left[0:3];
  integer cycle=0, b, group, lane, phase=0, payload_count=0;
  logic bind_valid=0, read_valid=0;
  logic [7:0] read_offset=0;
  logic [127:0] read_data;
  striped_ddr_read_adapter dut (
    .clk(clk),
    .rst_n(rst_n),
    .burst_start(start),
    .active_descriptor(descriptor),
    .active_slot_generation(8'd9),
    .burst_bytes(burst_bytes),
    .active_remaining_bytes(remaining),
    .burst_complete(complete),
    .busy(busy),
    .payload_valid(payload_valid),
    .payload_ready(payload_ready),
    .payload_slot(slot),
    .payload_epoch(epoch),
    .payload_slot_generation(generation),
    .payload_offset(offset),
    .payload_data(payload),
    .payload_keep(keep),
    .payload_last(last),
    .response_error_count(errors),
    .m_ddr_araddr(araddr),
    .m_ddr_arlen(arlen),
    .m_ddr_arsize(arsize),
    .m_ddr_arburst(arburst),
    .m_ddr_arvalid(arvalid),
    .m_ddr_arready(arready),
    .m_ddr_rdata(rdata),
    .m_ddr_rresp(rresp),
    .m_ddr_rlast(rlast),
    .m_ddr_rvalid(rvalid),
    .m_ddr_rready(rready)
  );
  ocm_payload_ram #(
    .SLOT_COUNT(2),
    .BYTES_PER_SLOT(256),
    .DATA_W(512),
    .READ_DATA_W(128)
  ) ram (
    .clk(clk),
    .rst_n(rst_n),
    .slot_bind_valid(bind_valid),
    .slot_bind_slot(1'b0),
    .slot_bind_epoch(16'd7),
    .slot_bind_generation(8'd9),
    .slot_release_valid(1'b0),
    .slot_release_slot(1'b0),
    .slot_release_epoch(16'd0),
    .slot_release_generation(8'd0),
    .slot_release2_valid(1'b0),
    .slot_release2_slot(1'b0),
    .slot_release2_epoch(16'd0),
    .slot_release2_generation(8'd0),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch(16'd0),
    .dma_write_valid(payload_valid && payload_ready),
    .dma_write_slot(slot[0]),
    .dma_write_offset(offset[7:0]),
    .dma_write_data(payload),
    .dma_write_strobe(keep),
    .dma_write_epoch(epoch),
    .dma_write_generation(generation),
    .dma_write_accept(),
    .read_valid(read_valid),
    .read_slot(1'b0),
    .read_offset(read_offset),
    .read_data(read_data),
    .discarded_write_count(discarded)
  );
  assign payload_ready=cycle%5!=0 && cycle%5!=1;
  // 首拍 AR 直通、背压稳定和 OCM 最后写同拍完成；不能以删除断言换取 PASS。
  logic [3:0] ar_stalled_previous;
  logic [255:0] araddr_previous;
  logic [31:0] arlen_previous;
  always @(posedge clk) ar_stalled_previous <= rst_n ? arvalid & ~arready : 0;
  always @(posedge clk) araddr_previous <= araddr;
  always @(posedge clk) arlen_previous <= arlen;
  for(genvar bank=0;bank<4;bank=bank+1) begin : ar_stability
    always @(posedge clk) begin
      if(rst_n && ar_stalled_previous[bank] &&
          (!arvalid[bank] || araddr[bank*64 +: 64]!=araddr_previous[bank*64 +: 64] ||
           arlen[bank*8 +: 8]!=arlen_previous[bank*8 +: 8])) $fatal(1,"AR changed under stall");
    end
  end
  always @(posedge clk) begin
    if(rst_n && start && !busy && arvalid!=4'b1111) $fatal(1,"AR launch bubble");
    if(rst_n && complete!=(payload_valid && payload_ready && last)) $fatal(1,"completion bubble");
  end
  always_comb begin
    for(integer bank=0;bank<4;bank=bank+1) arready[bank]=!active[bank] && cycle%(bank+2)!=0;
  end
  always @(posedge clk) begin
    cycle<=cycle+1;
    if(payload_valid && payload_ready) begin
      if(slot!=0 || epoch!=7 || generation!=9 || keep!='1) $fatal(1,"payload tags");
      for(integer byte_index=0;byte_index<64;byte_index=byte_index+1)
        if(payload[byte_index*8 +: 8]!=8'(int'(offset)+byte_index)) $fatal(1,"stripe join");
      payload_count<=payload_count+1;
    end
    for(b=0;b<4;b=b+1) begin
      if(arvalid[b] && arready[b]) begin
        if(arsize[b*3 +: 3]!=4 || arburst[b*2 +: 2]!=1 ||
           araddr[b*64 +: 64]!=(64'h1000+256-remaining)/4) $fatal(1,"AR geometry");
        active[b]<=1;
        left[b]<=int'(arlen[b*8 +: 8])+1;
        cursor[b]<=256-int'(remaining)+b*16;
      end
      if(rvalid[b] && rready[b]) begin
        rvalid[b]<=0;
        cursor[b]<=cursor[b]+64;
        left[b]<=left[b]-1;
        if(left[b]==1) active[b]<=0;
      end else if(active[b] && !rvalid[b] && cycle%(b+3)!=0) begin
        rvalid[b]<=1;
        rlast[b]<=left[b]==1;
        rresp[b*2 +: 2]<=phase==2 && b==2 ? 2'b10 : 2'b00;
        for(integer j=0;j<16;j=j+1) rdata[b*128+j*8 +: 8]<=8'(cursor[b]+j);
      end
    end
  end
  initial begin
    descriptor='0; descriptor.external_addr=64'h1000; descriptor.byte_count=256;
    descriptor.epoch_id=7; descriptor.ocm_slot=0;
    repeat(3) @(negedge clk); rst_n=1;
    bind_valid=1; @(negedge clk); bind_valid=0;
    start=1; @(negedge clk); start=0;
    // ready banks 已在首拍握手；其余 bank 的 pending AR 必须保留。
    wait(complete); @(posedge clk); @(negedge clk);
    if(payload_count!=4 || errors!=0 || busy || discarded!=0) $fatal(1,"conservation");
    for(group=0;group<16;group=group+1) begin
      read_valid=1; read_offset=8'(group*16); @(negedge clk);
      for(lane=0;lane<16;lane=lane+1)
        if(read_data[lane*8 +: 8]!=8'(group*16+lane)) $fatal(1,"wide RAM read selection");
    end
    read_valid=0;
    phase=1; remaining=128; burst_bytes=64;
    start=1; @(negedge clk); start=0; wait(complete); @(posedge clk); @(negedge clk);
    if(payload_count!=5 || errors!=0) $fatal(1,"partial burst conservation");
    phase=2; remaining=64;
    start=1; @(negedge clk); start=0; wait(complete); @(posedge clk); @(negedge clk);
    if(payload_count!=6 || errors!=1) $fatal(1,"response error count");
    $display("PASS: concurrent four-bank DDR, independent stalls, 384 byte conservation, wide OCM");
    $finish;
  end
  initial begin #1000000; $fatal(1,"timeout"); end
endmodule
