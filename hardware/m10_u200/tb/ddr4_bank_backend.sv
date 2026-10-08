`timescale 1ns/1ps
/** 验证专用：U200 单 rank RDIMM + MIG + 异步 AXI 桥。
 * 所有测试权重通过 AXI 写入并逐字读回，无 force/backdoor 初始化。
 * BANK_ID 参与 stripe 数值参考；延迟测量在校准与初始化完成后进行。
 */
module ddr4_bank_backend #(
  parameter integer BANK_ID=0,
  parameter integer LAYERS=3,
  parameter bit SWITCH_FIXTURE=0,
  parameter integer H=16,
  parameter integer F=32,
  parameter integer FT=16,
  parameter integer EXPERTS=6,
  parameter realtime INIT_TIMEOUT_NS=900000
) (
  input wire core_clk,
  output logic initialized=0,
  input wire [63:0] araddr,
  input wire [7:0] arlen,
  input wire arvalid,
  output wire arready,
  output wire [127:0] rdata,
  output wire [1:0] rresp,
  output wire rlast,
  output wire rvalid,
  input wire rready
);
  import arch_package::*;
  logic sys_clk=0;
  always #1.666 sys_clk=~sys_clk;
  logic sys_rst=1;
  initial begin #100; sys_rst=0; end
  wire ui_clk, ui_reset, calibrated;
  logic ui_aresetn=0;
  always @(posedge ui_clk) ui_aresetn<=!ui_reset;
  logic core_aresetn=0;
  initial begin
    wait(calibrated && ui_aresetn);
    repeat(8) @(posedge core_clk);
    @(negedge core_clk); core_aresetn=1;
  end
  wire act_n, parity, dram_reset_n;
  wire [16:0] adr;
  wire [1:0] ba, bg;
  wire [0:0] ck_t, ck_c, cke, cs_n, odt;
  wire [71:0] dq;
  wire [17:0] dqs_t, dqs_c;
  wire [33:0] m_awaddr, m_araddr;
  wire [7:0] m_awlen, m_arlen;
  wire [3:0] m_awid, m_arid, m_bid, m_rid;
  wire [2:0] m_awsize, m_arsize, m_awprot, m_arprot;
  wire [1:0] m_awburst, m_arburst, m_bresp, m_rresp;
  wire m_awlock, m_arlock, m_awvalid, m_awready, m_wvalid, m_wready;
  wire m_bvalid, m_bready, m_arvalid, m_arready, m_rvalid, m_rready, m_rlast, m_wlast;
  wire [3:0] m_awcache, m_arcache, m_awqos, m_arqos;
  wire [127:0] m_wdata, m_rdata;
  wire [15:0] m_wstrb;
  logic [33:0] s_awaddr=0, check_araddr=0;
  logic [7:0] s_awlen=0, check_arlen=0;
  logic s_awvalid=0, s_wvalid=0, s_wlast=0, check_arvalid=0, check_rready=0;
  logic [127:0] s_wdata=0;
  wire s_awready, s_wready, s_bvalid, s_arready, s_rvalid, s_rlast;
  wire [1:0] s_bresp, s_rresp;
  wire [127:0] s_rdata;
  integer core_cycle=0;
  always @(posedge core_clk) core_cycle<=core_cycle+1;
  assign arready=initialized && s_arready;
  assign rdata=s_rdata;
  assign rresp=s_rresp;
  assign rlast=s_rlast;
  assign rvalid=initialized && s_rvalid;

  ddr_axi_cdc cdc (
    .s_axi_aclk(core_clk),
    .s_axi_aresetn(core_aresetn),
    .s_axi_awid(4'd0),
    .s_axi_awaddr(s_awaddr),
    .s_axi_awlen(s_awlen),
    .s_axi_awsize(3'd4),
    .s_axi_awburst(2'b01),
    .s_axi_awlock(1'b0),
    .s_axi_awcache(4'd0),
    .s_axi_awprot(3'd0),
    .s_axi_awregion(4'd0),
    .s_axi_awqos(4'd0),
    .s_axi_awvalid(s_awvalid),
    .s_axi_awready(s_awready),
    .s_axi_wdata(s_wdata),
    .s_axi_wstrb(16'hffff),
    .s_axi_wlast(s_wlast),
    .s_axi_wvalid(s_wvalid),
    .s_axi_wready(s_wready),
    .s_axi_bid(),
    .s_axi_bresp(s_bresp),
    .s_axi_bvalid(s_bvalid),
    .s_axi_bready(1'b1),
    .s_axi_arid(4'd0),
    .s_axi_araddr(initialized ? araddr[33:0] : check_araddr),
    .s_axi_arlen(initialized ? arlen : check_arlen),
    .s_axi_arsize(3'd4),
    .s_axi_arburst(2'b01),
    .s_axi_arlock(1'b0),
    .s_axi_arcache(4'd0),
    .s_axi_arprot(3'd0),
    .s_axi_arregion(4'd0),
    .s_axi_arqos(4'd0),
    .s_axi_arvalid(initialized ? arvalid : check_arvalid),
    .s_axi_arready(s_arready),
    .s_axi_rid(),
    .s_axi_rdata(s_rdata),
    .s_axi_rresp(s_rresp),
    .s_axi_rlast(s_rlast),
    .s_axi_rvalid(s_rvalid),
    .s_axi_rready(initialized ? rready : check_rready),
    .m_axi_aclk(ui_clk),
    .m_axi_aresetn(ui_aresetn),
    .m_axi_awid(m_awid),
    .m_axi_awaddr(m_awaddr),
    .m_axi_awlen(m_awlen),
    .m_axi_awsize(m_awsize),
    .m_axi_awburst(m_awburst),
    .m_axi_awlock(m_awlock),
    .m_axi_awcache(m_awcache),
    .m_axi_awprot(m_awprot),
    .m_axi_awregion(),
    .m_axi_awqos(m_awqos),
    .m_axi_awvalid(m_awvalid),
    .m_axi_awready(m_awready),
    .m_axi_wdata(m_wdata),
    .m_axi_wstrb(m_wstrb),
    .m_axi_wlast(m_wlast),
    .m_axi_wvalid(m_wvalid),
    .m_axi_wready(m_wready),
    .m_axi_bid(m_bid),
    .m_axi_bresp(m_bresp),
    .m_axi_bvalid(m_bvalid),
    .m_axi_bready(m_bready),
    .m_axi_arid(m_arid),
    .m_axi_araddr(m_araddr),
    .m_axi_arlen(m_arlen),
    .m_axi_arsize(m_arsize),
    .m_axi_arburst(m_arburst),
    .m_axi_arlock(m_arlock),
    .m_axi_arcache(m_arcache),
    .m_axi_arprot(m_arprot),
    .m_axi_arregion(),
    .m_axi_arqos(m_arqos),
    .m_axi_arvalid(m_arvalid),
    .m_axi_arready(m_arready),
    .m_axi_rid(m_rid),
    .m_axi_rdata(m_rdata),
    .m_axi_rresp(m_rresp),
    .m_axi_rlast(m_rlast),
    .m_axi_rvalid(m_rvalid),
    .m_axi_rready(m_rready)
  );
  ddr4_0 controller (
    .c0_sys_clk_p(sys_clk),
    .c0_sys_clk_n(~sys_clk),
    .sys_rst(sys_rst),
    .c0_init_calib_complete(calibrated),
    .c0_ddr4_ui_clk(ui_clk),
    .c0_ddr4_ui_clk_sync_rst(ui_reset),
    .c0_ddr4_aresetn(ui_aresetn),
    .c0_ddr4_act_n(act_n),
    .c0_ddr4_adr(adr),
    .c0_ddr4_ba(ba),
    .c0_ddr4_bg(bg),
    .c0_ddr4_cke(cke),
    .c0_ddr4_odt(odt),
    .c0_ddr4_cs_n(cs_n),
    .c0_ddr4_ck_t(ck_t),
    .c0_ddr4_ck_c(ck_c),
    .c0_ddr4_reset_n(dram_reset_n),
    .c0_ddr4_parity(parity),
    .c0_ddr4_dq(dq),
    .c0_ddr4_dqs_t(dqs_t),
    .c0_ddr4_dqs_c(dqs_c),
    .c0_ddr4_s_axi_awid(m_awid),
    .c0_ddr4_s_axi_awaddr(m_awaddr),
    .c0_ddr4_s_axi_awlen(m_awlen),
    .c0_ddr4_s_axi_awsize(m_awsize),
    .c0_ddr4_s_axi_awburst(m_awburst),
    .c0_ddr4_s_axi_awlock(m_awlock),
    .c0_ddr4_s_axi_awcache(m_awcache),
    .c0_ddr4_s_axi_awprot(m_awprot),
    .c0_ddr4_s_axi_awqos(m_awqos),
    .c0_ddr4_s_axi_awvalid(m_awvalid),
    .c0_ddr4_s_axi_awready(m_awready),
    .c0_ddr4_s_axi_wdata(m_wdata),
    .c0_ddr4_s_axi_wstrb(m_wstrb),
    .c0_ddr4_s_axi_wlast(m_wlast),
    .c0_ddr4_s_axi_wvalid(m_wvalid),
    .c0_ddr4_s_axi_wready(m_wready),
    .c0_ddr4_s_axi_bid(m_bid),
    .c0_ddr4_s_axi_bresp(m_bresp),
    .c0_ddr4_s_axi_bvalid(m_bvalid),
    .c0_ddr4_s_axi_bready(m_bready),
    .c0_ddr4_s_axi_arid(m_arid),
    .c0_ddr4_s_axi_araddr(m_araddr),
    .c0_ddr4_s_axi_arlen(m_arlen),
    .c0_ddr4_s_axi_arsize(m_arsize),
    .c0_ddr4_s_axi_arburst(m_arburst),
    .c0_ddr4_s_axi_arlock(m_arlock),
    .c0_ddr4_s_axi_arcache(m_arcache),
    .c0_ddr4_s_axi_arprot(m_arprot),
    .c0_ddr4_s_axi_arqos(m_arqos),
    .c0_ddr4_s_axi_arvalid(m_arvalid),
    .c0_ddr4_s_axi_arready(m_arready),
    .c0_ddr4_s_axi_rid(m_rid),
    .c0_ddr4_s_axi_rdata(m_rdata),
    .c0_ddr4_s_axi_rresp(m_rresp),
    .c0_ddr4_s_axi_rlast(m_rlast),
    .c0_ddr4_s_axi_rvalid(m_rvalid),
    .c0_ddr4_s_axi_rready(m_rready),
    .c0_ddr4_s_axi_ctrl_awvalid(1'b0),
    .c0_ddr4_s_axi_ctrl_awaddr(32'd0),
    .c0_ddr4_s_axi_ctrl_wvalid(1'b0),
    .c0_ddr4_s_axi_ctrl_wdata(32'd0),
    .c0_ddr4_s_axi_ctrl_bready(1'b1),
    .c0_ddr4_s_axi_ctrl_arvalid(1'b0),
    .c0_ddr4_s_axi_ctrl_araddr(32'd0),
    .c0_ddr4_s_axi_ctrl_rready(1'b1)
  );
  ddr4_rdimm_wrapper #(
    .MC_DQ_WIDTH(72),
    .MC_DQS_BITS(18),
    .MC_DM_WIDTH(18),
    .MC_CKE_NUM(1),
    .MC_ODT_WIDTH(1),
    .MC_ABITS(17),
    .MC_BANK_WIDTH(2),
    .MC_BANK_GROUP(2),
    .MC_CS_NUM(1),
    .MC_RANKS_NUM(1),
    .NUM_PHYSICAL_PARTS(18),
    .CALIB_EN("NO"),
    .tCK(833),
    .MEM_PART_WIDTH("x4"),
    .MC_CA_MIRROR("OFF"),
    .DDR_SIM_MODEL("MICRON"),
    .DM_DBI("NONE"),
    .MC_REG_CTRL("ON"),
    .DIMM_MODEL("RDIMM"),
    .RDIMM_SLOTS(1),
    .CONFIGURED_DENSITY(_8G)
  ) memory (
    .ddr4_act_n(act_n),
    .ddr4_addr(adr),
    .ddr4_ba(ba),
    .ddr4_bg(bg),
    .ddr4_par(parity),
    .ddr4_cke(cke),
    .ddr4_odt(odt),
    .ddr4_cs_n(cs_n),
    .ddr4_ck_t(ck_t[0]),
    .ddr4_ck_c(ck_c[0]),
    .ddr4_reset_n(dram_reset_n),
    .ddr4_dm_dbi_n(),
    .ddr4_dq(dq),
    .ddr4_dqs_t(dqs_t),
    .ddr4_dqs_c(dqs_c),
    .ddr4_alert_n(),
    .initDone(calibrated),
    .scl(),
    .sa0(),
    .sa1(),
    .sa2(),
    .sda(),
    .bfunc(),
    .vddspd()
  );

  function automatic [127:0] fixture_word(input longint unsigned local_addr);
    longint unsigned virtual_addr;
    longint unsigned local_address, relative, offset;
    integer expert, tile, within_tile, row, column;
    bit down;
    logic [127:0] value;
    virtual_addr=local_addr*4+BANK_ID*16;
    if(SWITCH_FIXTURE) begin
      local_address=virtual_addr%64'h40000000;
      down=local_address>=64'h10000000;
      relative=local_address-(down ? 64'h10000000 : 64'h01000000);
      expert=int'(relative/8388608);
      offset=relative%8388608;
      if(expert>=EXPERTS || offset>=H*F) $fatal(1,"invalid Switch DDR address %h",virtual_addr);
      value=0;
      for(integer lane=0;lane<16;lane=lane+1) begin
        tile=int'((offset+lane)/(H*FT));
        within_tile=int'((offset+lane)%(H*FT));
        row=within_tile/(down ? FT : H);
        column=within_tile%(down ? FT : H);
        if(down) begin
          if(column==row%FT) value[lane*8 +: 8]=8'(1+expert%2);
        end else if(column==(tile*FT+row)%H) value[lane*8 +: 8]=8'd1;
      end
      return value;
    end
    return 128'd1 << (((virtual_addr/16)%16)*8);
  endfunction
  task automatic write_and_check(input longint unsigned address_bytes, input integer beats);
    logic [127:0] held_data;
    logic held_last;
    @(negedge core_clk);
    s_awaddr=34'(address_bytes);
    s_awlen=8'(beats-1);
    s_awvalid=1;
    do @(posedge core_clk); while(!s_awready);
    @(negedge core_clk); s_awvalid=0;
    for(integer i=0;i<beats;i=i+1) begin
      s_wdata=fixture_word(address_bytes+16*i);
      s_wlast=i==beats-1;
      s_wvalid=1;
      do @(posedge core_clk); while(!s_wready);
      @(negedge core_clk); s_wvalid=0;
    end
    do @(posedge core_clk); while(!s_bvalid);
    if(s_bresp!=0) $fatal(1,"DDR write failed bank=%0d addr=%h",BANK_ID,address_bytes);
    @(negedge core_clk);
    check_araddr=34'(address_bytes);
    check_arlen=8'(beats-1);
    check_arvalid=1;
    do @(posedge core_clk); while(!s_arready);
    @(negedge core_clk); check_arvalid=0;
    wait(s_rvalid);
    // VALID 与 DATA 经过独立 continuous/NBA 更新；在稳定边沿采样，避免 delta-cycle 竞态。
    @(negedge core_clk);
    held_data=s_rdata;
    held_last=s_rlast;
    repeat(3) begin
      @(posedge core_clk);
      if(!s_rvalid || s_rdata!==held_data || s_rlast!==held_last || s_rresp!=0)
        $fatal(1,"DDR R channel changed under backpressure bank=%0d held=%h actual=%h valid=%b resp=%b last=%b held_last=%b",
          BANK_ID,held_data,s_rdata,s_rvalid,s_rresp,s_rlast,held_last);
    end
    @(negedge core_clk);
    check_rready=1;
    for(integer i=0;i<beats;i=i+1) begin
      do @(posedge core_clk); while(!s_rvalid);
      if(s_rdata!==fixture_word(address_bytes+16*i) || s_rresp!=0 || s_rlast!=(i==beats-1))
        $fatal(1,"DDR readback failed bank=%0d addr=%h beat=%0d expected=%h actual=%h resp=%0d last=%0d",
          BANK_ID,address_bytes,i,fixture_word(address_bytes+16*i),s_rdata,s_rresp,s_rlast);
    end
    @(negedge core_clk); check_rready=0;
  endtask
  initial begin
    wait(core_aresetn);
    $display("DDR_CALIBRATED bank=%0d cycle=%0d",BANK_ID,core_cycle);
    if(SWITCH_FIXTURE) begin
      if(H*F%64 || FT%16 || F%FT || 2*H*F>8388608) $fatal(1,"invalid Switch DDR geometry");
      for(integer l=0;l<LAYERS;l=l+1)
        for(integer e=0;e<EXPERTS;e=e+1) begin
          for(integer matrix=0;matrix<2;matrix=matrix+1)
            for(integer offset=0;offset<H*F/4;offset=offset+4096)
              write_and_check((64'(l)*64'h40000000+(matrix ? 64'h10000000 : 64'h01000000)+64'(e)*8388608)/4+64'(offset),
                (H*F/4-offset>=4096) ? 256 : (H*F/4-offset)/16);
          $display("DDR_PRELOAD_PROGRESS bank=%0d layer=%0d expert=%0d cycle=%0d",BANK_ID,l,e,core_cycle);
        end
    end else begin
      for(integer l=0;l<LAYERS;l=l+1)
        for(integer e=0;e<6;e=e+1) begin
          write_and_check((64'h10000+l*64'h100000+e*4096)/4,16);
          write_and_check((64'h20000+l*64'h100000+e*4096)/4,8);
        end
    end
    @(negedge core_clk); initialized=1;
    $display("DDR_INITIALIZED bank=%0d cycle=%0d",BANK_ID,core_cycle);
  end
  integer trace_fd;
  integer accepted_cycle=-1, first_cycle=-1, beat_index=0;
  longint unsigned accepted_address;
  integer accepted_beats;
  initial begin
    trace_fd=$fopen($sformatf("ddr_bank_%0d.csv",BANK_ID),"w");
    if(trace_fd==0) $fatal(1,"DDR trace open failed");
    $fdisplay(trace_fd,"bank,address_bytes,beats,ar_cycle,first_r_cycle,last_r_cycle");
  end
  always @(posedge core_clk) begin
    if(initialized && arvalid && arready) begin
      if(accepted_cycle>=0 || araddr[63:34]!=0) $fatal(1,"illegal DDR request bank=%0d",BANK_ID);
      accepted_cycle=core_cycle;
      first_cycle=-1;
      beat_index=0;
      accepted_address=araddr;
      accepted_beats=int'(arlen)+1;
    end
    if(rvalid && rready) begin
      if(accepted_cycle<0 || rdata!==fixture_word(accepted_address+beat_index*16) || rresp!=0)
        $fatal(1,"DDR payload mismatch bank=%0d beat=%0d",BANK_ID,beat_index);
      if(first_cycle<0) first_cycle=core_cycle;
      beat_index=beat_index+1;
      if(rlast!=(beat_index==accepted_beats)) $fatal(1,"DDR burst length mismatch");
      if(rlast) begin
        $fdisplay(trace_fd,"%0d,%0d,%0d,%0d,%0d,%0d",BANK_ID,accepted_address,
          accepted_beats,accepted_cycle,first_cycle,core_cycle);
        $fflush(trace_fd);
        accepted_cycle=-1;
      end
    end
  end
  initial begin #(INIT_TIMEOUT_NS); if(!initialized) $fatal(1,"DDR calibration/preload watchdog bank=%0d",BANK_ID); end
  integer dram_reads=0, dram_activates=0, dram_refreshes=0, dram_precharges=0;
  always @(posedge ck_t[0])
    if(initialized && !cs_n[0] && act_n && adr[16:14]==3'b101) dram_reads<=dram_reads+1;
  always @(posedge ck_t[0])
    if(initialized && !cs_n[0] && !act_n) dram_activates<=dram_activates+1;
  always @(posedge ck_t[0])
    if(initialized && !cs_n[0] && act_n && adr[16:14]==3'b001) dram_refreshes<=dram_refreshes+1;
  always @(posedge ck_t[0])
    if(initialized && !cs_n[0] && act_n && adr[16:14]==3'b010) dram_precharges<=dram_precharges+1;
  final begin
    $display("DDR_DRAM_SUMMARY bank=%0d reads=%0d activates=%0d refreshes=%0d precharges=%0d",
      BANK_ID,dram_reads,dram_activates,dram_refreshes,dram_precharges);
    $fclose(trace_fd);
  end
endmodule
