`timescale 1ns/1ps
/** 完整 Switch-base MoE 尺寸。所有 dense payload/MAC 均实际执行。
 * 测试权重为确定性稀疏数值 fixture，不是训练 checkpoint；没有跳零计算。
 * 默认四 bank behavioral AXI responder；M10_SWITCH_DDR4 使用 MIG/RDIMM。无 force/backdoor。
 */
module tb_switch_full_size #(
  parameter integer H=768,
  parameter integer F=3072,
  parameter integer FT=128,
  parameter integer EXPERTS=8,
  parameter integer LAYERS=6,
  parameter bit WHOLE=0,
  parameter integer SLOT_BYTES=131072,
  parameter integer SLOTS_PER_BANK=32
);
  localparam integer TILES=F/FT;
  localparam integer FRAGMENT=H*FT;
  localparam integer EXPERT_BYTES=2*H*F;
  localparam integer REQUANT_SHIFT=$clog2(TILES);
  localparam longint unsigned LAYER_STRIDE=64'h40000000;
  localparam longint unsigned UP_BASE=64'h01000000;
  localparam longint unsigned DOWN_BASE=64'h10000000;
  localparam integer EXPERT_STRIDE=8388608;
  logic clk=0;
  always #2.5 clk=~clk;
  logic rst_n=0;
  logic sequence_start=0;
  wire start_ready, sequence_done, sequence_busy;
  wire [2:0] current_layer;
  logic [2:0] layer=0;
  logic begin_valid=0;
  wire begin_ready;
  logic candidate_valid=0;
  wire candidate_ready;
  logic reload=0, weight_valid=0, weight_last=0;
  wire weight_ready, weights_loaded, weight_error;
  logic [31:0] weight_index=0;
  logic [127:0] weight_data=0;
  logic future_begin=0, future_valid=0;
  wire future_begin_ready, future_ready;
  logic [2:0] future_layer=1;
  logic [4:0] future_rank=0;
  logic [9:0] future_expert=0;
  logic [11:0] future_confidence=0;
  wire result_valid, compute_done, router_done, parser_error, config_error, transaction_error;
  wire [H*32-1:0] result_y, sequence_y;
  wire [LAYERS-1:0] weights_done;
  wire [255:0] araddr;
  wire [31:0] arlen;
  wire [3:0] arvalid, rready;
`ifdef M10_SWITCH_DDR4
  wire [3:0] arready, rvalid, rlast, ddr_initialized;
  wire [511:0] rdata;
  wire [7:0] rresp;
  for(genvar bank=0;bank<4;bank=bank+1) begin : ddr_banks
    ddr4_bank_backend #(
      .BANK_ID(bank),
      .LAYERS(LAYERS),
      .SWITCH_FIXTURE(1'b1),
      .H(H),
      .F(F),
      .FT(FT),
      .EXPERTS(EXPERTS),
      .INIT_TIMEOUT_NS(1000000000)
    ) memory (
      .core_clk(clk),
      .initialized(ddr_initialized[bank]),
      .araddr(araddr[bank*64 +: 64]),
      .arlen(arlen[bank*8 +: 8]),
      .arvalid(arvalid[bank]),
      .arready(arready[bank]),
      .rdata(rdata[bank*128 +: 128]),
      .rresp(rresp[bank*2 +: 2]),
      .rlast(rlast[bank]),
      .rvalid(rvalid[bank]),
      .rready(rready[bank])
    );
  end
`else
  logic [3:0] arready=0, rvalid=0, rlast=0, active=0;
  logic [511:0] rdata=0;
  wire [7:0] rresp=0;
`endif
  wire [31:0] payload_errors, response_errors;
  wire [63:0] spec_bytes, demand_bytes, useful_bytes, wasted_bytes, cancelled_bytes;
  integer beat_left[0:3];
  longint unsigned virtual_cursor[0:3];
  integer cycle=0, first_cycle=-1, last_cycle=-1;
  integer prefetch=0, pattern=0, beat_gap=0, startup=0;
  integer emitted_candidates;
  integer due_cycle[0:3];
  integer operation_count=0, router_count=0, truth_count=0;
  integer gu_count=0, down_count=0, waiting_gu_cycles=0, waiting_down_cycles=0;
  integer peak_slots=0, first_gu[0:LAYERS-1], layer_begin[0:LAYERS-1];
  longint unsigned transferred[0:3];
  integer bursts[0:3];
  integer timeline;
  string timeline_path;
  integer alignment_debug=0;
  logic router_running=0;
  integer router_cycles=0, expert_cycles=0;
  integer expected_x, expected_y;

  function automatic integer truth(input integer l);
    return l%EXPERTS;
  endfunction
  function automatic integer predicted(input integer target, input integer rank);
    // pattern=0: all top-1 hits. pattern=1: deterministic top-2 miss on alternating targets.
    if(pattern==1 && target%2==1) return (truth(target)+rank+2)%EXPERTS;
    return (truth(target)+rank)%EXPERTS;
  endfunction
  function automatic [127:0] fixture_word(input longint unsigned address_bytes);
    longint unsigned local_address, relative, offset;
    integer expert, tile, within_tile, row, column;
    bit down;
    logic [127:0] word_value;
    local_address=address_bytes%LAYER_STRIDE;
    down=local_address>=DOWN_BASE;
    relative=local_address-(down ? DOWN_BASE : UP_BASE);
    expert=int'(relative/EXPERT_STRIDE);
    offset=relative%EXPERT_STRIDE;
    if(expert>=EXPERTS || offset>=H*F) $fatal(1,"invalid weight address %h",address_bytes);
    word_value=0;
    for(integer lane=0;lane<16;lane=lane+1) begin
      tile=int'((offset+lane)/FRAGMENT);
      within_tile=int'((offset+lane)%FRAGMENT);
      row=within_tile/(down ? FT : H);
      column=within_tile%(down ? FT : H);
      if(down) begin
        if(column==row%FT) word_value[lane*8 +: 8]=8'(1+expert%2);
      end else if(column==(tile*FT+row)%H)
        word_value[lane*8 +: 8]=8'd1;
    end
    return word_value;
  endfunction

  u200_moe_top #(
    .MAX_CANDIDATES(EXPERTS),
    .NUM_EXPERTS(EXPERTS),
    .TILES_PER_EXPERT(TILES),
    .DESCRIPTOR_TABLE_ENTRIES(128),
    .NUM_OCM_BANKS(4),
    .SLOTS_PER_BANK(SLOTS_PER_BANK),
    .OCM_SLOT_BYTES(SLOT_BYTES),
    .DMA_FIFO_DEPTH(64),
    .MAX_BURST_BYTES(256),
    .H(H),
    .F_TILE(FT),
    .SWITCH_RELU(1'b1),
    .LAYER_COUNT(LAYERS),
    .WAIT_FULL_EXPERT(WHOLE)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd0),
    .sequence_active(1'b1),
    .sequence_start(sequence_start),
    .sequence_start_ready(start_ready),
    .sequence_busy(sequence_busy),
    .sequence_done(sequence_done),
    .current_layer(current_layer),
    .layer_output_requant_shift(5'(REQUANT_SHIFT)),
    .future_begin_valid(future_begin),
    .future_begin_ready(future_begin_ready),
    .future_target_layer(future_layer),
    .future_epoch(16'(future_layer+1)),
    .future_sequence(16'd1),
    .future_candidate_count(5'(emitted_candidates)),
    .future_budget_bytes(32'(2*EXPERT_BYTES)),
    .future_channel_credit_bytes(32'(2*EXPERT_BYTES)),
    .future_gu_base_addr(UP_BASE+64'(future_layer)*LAYER_STRIDE),
    .future_down_base_addr(DOWN_BASE+64'(future_layer)*LAYER_STRIDE),
    .future_expert_stride_bytes(32'(EXPERT_STRIDE)),
    .future_candidate_valid(future_valid),
    .future_candidate_ready(future_ready),
    .future_candidate_expert(future_expert),
    .future_candidate_rank(future_rank),
    .future_candidate_confidence(future_confidence),
    .future_candidate_last(future_rank==emitted_candidates-1),
    .layer_weights_done(weights_done),
    .sequence_result_y(sequence_y),
    .layer_done(1'b0),
    .prediction_begin_valid(begin_valid),
    .prediction_begin_ready(begin_ready),
    .prediction_epoch(16'(layer+1)),
    .prediction_sequence(16'd1),
    .prediction_layer(layer),
    .prediction_count(5'd1),
    .speculative_budget_bytes(32'd0),
    .channel_credit_bytes(32'd0),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(10'd0),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd3000),
    .candidate_last(1'b1),
    .gu_base_addr(UP_BASE+64'(layer)*LAYER_STRIDE),
    .down_base_addr(DOWN_BASE+64'(layer)*LAYER_STRIDE),
    .expert_stride_bytes(32'(EXPERT_STRIDE)),
    .physical_tile_bytes(32'(FRAGMENT)),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch(16'd0),
    .compute_activation({H{8'd16}}),
    .gu_requant_shift(5'd0),
    .router_top_k_config(5'd1),
    .router_logit_shift(5'd0),
    .router_weight_reload_valid(reload),
    .router_weight_layer(layer),
    .router_weight_valid(weight_valid),
    .router_weight_ready(weight_ready),
    .router_weight_word_index(weight_index),
    .router_weight_data(weight_data),
    .router_weight_last(weight_last),
    .router_weights_loaded(weights_loaded),
    .router_weight_error(weight_error),
    .router_gemm_done(router_done),
    .result_valid(result_valid),
    .result_y(result_y),
    .compute_done(compute_done),
    .parser_error(parser_error),
    .config_error(config_error),
    .transaction_error(transaction_error),
    .payload_discarded_write_count(payload_errors),
    .dma_response_error_count(response_errors),
    .telemetry_speculative_issued_bytes(spec_bytes),
    .telemetry_demand_issued_bytes(demand_bytes),
    .telemetry_useful_speculative_bytes(useful_bytes),
    .telemetry_wasted_speculative_bytes(wasted_bytes),
    .telemetry_cancelled_speculative_bytes(cancelled_bytes),
    .m_ddr_araddr(araddr),
    .m_ddr_arlen(arlen),
    .m_ddr_arvalid(arvalid),
    .m_ddr_arready(arready),
    .m_ddr_rdata(rdata),
    .m_ddr_rresp(rresp),
    .m_ddr_rlast(rlast),
    .m_ddr_rvalid(rvalid),
    .m_ddr_rready(rready)
  );

`ifndef M10_SWITCH_DDR4
  always_comb arready=~active;
`endif
  always @(posedge clk) begin
    cycle<=cycle+1;
    for(integer bank=0;bank<4;bank=bank+1) begin
      if(arvalid[bank] && arready[bank]) begin
`ifndef M10_SWITCH_DDR4
        active[bank]<=1;
        beat_left[bank]<=int'(arlen[bank*8 +: 8])+1;
        virtual_cursor[bank]<=araddr[bank*64 +: 64]*4+bank*16;
        due_cycle[bank]<=cycle+startup;
`endif
        bursts[bank]<=bursts[bank]+1;
      end
      if(rvalid[bank] && rready[bank]) begin
`ifndef M10_SWITCH_DDR4
        rvalid[bank]<=0;
        virtual_cursor[bank]<=virtual_cursor[bank]+64;
        beat_left[bank]<=beat_left[bank]-1;
`endif
        transferred[bank]<=transferred[bank]+16;
`ifndef M10_SWITCH_DDR4
        due_cycle[bank]<=cycle+beat_gap;
        if(beat_left[bank]==1) active[bank]<=0;
      end else if(active[bank] && !rvalid[bank] && cycle>=due_cycle[bank]) begin
        rvalid[bank]<=1;
        rlast[bank]<=beat_left[bank]==1;
        rdata[bank*128 +: 128]<=fixture_word(virtual_cursor[bank]);
`endif
      end
    end
    if(begin_valid && begin_ready) begin
      if(first_cycle<0) first_cycle=cycle;
      layer_begin[layer]=cycle;
      $display("FULL_LAYER_START layer=%0d cycle=%0d",layer,cycle);
      $fdisplay(timeline,"%0d,LAYER_START,%0d,-1,-1",cycle,layer);
    end
    if(dut.core.router_start_pulse) router_running<=1;
    if(router_done) begin router_running<=0; router_count=router_count+1; end
    if(router_running) router_cycles=router_cycles+1;
    if(dut.compute_busy && !router_running) expert_cycles=expert_cycles+1;
    if(dut.compute_waiting_gu) waiting_gu_cycles=waiting_gu_cycles+1;
    if(dut.compute_waiting_down) waiting_down_cycles=waiting_down_cycles+1;
    if(dut.core.memory_dataflow.core.ocm_occupancy>peak_slots)
      peak_slots=int'(dut.core.memory_dataflow.core.ocm_occupancy);
    if(dut.core.memory_dataflow.core.ocm_occupancy>4*SLOTS_PER_BANK)
      $fatal(1,"OCM capacity exceeded");
    if(compute_done) operation_count=operation_count+1;
    if(dut.core.effective_route_expert_valid && dut.core.effective_route_expert_ready) begin
      if(dut.core.effective_route_expert!=truth(int'(layer)) || dut.core.effective_route_weight_q15!=16'd32768)
        $fatal(1,"actual Router truth/gate mismatch layer=%0d",layer);
      truth_count=truth_count+1;
    end
    if(dut.core.memory_dataflow.backend_gu_start) begin
      gu_count=gu_count+1;
      if(first_gu[layer]<0) first_gu[layer]=cycle;
      if(WHOLE && !weights_done[layer]) $fatal(1,"whole Expert compute before all weights ready");
      $fdisplay(timeline,"%0d,GU_START,%0d,%0d,%0d",cycle,layer,
        dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
    end
    if(dut.core.memory_dataflow.backend_down_start) begin
      down_count=down_count+1;
      $fdisplay(timeline,"%0d,DOWN_START,%0d,%0d,%0d",cycle,layer,
        dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
    end
    if(rst_n && (transaction_error || parser_error || weight_error || response_errors || payload_errors))
      $fatal(1,"full-size protocol/data path error cycle=%0d",cycle);
  end

  task automatic submit_future(input integer target);
    integer handshake_wait;
    @(negedge clk);
    future_layer=3'(target);
    future_begin=1;
    handshake_wait=0;
    do begin
      @(posedge clk);
      handshake_wait=handshake_wait+1;
      if(handshake_wait>1024) $fatal(1,"future begin rejected target=%0d stride=%0d",target,EXPERT_STRIDE);
    end while(!future_begin_ready);
    @(negedge clk); future_begin=0;
    for(integer rank=0;rank<emitted_candidates;rank=rank+1) begin
      future_rank=5'(rank);
      future_expert=10'(predicted(target,rank));
      future_confidence=rank==0 ? 12'd3104 : rank==1 ? 12'd352 : 12'(639/(EXPERTS-2));
      future_valid=1;
      handshake_wait=0;
      do begin
        @(posedge clk);
        handshake_wait=handshake_wait+1;
        if(handshake_wait>1024) $fatal(1,"future candidate rejected target=%0d rank=%0d",target,rank);
      end while(!future_ready);
      @(negedge clk); future_valid=0;
    end
    $display("FULL_PREDICTION target=%0d cycle=%0d candidates=%0d",target,cycle,emitted_candidates);
  endtask

  initial begin
    if(H%16 || FT%16 || F%FT || LAYERS<2 || LAYERS>8 || EXPERTS<3 || EXPERTS>16)
      $fatal(1,"invalid full-size geometry");
    if(FRAGMENT>SLOT_BYTES || FRAGMENT%64 || 2*H*F>EXPERT_STRIDE ||
       UP_BASE+64'(EXPERTS)*EXPERT_STRIDE>DOWN_BASE ||
       DOWN_BASE+64'(EXPERTS)*EXPERT_STRIDE>LAYER_STRIDE)
      $fatal(1,"weight payload does not fit slot/stride");
    if(!$value$plusargs("PREFETCH=%d",prefetch)) prefetch=0;
    if(!$value$plusargs("PATTERN=%d",pattern)) pattern=0;
    if(!$value$plusargs("BEAT_GAP=%d",beat_gap)) beat_gap=0;
    if(!$value$plusargs("STARTUP=%d",startup)) startup=0;
    if(!$value$plusargs("ALIGNMENT_DEBUG=%d",alignment_debug)) alignment_debug=0;
    if(!$value$plusargs("TIMELINE=%s",timeline_path)) timeline_path="full_timeline.csv";
    timeline=$fopen(timeline_path,"w");
    if(!timeline) $fatal(1,"cannot create timeline");
    $fdisplay(timeline,"cycle,event,layer_id,expert_id,tile_id");
    emitted_candidates=WHOLE ? 2 : EXPERTS;
    for(integer bank=0;bank<4;bank=bank+1) begin
      beat_left[bank]=0; virtual_cursor[bank]=0; transferred[bank]=0; bursts[bank]=0; due_cycle[bank]=0;
    end
    for(integer l=0;l<LAYERS;l=l+1) begin first_gu[l]=-1; layer_begin[l]=-1; end
`ifdef M10_SWITCH_DDR4
    if(beat_gap!=0 || startup!=0) $fatal(1,"behavioral delays cannot be applied to MIG backend");
    wait(&ddr_initialized);
    $display("FULL_BACKEND MIG_RDIMM initialized_cycle=%0d",cycle);
`else
    $display("FULL_BACKEND BEHAVIORAL_AXI");
`endif
    repeat(4) @(negedge clk);
    rst_n=1;
    wait(start_ready);
    @(negedge clk); sequence_start=1;
    @(negedge clk); sequence_start=0;
    expected_x=16;
    for(integer index=0;index<LAYERS;index=index+1) begin
      wait(current_layer==index && dut.chain_layer_ready);
      @(negedge clk); layer=3'(index); reload=1;
      @(negedge clk); reload=0;
      for(integer word_index=0;word_index<H*EXPERTS/16;word_index=word_index+1) begin
        weight_index=32'(word_index); weight_data=0;
        if(word_index%(H/16)==0)
          weight_data[7:0]=word_index/(H/16)==truth(index) ? 8'd1 : 8'hff;
        weight_last=word_index==H*EXPERTS/16-1; weight_valid=1;
        do @(posedge clk); while(!weight_ready);
        @(negedge clk); weight_valid=0;
      end
      weight_last=0;
      if(!weights_loaded || weight_error) $fatal(1,"Router weight load failed");
      begin_valid=1;
      do @(posedge clk); while(!begin_ready);
      @(negedge clk); begin_valid=0; candidate_valid=1;
      do @(posedge clk); while(!candidate_ready);
      @(negedge clk); candidate_valid=0;
      if(prefetch && index+1<LAYERS) submit_future(index+1);
      expected_y=TILES*expected_x*(1+truth(index)%2);
      wait(result_valid);
      last_cycle=cycle;
      for(integer row=0;row<H;row=row+1)
        if($signed(result_y[row*32 +: 32])!==expected_y)
          $fatal(1,"full-size golden mismatch layer=%0d row=%0d expected=%0d actual=%0d",index,row,expected_y,$signed(result_y[row*32 +: 32]));
      $display("FULL_LAYER_RESULT layer=%0d cycle=%0d y=%0d first_gu=%0d",index,cycle,expected_y,first_gu[index]);
      $fdisplay(timeline,"%0d,LAYER_DONE,%0d,-1,-1",cycle,index);
      @(negedge clk);
      expected_x=expected_y>>>REQUANT_SHIFT;
      if(expected_x>127) expected_x=127;
    end
    wait(sequence_done);
    wait(dut.core_drained);
    repeat(4) @(negedge clk);
    // compute_done counts completed logical tiles, not individual GEMMs.
    // GU and DOWN operation counters independently verify both matrix stages.
    if(operation_count!=TILES*LAYERS || gu_count!=TILES*LAYERS || down_count!=TILES*LAYERS || router_count!=LAYERS || truth_count!=LAYERS)
      $fatal(1,"operations missing gu=%0d down=%0d count=%0d router=%0d truth=%0d",gu_count,down_count,operation_count,router_count,truth_count);
    for(integer row=0;row<H;row=row+1)
      if($signed(sequence_y[row*32 +: 32])!==expected_y) $fatal(1,"final result lost");
    if(spec_bytes!=useful_bytes+wasted_bytes || spec_bytes+demand_bytes!=transferred[0]+transferred[1]+transferred[2]+transferred[3])
      $fatal(1,"physical/telemetry byte conservation");
    if(!prefetch && (spec_bytes!=0 || demand_bytes!=64'(LAYERS)*EXPERT_BYTES)) $fatal(1,"baseline transfer mismatch");
    $display("FULL_METRICS cycles=%0d wait_gu=%0d wait_down=%0d router=%0d expert=%0d peak_slots=%0d spec=%0d demand=%0d useful=%0d wasted=%0d cancelled=%0d",last_cycle-first_cycle,waiting_gu_cycles,waiting_down_cycles,router_cycles,expert_cycles,peak_slots,spec_bytes,demand_bytes,useful_bytes,wasted_bytes,cancelled_bytes);
    for(integer bank=0;bank<4;bank=bank+1)
      $display("FULL_BANK bank=%0d bytes=%0d bursts=%0d",bank,transferred[bank],bursts[bank]);
    $fclose(timeline);
    $display("TEST_PASS full Switch MoE H=%0d F=%0d E=%0d layers=%0d whole=%0d prefetch=%0d pattern=%0d",H,F,EXPERTS,LAYERS,WHOLE,prefetch,pattern);
    $finish;
  end

  // 仅观测 handshake，不驱动 DUT；默认关闭，避免对普通回归增加大量日志。
  always @(posedge clk) begin
    if(rst_n && alignment_debug) begin
      if(dut.core.memory_logical_valid && dut.core.logical_ready)
        $display("ALIGN_PAIR cycle=%0d layer=%0d expert=%0d tile=%0d class=%0d slots=%0d",cycle,
          dut.core.memory_gu_descriptor.layer_id,dut.core.memory_gu_descriptor.expert_id,
          dut.core.memory_gu_descriptor.tile_id,dut.core.memory_gu_descriptor.request_class,
          dut.core.memory_dataflow.core.ocm_occupancy);
      if(dut.core.memory_dataflow.core.claim_valid)
        $display("ALIGN_CLAIM cycle=%0d layer=%0d expert=%0d tile=%0d matrix=%0d class=%0d",cycle,
          dut.core.memory_dataflow.core.active_descriptor.layer_id,
          dut.core.memory_dataflow.core.active_descriptor.expert_id,
          dut.core.memory_dataflow.core.active_descriptor.tile_id,
          dut.core.memory_dataflow.core.active_descriptor.matrix_kind,
          dut.core.memory_dataflow.core.active_descriptor.request_class);
      if(dut.core.memory_dataflow.core.dma_done)
        $display("ALIGN_DONE cycle=%0d layer=%0d expert=%0d tile=%0d matrix=%0d",cycle,
          dut.core.memory_dataflow.core.active_descriptor.layer_id,
          dut.core.memory_dataflow.core.active_descriptor.expert_id,
          dut.core.memory_dataflow.core.active_descriptor.tile_id,
          dut.core.memory_dataflow.core.active_descriptor.matrix_kind);
      if(dut.core.memory_dataflow.core.table_mark_promoted || dut.core.memory_dataflow.core.table_mark_cancelled)
        $display("ALIGN_ROUTE cycle=%0d layer=%0d expert=%0d tile=%0d matrix=%0d truth=%0d",cycle,
          dut.core.memory_dataflow.core.route_query_descriptor.layer_id,
          dut.core.memory_dataflow.core.route_query_descriptor.expert_id,
          dut.core.memory_dataflow.core.route_query_descriptor.tile_id,
          dut.core.memory_dataflow.core.route_query_descriptor.matrix_kind,
          dut.core.memory_dataflow.core.table_mark_promoted);
    end
  end
  initial begin
`ifdef M10_SWITCH_DDR4
    wait(&ddr_initialized);
`endif
    #100000000;
    $fatal(1,"full-size timeout cycle=%0d layer=%0d running=%b occupancy=%0d",cycle,layer,sequence_busy,dut.core.memory_dataflow.core.ocm_occupancy);
  end
endmodule
