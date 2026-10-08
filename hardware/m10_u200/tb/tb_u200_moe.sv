`timescale 1ns/1ps
/** 真实 Router truth 3/4，外部 predictor 可命中/错误/无预算。
 * 4-bank 权重数据、有限 OCM、两 F tiles、Q15 combine，soft-flush 后重用。
 */
module tb_u200_moe #(
  parameter integer PREDICTED_EXPERT=3,
  parameter bit SWITCH_RELU=0,
  parameter integer BUDGET=SWITCH_RELU ? 512 : 768,
  parameter integer FLUSH_AT=0,
  parameter integer ERROR_BANK=-1,
  parameter integer DDR_GAPS=1,
  parameter integer LAYERS=1,
  parameter integer QUANT_SHIFT=0,
  parameter integer REPEATS=1,
  parameter integer FUTURE_MODE=0,
  parameter integer ABORT_FUTURE=0,
  parameter integer OCM_SLOTS_PER_BANK=2,
  parameter integer DESCRIPTOR_ENTRIES=8,
  parameter integer FIFO_DEPTH=4,
  parameter integer PREDICTOR_DELAY_CYCLES=0,
  parameter integer EXPECT_EARLY_OVERLAP=1,
  parameter bit WAIT_FULL_EXPERT=0,
  parameter integer ROUTE_K=2,
  parameter integer FUTURE_BUDGET_BYTES=SWITCH_RELU ? 1024 : 1536,
  parameter integer DDR_BEAT_GAP=0
);
  logic clk=0;
  always #2.5 clk=~clk;
  logic rst_n=0, pred_begin=0, pred_ready, candidate_valid=0, candidate_ready;
  logic weight_valid=0, weight_ready, weight_last=0, loaded, load_error, gemv_done;
  logic [31:0] weight_index=0;
  logic [127:0] weight_data=0;
  logic result_valid, parser_error, config_error, transaction_error, compute_done;
  logic [511:0] result_y;
  logic [255:0] araddr;
  logic [31:0] arlen, response_errors, payload_errors;
  logic [3:0] arvalid, arready, rready, active=0;
  wire [2:0] observed_chain_state;
  if(LAYERS>1) begin : observe_multilayer_state
    assign observed_chain_state=3'(dut.multilayer.controller.state);
  end else begin : observe_singlelayer_state
    assign observed_chain_state=0;
  end
`ifdef M10_DDR4_MODEL
  wire [3:0] rvalid, rlast;
  wire [511:0] rdata;
  wire [7:0] rresp;
  wire [3:0] ddr_initialized;
  for(genvar bank=0;bank<4;bank=bank+1) begin: ddr_backend
    ddr4_bank_backend #(
      .BANK_ID(bank),
      .LAYERS(LAYERS)
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
  logic [3:0] rvalid=0, rlast=0;
  logic [511:0] rdata=0;
  logic [7:0] rresp=0;
`endif
  logic flush=0;
  logic [15:0] epoch=1;
  logic injected_error=0;
  integer left[0:3], cursor[0:3];
  integer cycle=0, b, lane, count=0, router_count=0;
  integer layer_start_cycle=-1, layer_finish_cycle=-1, trace_file=0, alignment_weight_wait=0;
  integer alignment_descriptor_peak=0;
  logic nl_done_previous=0;
  logic chain_start=0, chain_start_ready, chain_done, chain_busy;
  logic weight_reload=0;
  logic [2:0] layer=0, current_layer;
  logic [15:0] prediction_sequence_id=1;
  logic [511:0] sequence_result;
  logic future_begin=0, future_begin_ready, future_valid=0, future_ready;
  logic [2:0] future_layer=1;
  logic [15:0] future_epoch_id=2;
  logic [9:0] future_expert=2;
  wire [LAYERS-1:0] weights_done;
  integer future_admissions=0, future_early_admissions=0, future_completions=0;
  integer future_dma_starts=0;
  logic [LAYERS-1:0] barriers_seen=0;
  integer admitted_count [0:LAYERS-1][0:5][0:1][0:1];
  string trace_path;
  logic [63:0] spec, demand, useful, wasted, cancelled;
  localparam integer PAIR_BYTES=SWITCH_RELU ? 512 : 768;
  u200_moe_top #(
    .MAX_CANDIDATES(2),
    .NUM_EXPERTS(6),
    .TILES_PER_EXPERT(2),
    .DESCRIPTOR_TABLE_ENTRIES(DESCRIPTOR_ENTRIES),
    .NUM_OCM_BANKS(2),
    .SLOTS_PER_BANK(OCM_SLOTS_PER_BANK),
    .OCM_SLOT_BYTES(512),
    .DMA_FIFO_DEPTH(FIFO_DEPTH),
    .LAYER_COUNT(LAYERS),
    .WAIT_FULL_EXPERT(WAIT_FULL_EXPERT),
    .SWITCH_RELU(SWITCH_RELU),
    .ENABLE_LAYER_CHAIN(LAYERS>1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd0),
    .sequence_active(1'b1),
    .sequence_start(chain_start),
    .sequence_start_ready(chain_start_ready),
    .sequence_busy(chain_busy),
    .sequence_done(chain_done),
    .current_layer(current_layer),
    .layer_output_requant_shift(5'(QUANT_SHIFT)),
    .future_begin_valid(future_begin),
    .future_begin_ready(future_begin_ready),
    .future_target_layer(future_layer),
    .future_epoch(future_epoch_id),
    .future_sequence(16'd1),
    .future_candidate_count(5'd1),
    .future_budget_bytes(32'(FUTURE_BUDGET_BYTES)),
    .future_channel_credit_bytes(32'(FUTURE_BUDGET_BYTES)),
    .future_gu_base_addr(64'h10000+64'(future_layer)*64'h100000),
    .future_down_base_addr(64'h20000+64'(future_layer)*64'h100000),
    .future_expert_stride_bytes(32'd4096),
    .future_candidate_valid(future_valid),
    .future_candidate_ready(future_ready),
    .future_candidate_expert(future_expert),
    .future_candidate_rank(5'd0),
    .future_candidate_confidence(12'd3000),
    .future_candidate_last(1'b1),
    .layer_weights_done(weights_done),
    .sequence_result_y(sequence_result),
    .layer_done(1'b0),
    .prediction_begin_valid(pred_begin),
    .prediction_begin_ready(pred_ready),
    .prediction_epoch(epoch),
    .prediction_sequence(prediction_sequence_id),
    .prediction_layer(layer),
    .prediction_count(5'd1),
    .speculative_budget_bytes(BUDGET),
    .channel_credit_bytes(32'(PAIR_BYTES)),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(10'(PREDICTED_EXPERT)),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd3000),
    .candidate_last(1'b1),
    .gu_base_addr(64'h10000+64'(layer)*64'h100000),
    .down_base_addr(64'h20000+64'(layer)*64'h100000),
    .expert_stride_bytes(32'd4096),
    .physical_tile_bytes(32'd256),
    .soft_flush_valid(flush),
    .soft_flush_epoch(epoch),
    .compute_activation({16{8'd16}}),
    .gu_requant_shift(5'd0),
    .router_top_k_config(5'(ROUTE_K)),
    .router_logit_shift(5'd0),
    .router_weight_reload_valid(weight_reload),
    .router_weight_layer(layer),
    .router_weight_valid(weight_valid),
    .router_weight_ready(weight_ready),
    .router_weight_word_index(weight_index),
    .router_weight_data(weight_data),
    .router_weight_last(weight_last),
    .router_weights_loaded(loaded),
    .router_weight_error(load_error),
    .router_gemm_done(gemv_done),
    .result_valid(result_valid),
    .result_y(result_y),
    .compute_done(compute_done),
    .parser_error(parser_error),
    .config_error(config_error),
    .transaction_error(transaction_error),
    .payload_discarded_write_count(payload_errors),
    .dma_response_error_count(response_errors),
    .telemetry_speculative_issued_bytes(spec),
    .telemetry_demand_issued_bytes(demand),
    .telemetry_useful_speculative_bytes(useful),
    .telemetry_wasted_speculative_bytes(wasted),
    .telemetry_cancelled_speculative_bytes(cancelled),
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
`ifndef M10_DDR4_MODEL
  always_comb begin
    for(integer bank=0;bank<4;bank=bank+1)
      arready[bank]=!active[bank] && (DDR_GAPS==0 || cycle%(bank+2)!=0);
  end
`endif
  always @(posedge clk) begin
    if(rst_n && dut.core.memory_logical_valid && dut.core.logical_ready &&
        (dut.core.memory_gu_descriptor.byte_count!=(SWITCH_RELU ? 256 : 512) ||
         dut.core.memory_down_descriptor.byte_count!=256))
      $fatal(1,"Expert mode/descriptor byte mismatch, including future prediction");
    if(chain_start && chain_start_ready) barriers_seen<=0;
    else barriers_seen<=weights_done;
    for(integer l=0;l<LAYERS;l=l+1)
      if(LAYERS>1 && weights_done[l] && !barriers_seen[l])
        $display("WEIGHTS_READY layer=%0d cycle=%0d",l,cycle);
  end
  always @(posedge clk) begin
    nl_done_previous <= dut.core.memory_dataflow.backend_nonlinear_done;
    cycle<=cycle+1;
    if(compute_done) count<=count+1;
    if(gemv_done) router_count<=router_count+1;
    if(dut.core.memory_dataflow.core.ocm_occupancy>2*OCM_SLOTS_PER_BANK) $fatal(1,"OCM capacity exceeded");
    if(LAYERS>1 && dut.chain_retiring && dut.future_logical_valid && dut.future_logical_ready)
      $fatal(1,"future admission raced context cleanup");
    if(FUTURE_MODE>0 && dut.core.physical_accept_valid) begin
      admitted_count[dut.core.physical_accept_descriptor.layer_id]
        [dut.core.physical_accept_descriptor.expert_id][dut.core.physical_accept_descriptor.tile_id]
        [dut.core.physical_accept_descriptor.matrix_kind]=
        admitted_count[dut.core.physical_accept_descriptor.layer_id]
          [dut.core.physical_accept_descriptor.expert_id][dut.core.physical_accept_descriptor.tile_id]
          [dut.core.physical_accept_descriptor.matrix_kind]+1;
      if(admitted_count[dut.core.physical_accept_descriptor.layer_id]
          [dut.core.physical_accept_descriptor.expert_id][dut.core.physical_accept_descriptor.tile_id]
          [dut.core.physical_accept_descriptor.matrix_kind]>1) $fatal(1,"duplicate future/demand physical key");
      if(dut.core.physical_accept_descriptor.layer_id>16'(current_layer)) begin
        future_admissions=future_admissions+1;
        if(chain_busy && dut.core.workload_busy) future_early_admissions=future_early_admissions+1;
        $display("FUTURE_ADMIT layer=%0d current=%0d cycle=%0d",dut.core.physical_accept_descriptor.layer_id,current_layer,cycle);
      end
    end
    if(FUTURE_MODE>0 && dut.core.dma_completion_valid && dut.core.dma_completion_descriptor.layer_id>16'(current_layer))
      future_completions=future_completions+1;
    if(FUTURE_MODE>0 && dut.core.memory_dataflow.dma_burst_start &&
        dut.core.memory_dataflow.dma_active_descriptor.layer_id>16'(current_layer)) begin
      future_dma_starts=future_dma_starts+1;
      $display("FUTURE_DMA layer=%0d current=%0d cycle=%0d",dut.core.memory_dataflow.dma_active_descriptor.layer_id,current_layer,cycle);
    end
    if(dut.core.workload_busy && weight_ready) $fatal(1,"Router weights writable during transaction");
    if(dut.core.effective_route_expert_valid && dut.core.effective_route_expert_ready &&
       ((dut.core.effective_route_expert!=(LAYERS>1 ? layer%4+1 : 3) &&
         dut.core.effective_route_expert!=(LAYERS>1 ? layer%4+2 : 4)) ||
         dut.core.effective_route_weight_q15!=(ROUTE_K==1 ? 32768 : 16384)))
      $fatal(1,"predictor altered real Router truth/gates");
    // 任意 Expert 数值运算必须在 Router MAC 完成后；两者共享同一物理 backend。
    if(dut.core.memory_dataflow.backend_gu_start && router_count==0)
      $fatal(1,"expert before Router");
    if(WAIT_FULL_EXPERT && ROUTE_K==1 && dut.core.memory_dataflow.backend_gu_start &&
        dut.core.auto_gu_descriptor.tile_id==0 && !weights_done[int'(layer)])
      $fatal(1,"whole Expert mode began before all GU/DOWN weights completed");
`ifndef M10_DDR4_MODEL
    for(b=0;b<4;b=b+1) begin
      if(arvalid[b] && arready[b]) begin
        active[b]<=1;
        left[b]<=int'(arlen[b*8 +: 8])+1;
        cursor[b]<=int'(araddr[b*64 +: 64])*4+b*16;
      end
      if(rvalid[b] && rready[b]) begin
        rvalid[b]<=0;
        cursor[b]<=cursor[b]+64;
        left[b]<=left[b]-1;
        if(left[b]==1) active[b]<=0;
      end else if(active[b] && !rvalid[b] && (DDR_GAPS==0 || cycle%(b+3)!=0) &&
          (DDR_BEAT_GAP==0 || cycle%DDR_BEAT_GAP==0)) begin
        rvalid[b]<=1;
        rlast[b]<=left[b]==1;
        rdata[b*128 +: 128]<=128'd1 << (((cursor[b]/16)%16)*8);
        rresp[b*2 +: 2]<=b==ERROR_BANK && !injected_error ? 2'b10 : 2'b00;
        if(b==ERROR_BANK && !injected_error) injected_error<=1;
      end
    end
`endif
  end
  // 对齐采样统一为 posedge 前可见的 valid/ready；cycle 0 是 layer begin handshake。
  // 只观测，不回驱 DUT，不改变功能或周期。
  initial begin
    if($value$plusargs("TRACE=%s",trace_path)) begin
      trace_file=$fopen(trace_path,"w");
      if(trace_file==0) $fatal(1,"cannot open alignment trace");
      $fdisplay(trace_file,"cycle,event,expert_id,tile_id,matrix,bytes,request_type,value");
    end
  end
  always @(posedge clk) begin
    if(pred_begin && pred_ready) begin
      if(LAYERS>1) $display("LAYER_START layer=%0d cycle=%0d epoch=%0d",layer,cycle,epoch);
      layer_start_cycle=cycle;
      if(trace_file!=0) $fdisplay(trace_file,"0,LAYER_START,-1,-1,none,0,none,0");
    end
    if(trace_file!=0 && layer_start_cycle>=0) begin
      if(dut.core.automatic_execution.execution.truth_ready &&
         !dut.core.automatic_execution.execution.running &&
         !dut.core.automatic_execution.execution.issue_pending &&
         !dut.core.automatic_execution.execution.combine_valid1 &&
         !dut.core.automatic_execution.execution.combine_valid2 &&
         !dut.core.automatic_execution.execution.gu_found &&
         dut.core.automatic_execution.execution.completed_experts <
           dut.core.automatic_execution.execution.truth_count)
        alignment_weight_wait=alignment_weight_wait+1;
      if(candidate_valid && candidate_ready)
        $fdisplay(trace_file,"%0d,PREDICTOR_READY,%0d,0,none,0,none,3000",cycle-layer_start_cycle,PREDICTED_EXPERT);
      if(dut.core.router_start_pulse)
        $fdisplay(trace_file,"%0d,ROUTER_START,-1,-1,router,0,none,0",cycle-layer_start_cycle);
      if(gemv_done)
        $fdisplay(trace_file,"%0d,ROUTER_DONE,-1,-1,router,0,none,0",cycle-layer_start_cycle);
      if(dut.core.effective_route_expert_valid && dut.core.effective_route_expert_ready)
        $fdisplay(trace_file,"%0d,TRUTH,%0d,-1,none,0,none,%0d",cycle-layer_start_cycle,
          dut.core.effective_route_expert,dut.core.effective_route_weight_q15);
      if(dut.core.physical_accept_valid)
        $fdisplay(trace_file,"%0d,ADMIT,%0d,%0d,%0d,%0d,%0d,0",cycle-layer_start_cycle,
          dut.core.physical_accept_descriptor.expert_id,dut.core.physical_accept_descriptor.tile_id,
          dut.core.physical_accept_descriptor.matrix_kind,dut.core.physical_accept_descriptor.byte_count,
          dut.core.physical_accept_descriptor.request_class);
      if(dut.core.memory_dataflow.dma_burst_start)
        $fdisplay(trace_file,"%0d,DMA_START,%0d,%0d,%0d,%0d,%0d,0",cycle-layer_start_cycle,
          dut.core.memory_dataflow.dma_active_descriptor.expert_id,
          dut.core.memory_dataflow.dma_active_descriptor.tile_id,
          dut.core.memory_dataflow.dma_active_descriptor.matrix_kind,
          dut.core.memory_dataflow.dma_burst_bytes,
          dut.core.memory_dataflow.dma_active_descriptor.request_class);
      if(dut.core.memory_dataflow.adapter_burst_complete)
        $fdisplay(trace_file,"%0d,DMA_DONE,%0d,%0d,%0d,%0d,%0d,0",cycle-layer_start_cycle,
          dut.core.memory_dataflow.dma_active_descriptor.expert_id,
          dut.core.memory_dataflow.dma_active_descriptor.tile_id,
          dut.core.memory_dataflow.dma_active_descriptor.matrix_kind,
          dut.core.memory_dataflow.dma_burst_bytes,
          dut.core.memory_dataflow.dma_active_descriptor.request_class);
      if(dut.core.memory_dataflow.backend_gu_start)
        $fdisplay(trace_file,"%0d,GU_START,%0d,%0d,gate_up,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_gu_done)
        $fdisplay(trace_file,"%0d,GU_DONE,%0d,%0d,gate_up,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_nonlinear_done && !nl_done_previous)
        $fdisplay(trace_file,"%0d,NL_DONE,%0d,%0d,nonlinear,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_down_start)
        $fdisplay(trace_file,"%0d,DOWN_START,%0d,%0d,down,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_down_done)
        $fdisplay(trace_file,"%0d,DOWN_DONE,%0d,%0d,down,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
      if(result_valid) begin
        layer_finish_cycle=cycle-layer_start_cycle;
        $fdisplay(trace_file,"%0d,LAYER_DONE,-1,-1,none,0,none,0",layer_finish_cycle);
      end
    end
  end
  task automatic begin_transaction;
    begin
      pred_begin=1;
      wait(pred_ready);
      @(negedge clk); pred_begin=0;
      candidate_valid=1;
      wait(candidate_ready);
      @(negedge clk); candidate_valid=0;
    end
  endtask
  task automatic load_layer_router(input integer index);
    weight_reload=1;
    @(negedge clk); weight_reload=0;
    for(weight_index=0;weight_index<6;weight_index=weight_index+1) begin
      weight_data=0;
      weight_data[7:0]=(weight_index==index%4+1 || weight_index==index%4+2) ? 8'd1 : 8'hff;
      weight_last=weight_index==5; weight_valid=1;
      @(negedge clk);
    end
    weight_valid=0; weight_last=0;
    if(!loaded || load_error) $fatal(1,"layer Router load failed");
  endtask
  task automatic submit_future(input integer target, input integer tag);
    @(negedge clk);
    future_layer=3'(target); future_epoch_id=16'(tag);
    future_expert=FUTURE_MODE==2 ? 10'd5 : 10'(target%4+1);
    future_begin=1;
    wait(future_begin_ready);
    @(negedge clk); future_begin=0; future_valid=1;
    wait(future_ready);
    @(negedge clk); future_valid=0;
  endtask
  initial begin
    for(integer l=0;l<LAYERS;l=l+1)
      for(integer e=0;e<6;e=e+1)
        for(integer t=0;t<2;t=t+1)
          for(integer m=0;m<2;m=m+1) admitted_count[l][e][t][m]=0;
`ifdef M10_DDR4_MODEL
    if(FLUSH_AT!=0 || ERROR_BANK!=-1) $fatal(1,"DDR model backend requires normal traffic fixture");
    wait(&ddr_initialized);
    $display("DDR_MEASUREMENT_READY cycle=%0d",cycle);
`endif
    repeat(3) @(negedge clk); rst_n=1;
    if (LAYERS>1) begin : chain_stimulus
      integer expected_x, expected_y, gate;
      if(ABORT_FUTURE>0) begin
        wait(chain_start_ready);
        @(negedge clk); chain_start=1;
        @(negedge clk); chain_start=0;
        load_layer_router(0);
        begin_transaction();
        submit_future(1,2);
        wait(future_dma_starts>0);
        @(negedge clk); flush=1;
        @(negedge clk); flush=0;
        wait(dut.core_drained);
        repeat(LAYERS+5) @(negedge clk);
        if(chain_busy || chain_done || transaction_error || dut.core.memory_dataflow.core.ocm_occupancy!=0)
          $fatal(1,"cross-layer abort did not retire all contexts");
        count=0; router_count=0;
        $display("CROSS_ABORT_RECOVERED cycle=%0d",cycle);
      end
      for (integer repeat_id=0; repeat_id<REPEATS; repeat_id=repeat_id+1) begin
        for(integer l=0;l<LAYERS;l=l+1)
          for(integer e=0;e<6;e=e+1)
            for(integer t=0;t<2;t=t+1)
              for(integer m=0;m<2;m=m+1) admitted_count[l][e][t][m]=0;
        wait(chain_start_ready);
        @(negedge clk); chain_start=1;
        @(negedge clk); chain_start=0;
        if(FUTURE_MODE>0) fork
          begin
            wait(dut.sequence_context_live);
            repeat(PREDICTOR_DELAY_CYCLES) @(negedge clk);
            for(integer target=1;target<LAYERS;target=target+1) begin
              submit_future(target,repeat_id*8+target+1);
            end
          end
        join_none
        expected_x=16;
        for (integer index=0; index<LAYERS; index=index+1) begin
          wait(current_layer==index && dut.chain_layer_ready);
          @(negedge clk); layer=3'(index); epoch=16'(repeat_id*8+index+1);
          // 新层尚未reload时，不能使用上一层Router权重启动。
          if(index>0) begin
            pred_begin=1; #0.1;
            if(pred_ready || !config_error) $fatal(1,"wrong-layer Router weights accepted");
            pred_begin=0;
          end
          load_layer_router(index);
          // 已加载正确权重时仍必须拒绝错误层号/sequence；不能启动MAC。
          layer=3'((index+1)%LAYERS); pred_begin=1;
          repeat(2) @(negedge clk);
          if(pred_ready || dut.compute_busy) $fatal(1,"future layer transaction admitted");
          pred_begin=0; layer=3'(index);
          if(index>0) begin
            prediction_sequence_id=2; pred_begin=1;
            repeat(2) @(negedge clk);
            if(pred_ready || dut.compute_busy) $fatal(1,"wrong sequence transaction admitted");
            pred_begin=0; prediction_sequence_id=1;
          end
          begin_transaction();
          // 固定输入出现过的Q4 SiLU值；不是读取DUT LUT做参考。
          if(SWITCH_RELU) expected_y=expected_x>0 ? expected_x : 0;
          else begin
          case(expected_x)
            16: gate=12;
            24: gate=20;
            60: gate=59;
            127: gate=127;
            12: gate=8;
            6: gate=4;
            1: gate=1;
            0: gate=0;
            default: $fatal(1,"unsupported independent golden activation %0d",expected_x);
          endcase
          expected_y=gate*expected_x/16;
          if(expected_y>127) expected_y=127;
          end
          expected_y=2*expected_y;
          wait(result_valid);
          $display("LAYER_RESULT layer=%0d cycle=%0d y=%0d",index,cycle,$signed(result_y[31:0]));
          for(integer row=0;row<16;row=row+1)
            if($signed(result_y[row*32 +: 32])!=expected_y)
              $fatal(1,"layer=%0d activation=%0d expected=%0d got=%0d",index,expected_x,expected_y,$signed(result_y[row*32 +: 32]));
          @(negedge clk);
          expected_x=expected_y>>QUANT_SHIFT;
          if(expected_x>127) expected_x=127;
        end
        wait(chain_done);
        wait(dut.core_drained);
        repeat(4) @(negedge clk);
        if(chain_busy || current_layer!=LAYERS-1 || active || parser_error || transaction_error ||
            dut.core_drained!==1'b1) $fatal(1,"chain completion before retirement");
        for(integer row=0;row<16;row=row+1)
          if($signed(sequence_result[row*32 +: 32])!=expected_y) $fatal(1,"final result lost during cleanup");
        if(count!=2*ROUTE_K*LAYERS*(repeat_id+1) || router_count!=LAYERS*(repeat_id+1)) $fatal(1,"multilayer operations missing");
        @(negedge clk);
      end
      // overlap 是早预测合同，不是任意 late predictor 都必须有的功能。
      // 晚事务可以合法错过 target；数值、因果、drain 和 byte 守恒仍无条件检查。
      if(FUTURE_MODE>0 && EXPECT_EARLY_OVERLAP && PREDICTOR_DELAY_CYCLES==0 &&
          (future_admissions<2 || future_early_admissions<2 || future_dma_starts<1))
        $fatal(1,"no real cross-layer DMA overlap admissions=%0d early=%0d starts=%0d",future_admissions,future_early_admissions,future_dma_starts);
      if(FUTURE_MODE>0 && EXPECT_EARLY_OVERLAP && PREDICTOR_DELAY_CYCLES==0 && OCM_SLOTS_PER_BANK>2 && future_completions==0)
        $fatal(1,"larger OCM case did not complete prefetch before target layer");
      // issued 是实际完成传输的payload；未传输的cancelled只计入allocated守恒。
      if(FUTURE_MODE>0 && (spec!=useful+wasted ||
          dut.core.memory_dataflow.telemetry_speculative_allocated_bytes!=spec+cancelled ||
          parser_error || transaction_error))
        $fatal(1,"cross-layer byte accounting spec=%0d useful=%0d wasted=%0d cancelled=%0d parser=%0d transaction=%0d",spec,useful,wasted,cancelled,parser_error,transaction_error);
      if(FUTURE_MODE>0) $display("CROSS_BYTES allocated=%0d transferred=%0d useful=%0d wasted=%0d cancelled=%0d",dut.core.memory_dataflow.telemetry_speculative_allocated_bytes,spec,useful,wasted,cancelled);
      $display("PASS: real %0d-layer MoE chain quant=%0d repeats=%0d activation dependency, Router reload, context drain and final result",LAYERS,QUANT_SHIFT,REPEATS);
      $finish;
    end
    // 不存在 Router 权重时拒绝开始，不能偷偷使用外部 truth。
    pred_begin=1; #1;
    if(pred_ready || !config_error) $fatal(1,"unloaded Router accepted");
    pred_begin=0;
    for(weight_index=0;weight_index<6;weight_index=weight_index+1) begin
      weight_data=0;
      weight_data[7:0]=(weight_index==3 || weight_index==4) ? 8'd1 : 8'hff;
      weight_last=weight_index==5; weight_valid=1;
      @(negedge clk);
    end
    weight_valid=0; weight_last=0;
    if(!loaded || load_error) $fatal(1,"Router load failed loaded=%b error=%b next=%0d busy=%b", loaded, load_error, dut.core.real_router_control.weights.next_word, dut.core.workload_busy);
    begin_transaction();
    if(FLUSH_AT>0) begin
      repeat(FLUSH_AT) @(negedge clk);
      flush=1; @(negedge clk); flush=0;
      repeat(300) @(negedge clk);
      if(result_valid || active!=0) $fatal(1,"flush did not drain DDR");
      epoch=2; count=0; router_count=0;
      begin_transaction();
    end
    if(ERROR_BANK>=0) begin
      wait(transaction_error); repeat(300) @(negedge clk);
      if(result_valid || response_errors!=1) $fatal(1,"DDR error escaped");
      flush=1; @(negedge clk); flush=0;
      repeat(100) @(negedge clk);
      epoch=2; count=0; router_count=0;
      begin_transaction();
    end
    wait(result_valid); @(negedge clk);
    repeat(10) @(negedge clk);
    for(lane=0;lane<16;lane=lane+1)
      // SiLU_Q4(16)=12；12*16>>4=12；两 F tiles -> 24；等权 Top-2 保持 24。
      if($signed(result_y[lane*32 +: 32])!=(SWITCH_RELU ? 32 : 24)) $fatal(1,"numeric output lane=%0d value=%0d",lane,$signed(result_y[lane*32 +: 32]));
    if(parser_error || config_error || transaction_error || count!=4 || router_count!=1)
      $fatal(1,"end-to-end protocol/execution error");
    if(FLUSH_AT==0 && ERROR_BANK<0) begin
      if(spec+demand!=(PREDICTED_EXPERT==3 || PREDICTED_EXPERT==4 || BUDGET==0 ? 4*PAIR_BYTES : 5*PAIR_BYTES)) $fatal(1,"transfer bytes");
      if(BUDGET==0 && (spec!=0 || demand!=4*PAIR_BYTES)) $fatal(1,"no prefetch");
      if(BUDGET>0 && (PREDICTED_EXPERT==3 || PREDICTED_EXPERT==4) && useful!=PAIR_BYTES) $fatal(1,"useful bytes");
      if(BUDGET>0 && PREDICTED_EXPERT==5 && (demand!=4*PAIR_BYTES || wasted!=PAIR_BYTES)) $fatal(1,"wrong predictor changed Router");
    end
    $display("PASS: real Router + four DDR + finite OCM + Top-2 numerical MoE (pred=%0d budget=%0d flush=%0d error=%0d)",
      PREDICTED_EXPERT,BUDGET,FLUSH_AT,ERROR_BANK);
    if(trace_file!=0) begin
      $display("ALIGN_METRICS %0d %0d %0d %0d %0d %0d %0d %0d %0d %h",layer_finish_cycle,
        spec,demand,useful,wasted,cancelled,dut.core.demand_stall_cycles,alignment_weight_wait,
        dut.core.memory_dataflow.telemetry_ocm_peak_occupancy,result_y);
      $fclose(trace_file);
    end
    $finish;
  end
  initial begin
    #1000000;
    $fatal(1,"timeout layer=%0d logical=%b ready=%b free_descriptors=%0d occupancy=%0d demand_pair=%b spec_pair=%b",
      layer,dut.core.memory_dataflow.core.logical_valid,dut.core.memory_dataflow.core.logical_ready,
      dut.core.memory_dataflow.core.table_free_entries,dut.core.memory_dataflow.core.ocm_occupancy,
      dut.core.memory_dataflow.core.demand_pair_ready,dut.core.memory_dataflow.core.speculative_pair_ready);
  end
  // 单独的验证峰值寄存器，不参与 DUT 控制。
  always @(posedge clk) begin
    if(!rst_n) alignment_descriptor_peak <= 0;
    else if(DESCRIPTOR_ENTRIES-int'(dut.core.memory_dataflow.core.table_free_entries)>alignment_descriptor_peak)
      alignment_descriptor_peak <= DESCRIPTOR_ENTRIES-int'(dut.core.memory_dataflow.core.table_free_entries);
  end
  // 独立绝对周期观测；不参与驱动，不改变旧 TRACE 文件格式。
  task automatic observe_control(input string event_name, input tile_prefetch_package::tile_descriptor_t desc,
                                 input logic [7:0] slot_generation);
    $display("MAIN_CTL event=%s layer=%0d cycle=%0d expert=%0d tile=%0d matrix=%0d slot=%0d generation=%0d epoch=%0d",
      event_name,desc.layer_id,cycle,desc.expert_id,desc.tile_id,desc.matrix_kind,
      desc.ocm_slot,slot_generation,desc.epoch_id);
  endtask
  wire [63:0] alignment_credits={32'(dut.core.memory_dataflow.core.table_free_entries),
    16'(dut.core.memory_dataflow.core.ocm_occupancy),14'd0,
    dut.core.memory_dataflow.core.demand_pair_ready,dut.core.memory_dataflow.core.speculative_pair_ready};
  logic [63:0] alignment_credits_previous;
  always @(posedge clk) alignment_credits_previous <= alignment_credits;
  // 一拍后观察真实 metadata FREE，不以全局 occupancy 降低代替 slot/generation 回收。
  logic alignment_gu_release_previous;
  logic alignment_down_release_previous;
  tile_prefetch_package::tile_descriptor_t alignment_gu_release_descriptor;
  tile_prefetch_package::tile_descriptor_t alignment_down_release_descriptor;
  logic [7:0] alignment_gu_release_generation;
  logic [7:0] alignment_down_release_generation;
  always @(posedge clk) alignment_gu_release_previous <= rst_n && dut.core.compute_gu_release_valid;
  always @(posedge clk) alignment_down_release_previous <= rst_n && compute_done;
  always @(posedge clk) if(dut.core.compute_gu_release_valid) alignment_gu_release_descriptor <= dut.core.auto_gu_descriptor;
  always @(posedge clk) if(compute_done) alignment_down_release_descriptor <= dut.core.auto_down_descriptor;
  always @(posedge clk) if(dut.core.compute_gu_release_valid) alignment_gu_release_generation <= dut.core.auto_gu_generation;
  always @(posedge clk) if(compute_done) alignment_down_release_generation <= dut.core.auto_down_generation;
  always @(posedge clk) begin
    if ($test$plusargs("ALIGN_MAIN")) begin
      if(rst_n && alignment_gu_release_previous &&
          dut.core.memory_dataflow.core.ocm.state_mem[int'(alignment_gu_release_descriptor.ocm_slot)]==tile_prefetch_package::META_FREE &&
          dut.core.memory_dataflow.core.ocm.generation_mem[int'(alignment_gu_release_descriptor.ocm_slot)]==alignment_gu_release_generation)
        observe_control("SLOT_FREE",alignment_gu_release_descriptor,alignment_gu_release_generation);
      if(rst_n && alignment_down_release_previous &&
          dut.core.memory_dataflow.core.ocm.state_mem[int'(alignment_down_release_descriptor.ocm_slot)]==tile_prefetch_package::META_FREE &&
          dut.core.memory_dataflow.core.ocm.generation_mem[int'(alignment_down_release_descriptor.ocm_slot)]==alignment_down_release_generation)
        observe_control("SLOT_FREE",alignment_down_release_descriptor,alignment_down_release_generation);
      if(rst_n && dut.core.automatic_execution.execution.truth_ready && !dut.core.automatic_execution.execution.running)
        $display("MAIN_BLOCK cycle=%0d layer=%0d pending=%0d stale=%0d combine1=%0d combine2=%0d gu=%0d down=%0d",
          cycle,layer,dut.core.automatic_execution.execution.issue_pending,dut.core.automatic_execution.execution.search_stale,
          dut.core.automatic_execution.execution.combine_valid1,dut.core.automatic_execution.execution.combine_valid2,
          dut.core.automatic_execution.execution.gu_found,dut.core.automatic_execution.execution.down_found);
      if(rst_n && alignment_credits!=alignment_credits_previous)
        $display("MAIN_CREDIT cycle=%0d descriptor_free=%0d occupancy=%0d demand_pair=%0d spec_pair=%0d",
          cycle,dut.core.memory_dataflow.core.table_free_entries,dut.core.memory_dataflow.core.ocm_occupancy,
          dut.core.memory_dataflow.core.demand_pair_ready,dut.core.memory_dataflow.core.speculative_pair_ready);
      if(rst_n && dut.chain_retiring)
        $display("MAIN_RETIRE cycle=%0d layer=%0d state=%0d ingress=%0d ready=%0d allocator=%0d demand=%0d resolver=%0d compute=%0d",
          cycle,layer,observed_chain_state,dut.core.memory_ingress_idle,dut.core_retire_ready,
          dut.core.allocation_active,dut.core.demand_active,dut.core.resolver_accept_ready,dut.core.compute_busy);
      if(dut.core.automatic_execution.execution.select_for_issue)
        observe_control("SELECT",dut.core.automatic_execution.execution.selected_gu,
                        dut.core.automatic_execution.execution.selected_gu_generation);
      if(dut.core.auto_req_valid && dut.core.memory_compute_ready && dut.core.sequence_compute_ready && !transaction_error)
        observe_control("COMPUTE_ACCEPT",dut.core.auto_gu_descriptor,dut.core.auto_gu_generation);
      if(dut.core.physical_accept_valid)
        observe_control("ADMIT",dut.core.physical_accept_descriptor,dut.core.physical_accept_slot_generation);
      if(dut.core.dma_completion_valid)
        observe_control("READY",dut.core.dma_completion_descriptor,dut.core.memory_dataflow.dma_active_slot_generation);
      if(dut.core.compute_gu_release_valid)
        observe_control("RELEASE_GU",dut.core.auto_gu_descriptor,dut.core.auto_gu_generation);
      if(compute_done)
        observe_control("RELEASE_DOWN",dut.core.auto_down_descriptor,dut.core.auto_down_generation);
      for(integer bank=0;bank<4;bank=bank+1) begin
        if(arvalid[bank] && arready[bank])
          $display("MAIN_AXI event=AR cycle=%0d bank=%0d last=0",cycle,bank);
        if(rvalid[bank] && rready[bank])
          $display("MAIN_AXI event=R cycle=%0d bank=%0d last=%0d",cycle,bank,rlast[bank]);
      end
      if(dut.core.memory_dataflow.payload_valid)
        $display("MAIN_JOIN cycle=%0d last=%0d",cycle,dut.core.memory_dataflow.payload_last);
      if(dut.core.dma_completion_valid)
        $display("MAIN_READY layer=%0d cycle=%0d expert=%0d tile=%0d matrix=%0d",
          dut.core.dma_completion_descriptor.layer_id,cycle,dut.core.dma_completion_descriptor.expert_id,
          dut.core.dma_completion_descriptor.tile_id,dut.core.dma_completion_descriptor.matrix_kind);
      if(future_valid && future_ready)
        $display("MAIN_PREDICTOR layer=%0d cycle=%0d expert=%0d",future_layer,cycle,future_expert);
      if(chain_done)
        $display("MAIN_OCM peak_slots=%0d",dut.core.memory_dataflow.telemetry_ocm_peak_occupancy);
      if(chain_done)
        $display("MAIN_CREDITS descriptors=%0d demand=%0d promoted=%0d speculative=%0d",alignment_descriptor_peak,
          dut.core.memory_dataflow.core.unused_fifo_peak,dut.core.memory_dataflow.core.unused_promoted_peak,
          dut.core.memory_dataflow.core.unused_speculative_peak);
      if(dut.core.physical_accept_valid)
        $display("MAIN_ADMIT layer=%0d cycle=%0d expert=%0d tile=%0d matrix=%0d bytes=%0d class=%0d",
          dut.core.physical_accept_descriptor.layer_id,cycle,
          dut.core.physical_accept_descriptor.expert_id,dut.core.physical_accept_descriptor.tile_id,
          dut.core.physical_accept_descriptor.matrix_kind,dut.core.physical_accept_descriptor.byte_count,
          dut.core.physical_accept_descriptor.request_class);
      if(dut.core.memory_dataflow.dma_burst_start)
        $display("MAIN_DMA layer=%0d cycle=%0d expert=%0d tile=%0d matrix=%0d bytes=%0d class=%0d",
          dut.core.memory_dataflow.dma_active_descriptor.layer_id,cycle,
          dut.core.memory_dataflow.dma_active_descriptor.expert_id,dut.core.memory_dataflow.dma_active_descriptor.tile_id,
          dut.core.memory_dataflow.dma_active_descriptor.matrix_kind,dut.core.memory_dataflow.dma_burst_bytes,
          dut.core.memory_dataflow.dma_active_descriptor.request_class);
      if(dut.core.effective_route_expert_valid && dut.core.effective_route_expert_ready)
        $display("MAIN_TRUTH layer=%0d cycle=%0d expert=%0d weight=%0d",layer,cycle,
          dut.core.effective_route_expert,dut.core.effective_route_weight_q15);
      if(dut.core.memory_dataflow.backend_gu_start)
        $display("MAIN_STAGE layer=%0d cycle=%0d expert=%0d tile=%0d stage=GU_START",layer,cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_gu_done)
        $display("MAIN_STAGE layer=%0d cycle=%0d expert=%0d tile=%0d stage=GU_DONE",layer,cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_down_start)
        $display("MAIN_STAGE layer=%0d cycle=%0d expert=%0d tile=%0d stage=DOWN_START",layer,cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
      if(dut.core.memory_dataflow.backend_down_done)
        $display("MAIN_STAGE layer=%0d cycle=%0d expert=%0d tile=%0d stage=DOWN_DONE",layer,cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
    end
  end
endmodule
