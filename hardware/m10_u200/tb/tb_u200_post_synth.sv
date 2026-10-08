`timescale 1ns/1ps
/** 完整默认 U200 配置的端口级自检；RTL 与最终 funcsim 网表共用。
 * 单层 Top-2，全部16个 tile；网表仅端口观察，RTL额外只读指标；无参数覆盖，无 SDF。
 */
module tb_u200_post_synth;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic pred_begin = 0, pred_ready, candidate_valid = 0, candidate_ready;
  logic weight_valid = 0, weight_ready, weight_last = 0, loaded, load_error;
  logic [31:0] weight_index = 0;
  logic [127:0] weight_data = 0;
  logic result_valid, parser_error, config_error, transaction_error;
  logic compute_done, gemv_done, physical_accept_valid;
  tile_descriptor_t admitted;
`ifdef NETLIST
  wire net_matrix_kind;
  wire net_request_class;
  assign admitted.matrix_kind = matrix_kind_t'(net_matrix_kind);
  assign admitted.request_class = request_class_t'(net_request_class);
`endif
  logic [511:0] result_y;
  logic [255:0] araddr;
  logic [31:0] arlen, response_errors, payload_errors;
  logic [11:0] arsize;
  logic [7:0] arburst;
  logic [3:0] arvalid, arready, rvalid = 0, rready, rlast = 0, active = 0;
  logic [511:0] rdata = 0;
  logic [7:0] rresp = 0;
  logic [63:0] spec, demand, useful, wasted, cancelled;
  logic [63:0] admitted_bitmap = 0;
  integer left [0:3], cursor [0:3];
  integer cycle = 0, compute_count = 0, router_count = 0, admission_count = 0;
  integer accepted_bytes = 0, returned_bytes = 0, bank_bursts [0:3];
  integer layer_start_cycle, layer_finish_cycle, identity_index;
  integer predicted_expert = 3, budget_bytes = 768, ddr_gaps = 1;
  integer wrong_admissions = 0, trace_file = 0;
  integer weight_wait_cycles = 0;
  logic nonlinear_previous = 0;
  string trace_path;
`ifndef NETLIST
  integer expert_issue_idle_cycles = 0;
  integer memory_wait_cycles = 0, ready_control_wait_cycles = 0;
  integer legacy_memory_wait_cycles = 0, legacy_control_wait_cycles = 0;
  integer issue_pending_wait_cycles = 0, snapshot_refresh_wait_cycles = 0;
  integer peak_banks [0:3];
  bit boundary_window_previous = 0, boundary_truth_previous = 0;
  bit boundary_combine_previous = 0;
  bit boundary_payload_seen [0:127];
  integer boundary_payload_generation [0:127];
  integer boundary_combine_expert = -1;
  integer demand_fifo_peak = 0, descriptor_peak = 0;
  initial begin
    for (integer bank = 0; bank < 4; bank = bank + 1) peak_banks[bank] = 0;
    for (integer slot = 0; slot < 128; slot = slot + 1) boundary_payload_seen[slot] = 0;
    for (integer slot = 0; slot < 128; slot = slot + 1) boundary_payload_generation[slot] = -1;
  end
`endif

  // 不覆盖任何参数：必须匹配默认配置最终综合 DCP。
  u200_moe_top dut (
    .clk(clk),
    .rst_n(rst_n),
    .bandwidth_enable(1'b1),
    .setup_cycles(32'd0),
    .sequence_active(1'b1),
    .layer_done(1'b0),
    .prediction_begin_valid(pred_begin),
    .prediction_begin_ready(pred_ready),
    .prediction_epoch(16'd1),
    .prediction_sequence(16'd1),
    .prediction_layer(3'd0),
    .prediction_count(5'd1),
    .speculative_budget_bytes(32'(budget_bytes)),
    .channel_credit_bytes(32'(budget_bytes)),
    .candidate_valid(candidate_valid),
    .candidate_ready(candidate_ready),
    .candidate_expert(10'(predicted_expert)),
    .candidate_rank(5'd0),
    .candidate_confidence(12'd3000),
    .candidate_last(1'b1),
    .gu_base_addr(64'h10000),
    .down_base_addr(64'h20000),
    .expert_stride_bytes(32'd16384),
    .physical_tile_bytes(32'd256),
    .soft_flush_valid(1'b0),
    .soft_flush_epoch(16'd1),
    .compute_activation({16{8'd16}}),
    .gu_requant_shift(5'd0),
    .router_top_k_config(5'd2),
    .router_logit_shift(5'd0),
    .router_weight_reload_valid(1'b0),
    .router_weight_layer(3'd0),
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
    .physical_accept_valid(physical_accept_valid),
`ifdef NETLIST
    // Vivado write_verilog 将 packed struct 顶层端口展开为 escaped 字段名。
    .\physical_accept_descriptor[epoch_id] (admitted.epoch_id),
    .\physical_accept_descriptor[layer_id] (admitted.layer_id),
    .\physical_accept_descriptor[sequence_id] (admitted.sequence_id),
    .\physical_accept_descriptor[expert_id] (admitted.expert_id),
    .\physical_accept_descriptor[tile_id] (admitted.tile_id),
    .\physical_accept_descriptor[matrix_kind] (net_matrix_kind),
    .\physical_accept_descriptor[external_addr] (admitted.external_addr),
    .\physical_accept_descriptor[byte_count] (admitted.byte_count),
    .\physical_accept_descriptor[ocm_bank] (admitted.ocm_bank),
    .\physical_accept_descriptor[ocm_slot] (admitted.ocm_slot),
    .\physical_accept_descriptor[memory_channel] (admitted.memory_channel),
    .\physical_accept_descriptor[request_class] (net_request_class),
    .\physical_accept_descriptor[candidate_rank] (admitted.candidate_rank),
    .\physical_accept_descriptor[confidence_q12] (admitted.confidence_q12),
`else
    .physical_accept_descriptor(admitted),
`endif
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

  always_comb begin
    for (integer bank = 0; bank < 4; bank = bank + 1)
      arready[bank] = !active[bank] && (ddr_gaps == 0 || cycle % (bank + 2) != 0);
  end
  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (compute_done) begin
      if (router_count != 1) $fatal(1, "Expert completion before Router");
      compute_count = compute_count + 1;
    end
    if (gemv_done) router_count = router_count + 1;
    if (physical_accept_valid) begin
      if ((admitted.expert_id != 3 && admitted.expert_id != 4 &&
           !(admitted.expert_id == predicted_expert && admitted.request_class == REQUEST_SPECULATIVE)) ||
          admitted.tile_id >= 16 || admitted.ocm_slot >= 128 ||
          admitted.epoch_id != 1 || admitted.layer_id != 0)
        $fatal(1, "invalid admitted descriptor");
      if (admitted.byte_count != (admitted.matrix_kind == MATRIX_GU ? 512 : 256))
        $fatal(1, "invalid logical payload bytes");
      if (admitted.ocm_bank != ((1 ^ admitted.expert_id ^ admitted.tile_id ^ admitted.matrix_kind) % 4) ||
          int'(admitted.ocm_slot) / 32 != admitted.ocm_bank)
        $fatal(1, "banked OCM placement mismatch");
      if (admitted.external_addr !=
          (admitted.matrix_kind == MATRIX_GU ? 64'h10000 : 64'h20000) +
          admitted.expert_id * 16384 + admitted.tile_id * admitted.byte_count)
        $fatal(1, "descriptor address mismatch");
      if (admitted.expert_id == 3 || admitted.expert_id == 4) begin
        identity_index = (int'(admitted.expert_id) - 3) * 32 +
          int'(admitted.tile_id) * 2 + int'(admitted.matrix_kind);
        if (admitted_bitmap[identity_index]) $fatal(1, "duplicate admission");
        admitted_bitmap[identity_index] = 1;
      end else wrong_admissions = wrong_admissions + 1;
      admission_count = admission_count + 1;
    end
    for (integer bank = 0; bank < 4; bank = bank + 1) begin
      if (arvalid[bank] && arready[bank]) begin
        if (arsize[bank*3 +: 3] != 4 || arburst[bank*2 +: 2] != 1)
          $fatal(1, "invalid DDR burst format");
        active[bank] <= 1;
        left[bank] <= int'(arlen[bank*8 +: 8]) + 1;
        cursor[bank] <= int'(araddr[bank*64 +: 64]) * 4 + bank * 16;
        bank_bursts[bank] = bank_bursts[bank] + 1;
        accepted_bytes = accepted_bytes + (int'(arlen[bank*8 +: 8]) + 1) * 16;
      end
      if (rvalid[bank] && rready[bank]) begin
        rvalid[bank] <= 0;
        cursor[bank] <= cursor[bank] + 64;
        left[bank] <= left[bank] - 1;
        returned_bytes = returned_bytes + 16;
        if (left[bank] == 1) active[bank] <= 0;
      end else if (active[bank] && !rvalid[bank] && (ddr_gaps == 0 || cycle % (bank + 3) != 0)) begin
        rvalid[bank] <= 1;
        rlast[bank] <= left[bank] == 1;
        // GU 的 G/U 与 DOWN 均为16×16 identity，独立解析 stripe 地址。
        rdata[bank*128 +: 128] <= 128'd1 << (((cursor[bank] / 16) % 16) * 8);
      end
    end
  end

  initial begin
    if ($value$plusargs("PREDICTED_EXPERT=%d", predicted_expert)) begin end
    if ($value$plusargs("BUDGET=%d", budget_bytes)) begin end
    if ($value$plusargs("DDR_GAPS=%d", ddr_gaps)) begin end
    if (predicted_expert < 3 || predicted_expert > 5 ||
        (budget_bytes != 0 && budget_bytes != 768) || (ddr_gaps != 0 && ddr_gaps != 1))
      $fatal(1, "unsupported shared fixture inputs");
    if ($value$plusargs("TRACE=%s", trace_path)) begin
      trace_file = $fopen(trace_path, "w");
      if (trace_file == 0) $fatal(1, "cannot open event trace");
      $fdisplay(trace_file, "cycle,event,expert_id,tile_id,matrix,bytes,request_type,value");
    end
    for (integer bank = 0; bank < 4; bank = bank + 1) bank_bursts[bank] = 0;
    // UNISIM glbl 的 GSR 初始100ns，显式等待后再解除 testbench reset。
    repeat (40) @(negedge clk);
    rst_n = 1;
    pred_begin = 1;
    #1;
    if (pred_ready !== 0 || config_error !== 1) $fatal(1, "unloaded Router accepted");
    pred_begin = 0;
    for (weight_index = 0; weight_index < 64; weight_index = weight_index + 1) begin
      weight_data = 0;
      weight_data[7:0] = (weight_index == 3 || weight_index == 4) ? 8'd1 : 8'hff;
      weight_last = weight_index == 63;
      weight_valid = 1;
      #1;
      if (weight_ready !== 1) $fatal(1, "Router weight load backpressure");
      @(negedge clk);
    end
    weight_valid = 0;
    weight_last = 0;
    if (loaded !== 1 || load_error !== 0) $fatal(1, "Router load failed");
    pred_begin = 1;
    do @(posedge clk); while (!pred_ready);
    layer_start_cycle = cycle;
    @(negedge clk);
    pred_begin = 0;
    candidate_valid = 1;
    do @(posedge clk); while (!candidate_ready);
    @(negedge clk);
    candidate_valid = 0;
    wait (result_valid === 1);
    layer_finish_cycle = cycle;
    repeat (10) @(negedge clk);
    for (integer lane = 0; lane < 16; lane = lane + 1)
      // SiLU_Q4(16)=12，GU product=12；16 tiles→192，等权Top-2仍为192。
      if ($signed(result_y[lane*32 +: 32]) !== 32'sd192)
        $fatal(1, "numeric lane=%0d got=%0d expected=192", lane, $signed(result_y[lane*32 +: 32]));
    if ({parser_error, config_error, transaction_error} !== 3'b000 ||
        compute_count != 32 || router_count != 1 || admission_count != 64 + wrong_admissions ||
        admitted_bitmap !== 64'hffffffffffffffff)
      $fatal(1, "protocol or tile coverage mismatch");
    if (spec !== (budget_bytes == 0 ? 0 : 768) ||
        demand !== (budget_bytes > 0 && predicted_expert != 5 ? 23808 : 24576) ||
        useful !== (budget_bytes > 0 && predicted_expert != 5 ? 768 : 0) ||
        wasted !== (budget_bytes > 0 && predicted_expert == 5 ? 768 : 0) || cancelled !== 0)
      $fatal(1, "byte classification mismatch spec=%0d demand=%0d useful=%0d wasted=%0d cancelled=%0d", spec, demand, useful, wasted, cancelled);
    if (accepted_bytes != 24576 + (budget_bytes > 0 && predicted_expert == 5 ? 768 : 0) ||
        returned_bytes != accepted_bytes || active !== 0 ||
        response_errors !== 0 || payload_errors !== 0)
      $fatal(1, "DDR byte conservation/error mismatch");
    for (integer bank = 0; bank < 4; bank = bank + 1)
      if (bank_bursts[bank] != 96 + (budget_bytes > 0 && predicted_expert == 5 ? 3 : 0))
        $fatal(1, "DDR bank burst coverage");
    if (wrong_admissions != (budget_bytes > 0 && predicted_expert == 5 ? 2 : 0))
      $fatal(1, "wrong speculative admission count");
    $display("TEST_PASS: default U200 Top-2 16 tiles, pred=%0d budget=%0d gaps=%0d", predicted_expert, budget_bytes, ddr_gaps);
    $display("RUN_METRICS cycles=%0d simulated_ps=%0t spec=%0d demand=%0d result_y=%h", layer_finish_cycle-layer_start_cycle, $time, spec, demand, result_y);
`ifndef NETLIST
    if (expert_issue_idle_cycles != memory_wait_cycles + ready_control_wait_cycles ||
        weight_wait_cycles != legacy_memory_wait_cycles + legacy_control_wait_cycles ||
        legacy_memory_wait_cycles > memory_wait_cycles || legacy_control_wait_cycles > ready_control_wait_cycles)
      $fatal(1, "wait partition conservation failed");
    $display("WAIT_METRICS %0d %0d %0d %0d %0d %0d %0d", expert_issue_idle_cycles,
      memory_wait_cycles, ready_control_wait_cycles, legacy_memory_wait_cycles,
      legacy_control_wait_cycles, issue_pending_wait_cycles, snapshot_refresh_wait_cycles);
    $display("ALIGN_METRICS %0d %0d %0d %0d %0d %0d %0d %0d %0d %h", layer_finish_cycle-layer_start_cycle,
      spec, demand, useful, wasted, cancelled, dut.core.demand_stall_cycles, weight_wait_cycles,
      dut.core.memory_dataflow.telemetry_ocm_peak_occupancy, result_y);
    $display("RESOURCE_METRICS %0d %0d %0d %0d %0d %0d", peak_banks[0], peak_banks[1],
      peak_banks[2], peak_banks[3], demand_fifo_peak, descriptor_peak);
`endif
    $display("PORT_METRICS %0d %0d %0d %0d %0d %0d %0d %0d %0d %h", layer_finish_cycle-layer_start_cycle,
      spec, demand, useful, wasted, cancelled, admission_count, compute_count, returned_bytes, result_y);
    if (trace_file != 0) $fclose(trace_file);
    $finish;
  end

  // 端口事件可用于RTL/网表逐交易比较；内部事件只在RTL参考运行中观测。
  always @(posedge clk) begin
    if (pred_begin && pred_ready) layer_start_cycle = cycle;
    if (trace_file != 0 && pred_begin && pred_ready)
      $fdisplay(trace_file, "0,LAYER_START,-1,-1,none,0,none,0");
    if (trace_file != 0 && candidate_valid && candidate_ready)
      $fdisplay(trace_file, "%0d,PREDICTOR_READY,%0d,0,none,0,none,3000", cycle-layer_start_cycle, predicted_expert);
    if (trace_file != 0 && physical_accept_valid)
      $fdisplay(trace_file, "%0d,ADMIT,%0d,%0d,%0d,%0d,%0d,%0d", cycle-layer_start_cycle,
        admitted.expert_id, admitted.tile_id, admitted.matrix_kind, admitted.byte_count, admitted.request_class, admitted.ocm_slot);
    if (trace_file != 0 && gemv_done)
      $fdisplay(trace_file, "%0d,ROUTER_DONE,-1,-1,router,0,none,0", cycle-layer_start_cycle);
    if (trace_file != 0 && compute_done)
      $fdisplay(trace_file, "%0d,COMPUTE_DONE,-1,-1,none,0,none,0", cycle-layer_start_cycle);
    if (trace_file != 0 && result_valid)
      $fdisplay(trace_file, "%0d,LAYER_DONE,-1,-1,none,0,none,0", cycle-layer_start_cycle);
`ifndef NETLIST
    if (rst_n) begin
      for (integer bank = 0; bank < 4; bank = bank + 1) begin
        integer occupied;
        occupied = 0;
        for (integer slot = 0; slot < 32; slot = slot + 1)
          if (dut.core.memory_dataflow.core.ocm.state_mem[bank*32+slot] != META_FREE)
            occupied = occupied + 1;
        if (occupied > peak_banks[bank]) peak_banks[bank] = occupied;
      end
      if (dut.core.memory_dataflow.core.dma.demand_count > 16 ||
          dut.core.memory_dataflow.core.dma.spec_count > 16 ||
          dut.core.memory_dataflow.core.dma.promoted_count > 16)
        $fatal(1, "DMA FIFO exceeded configured capacity");
      if (dut.core.memory_dataflow.core.dma.demand_count > demand_fifo_peak)
        demand_fifo_peak = int'(dut.core.memory_dataflow.core.dma.demand_count);
      if (64-int'(dut.core.memory_dataflow.core.descriptors.free_entry_count) > descriptor_peak)
        descriptor_peak = 64-int'(dut.core.memory_dataflow.core.descriptors.free_entry_count);
    end
    nonlinear_previous <= dut.core.memory_dataflow.backend_nonlinear_done;
    if (rst_n && dut.core.automatic_execution.execution.truth_ready &&
        !dut.core.automatic_execution.execution.running && !dut.core.automatic_execution.execution.issue_pending &&
        !dut.core.automatic_execution.execution.combine_valid1 && !dut.core.automatic_execution.execution.combine_valid2 &&
        !dut.core.automatic_execution.execution.gu_found &&
        dut.core.automatic_execution.execution.completed_experts < dut.core.automatic_execution.execution.truth_count)
      weight_wait_cycles = weight_wait_cycles + 1;
    // 在posedge、NBA写回前读取当前状态，计数当前采样区间。绕过gu_match/gu_found
    // 注册快照，从带generation的物理OCM完整payload判定所有真实Expert的下一prefix。
    // 只用同一个真实Expert下一prefix的GU payload判断启动；DOWN在其阶段单独检查。
    if (rst_n && dut.core.automatic_execution.execution.truth_ready &&
        !dut.core.automatic_execution.execution.running &&
        !dut.core.automatic_execution.execution.combine_valid1 &&
        !dut.core.automatic_execution.execution.combine_valid2 &&
        dut.core.automatic_execution.execution.completed_experts < dut.core.automatic_execution.execution.truth_count) begin : wait_observation
      bit any_gu_ready;
      bit gu_ready;
      tile_descriptor_t entry;
      any_gu_ready = 0;
      for (integer expert = 0; expert < 8; expert = expert + 1) begin
        gu_ready = 0;
        if (expert < int'(dut.core.automatic_execution.execution.truth_count) &&
            int'(dut.core.automatic_execution.execution.next_tiles[expert]) < 16) begin
          for (integer slot = 0; slot < 128; slot = slot + 1) begin
            entry = dut.core.automatic_execution.execution.descriptor_mem[slot];
            if (dut.core.automatic_execution.execution.valid_mem[slot] &&
                entry.expert_id == dut.core.automatic_execution.execution.true_ids[expert] &&
                entry.tile_id == dut.core.automatic_execution.execution.next_tiles[expert] &&
                dut.core.memory_dataflow.core.ocm.payload_complete_mem[slot] &&
                !dut.core.memory_dataflow.core.ocm.quarantined_mem[slot] &&
                !dut.core.memory_dataflow.core.ocm.cancelled_mem[slot] &&
                dut.core.memory_dataflow.core.ocm.state_mem[slot] != META_FREE &&
                dut.core.memory_dataflow.core.ocm.epoch_mem[slot] == entry.epoch_id &&
                dut.core.memory_dataflow.core.ocm.generation_mem[slot] ==
                  dut.core.automatic_execution.execution.generation_mem[slot]) begin
              if (entry.matrix_kind == MATRIX_GU) gu_ready = 1;
            end
          end
          any_gu_ready = any_gu_ready || gu_ready;
        end
      end
      expert_issue_idle_cycles = expert_issue_idle_cycles + 1;
      if (any_gu_ready) ready_control_wait_cycles = ready_control_wait_cycles + 1;
      else memory_wait_cycles = memory_wait_cycles + 1;
      if (dut.core.automatic_execution.execution.issue_pending)
        issue_pending_wait_cycles = issue_pending_wait_cycles + 1;
      if (any_gu_ready && !dut.core.automatic_execution.execution.gu_found)
        snapshot_refresh_wait_cycles = snapshot_refresh_wait_cycles + 1;
      if (!dut.core.automatic_execution.execution.issue_pending &&
          !dut.core.automatic_execution.execution.gu_found) begin
        if (any_gu_ready) legacy_control_wait_cycles = legacy_control_wait_cycles + 1;
        else legacy_memory_wait_cycles = legacy_memory_wait_cycles + 1;
      end
      if (trace_file != 0) begin
        if (any_gu_ready)
          $fdisplay(trace_file,"%0d,WAIT_CONTROL,-1,-1,none,0,none,1",cycle-layer_start_cycle);
        else
          $fdisplay(trace_file,"%0d,WAIT_MEMORY,-1,-1,none,0,none,1",cycle-layer_start_cycle);
      end
    end
    if (trace_file != 0) begin
      // 统一功能边界：均是当前posedge前可见的状态或在该edge接受的握手。
      // 不借助running反推请求接受；PAYLOAD_READY扫描OCM物理完成标志。
      if (rst_n) begin : boundary_observation
        bit window_open;
        tile_descriptor_t entry;
        window_open = dut.core.automatic_execution.execution.truth_ready &&
          !dut.core.automatic_execution.execution.running &&
          !dut.core.automatic_execution.execution.combine_valid1 &&
          !dut.core.automatic_execution.execution.combine_valid2 &&
          dut.core.automatic_execution.execution.completed_experts < dut.core.automatic_execution.execution.truth_count;
        if (window_open && !boundary_window_previous)
          $fdisplay(trace_file,"%0d,TILE_ISSUE_WINDOW_OPEN,-1,-1,none,0,none,0",cycle-layer_start_cycle);
        if (dut.core.automatic_execution.execution.truth_ready && !boundary_truth_previous)
          $fdisplay(trace_file,"%0d,ROUTE_RESOLVED,-1,-1,none,0,none,0",cycle-layer_start_cycle);
        boundary_window_previous = window_open;
        boundary_truth_previous = dut.core.automatic_execution.execution.truth_ready;
        for (integer slot = 0; slot < 128; slot = slot + 1) begin
          entry = dut.core.automatic_execution.execution.descriptor_mem[slot];
          // 被取消的slot可直接capture新descriptor，valid未必先降为0。
          // 观察器同样必须按generation退休，不能继承旧的已记录状态。
          if (boundary_payload_generation[slot] != int'(dut.core.automatic_execution.execution.generation_mem[slot]))
            boundary_payload_seen[slot] = 0;
          boundary_payload_generation[slot] = int'(dut.core.automatic_execution.execution.generation_mem[slot]);
          if (!dut.core.automatic_execution.execution.valid_mem[slot]) boundary_payload_seen[slot] = 0;
          else if (!boundary_payload_seen[slot] && dut.core.memory_dataflow.core.ocm.payload_complete_mem[slot] &&
              !dut.core.memory_dataflow.core.ocm.quarantined_mem[slot] &&
              !dut.core.memory_dataflow.core.ocm.cancelled_mem[slot] &&
              dut.core.memory_dataflow.core.ocm.state_mem[slot] != META_FREE &&
              dut.core.memory_dataflow.core.ocm.epoch_mem[slot] == entry.epoch_id &&
              dut.core.memory_dataflow.core.ocm.generation_mem[slot] == dut.core.automatic_execution.execution.generation_mem[slot]) begin
            boundary_payload_seen[slot] = 1;
            if (entry.matrix_kind == MATRIX_GU)
              $fdisplay(trace_file,"%0d,PAYLOAD_READY,%0d,%0d,gate_up,%0d,%0d,0",cycle-layer_start_cycle,
                entry.expert_id,entry.tile_id,entry.byte_count,entry.request_class);
            else
              $fdisplay(trace_file,"%0d,PAYLOAD_READY,%0d,%0d,down,%0d,%0d,0",cycle-layer_start_cycle,
                entry.expert_id,entry.tile_id,entry.byte_count,entry.request_class);
          end
        end
        if (dut.core.memory_dataflow.core.claim_valid) begin
          entry = dut.core.memory_dataflow.core.active_descriptor;
          if (entry.matrix_kind == MATRIX_GU)
            $fdisplay(trace_file,"%0d,MEMORY_SERVICE_START,%0d,%0d,gate_up,%0d,%0d,0",cycle-layer_start_cycle,
              entry.expert_id,entry.tile_id,entry.byte_count,entry.request_class);
          else
            $fdisplay(trace_file,"%0d,MEMORY_SERVICE_START,%0d,%0d,down,%0d,%0d,0",cycle-layer_start_cycle,
              entry.expert_id,entry.tile_id,entry.byte_count,entry.request_class);
        end
        if (dut.core.automatic_execution.execution.compute_req_valid && dut.core.automatic_execution.execution.compute_req_ready)
          $fdisplay(trace_file,"%0d,COMPUTE_REQUEST_ACCEPTED,%0d,%0d,none,0,none,0",cycle-layer_start_cycle,
            dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
        if (dut.core.automatic_execution.execution.running && compute_done) begin
          $fdisplay(trace_file,"%0d,TILE_RETIRED,%0d,%0d,none,0,none,0",cycle-layer_start_cycle,
            dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
          if (dut.core.auto_gu_descriptor.tile_id == 15) begin
            boundary_combine_expert = int'(dut.core.auto_gu_descriptor.expert_id);
            $fdisplay(trace_file,"%0d,EXPERT_COMBINE_START,%0d,15,none,0,none,0",cycle-layer_start_cycle,
              boundary_combine_expert);
          end
        end
        if (boundary_combine_previous)
          $fdisplay(trace_file,"%0d,EXPERT_COMBINE_DONE,%0d,15,none,0,none,0",cycle-layer_start_cycle,
            boundary_combine_expert);
        boundary_combine_previous = dut.core.automatic_execution.execution.combine_valid2;
      end
      if (dut.core.router_start_pulse)
        $fdisplay(trace_file,"%0d,ROUTER_START,-1,-1,router,0,none,0",cycle-layer_start_cycle);
      if (dut.core.effective_route_expert_valid && dut.core.effective_route_expert_ready)
        $fdisplay(trace_file,"%0d,TRUTH,%0d,-1,none,0,none,%0d",cycle-layer_start_cycle,
          dut.core.effective_route_expert,dut.core.effective_route_weight_q15);
      if (dut.core.memory_dataflow.dma_burst_start)
        $fdisplay(trace_file,"%0d,DMA_START,%0d,%0d,%0d,%0d,%0d,0",cycle-layer_start_cycle,
          dut.core.memory_dataflow.dma_active_descriptor.expert_id,dut.core.memory_dataflow.dma_active_descriptor.tile_id,
          dut.core.memory_dataflow.dma_active_descriptor.matrix_kind,dut.core.memory_dataflow.dma_burst_bytes,
          dut.core.memory_dataflow.dma_active_descriptor.request_class);
      if (dut.core.memory_dataflow.adapter_burst_complete)
        $fdisplay(trace_file,"%0d,DMA_DONE,%0d,%0d,%0d,%0d,%0d,0",cycle-layer_start_cycle,
          dut.core.memory_dataflow.dma_active_descriptor.expert_id,dut.core.memory_dataflow.dma_active_descriptor.tile_id,
          dut.core.memory_dataflow.dma_active_descriptor.matrix_kind,dut.core.memory_dataflow.dma_burst_bytes,
          dut.core.memory_dataflow.dma_active_descriptor.request_class);
      if (dut.core.memory_dataflow.backend_gu_start)
        $fdisplay(trace_file,"%0d,GU_START,%0d,%0d,gate_up,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if (dut.core.memory_dataflow.backend_gu_done)
        $fdisplay(trace_file,"%0d,GU_DONE,%0d,%0d,gate_up,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if (dut.core.compute_gu_release_valid)
        $fdisplay(trace_file,"%0d,GU_SLOT_RELEASE,%0d,%0d,gate_up,0,none,%0d",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id,
          dut.core.auto_gu_generation);
      if (dut.core.memory_dataflow.backend_nonlinear_done && !nonlinear_previous)
        $fdisplay(trace_file,"%0d,NL_DONE,%0d,%0d,nonlinear,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_gu_descriptor.expert_id,dut.core.auto_gu_descriptor.tile_id);
      if (dut.core.memory_dataflow.backend_down_start)
        $fdisplay(trace_file,"%0d,DOWN_START,%0d,%0d,down,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
      if (dut.core.memory_dataflow.backend_down_done)
        $fdisplay(trace_file,"%0d,DOWN_DONE,%0d,%0d,down,0,none,0",cycle-layer_start_cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id);
      if (compute_done)
        $fdisplay(trace_file,"%0d,DOWN_SLOT_RELEASE,%0d,%0d,down,0,none,%0d",cycle-layer_start_cycle,
          dut.core.auto_down_descriptor.expert_id,dut.core.auto_down_descriptor.tile_id,
          dut.core.auto_down_generation);
    end
`endif
  end
  initial begin
    #1000000;
    $fatal(1, "watchdog phase cycle=%0d computes=%0d admissions=%0d", cycle, compute_count, admission_count);
  end
endmodule
