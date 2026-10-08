`timescale 1ns/1ps
/** 仅验证现有 HardwareGreedy-v1：宽候选、真实 Switch byte 几何、credit 和停顿。
 * 本 testbench 不包含 DMA/OCM/计算，不声称验证 Uniform/Ranked 或整层周期。
 */
module tb_wide_candidate_allocation #(
  parameter integer CANDIDATES = 8,
  parameter integer STALL = 0
);
  import tile_prefetch_package::*;
  localparam integer TILES = 24;
  localparam integer PHYSICAL_BYTES = 98304;
  localparam integer LOGICAL_BYTES = 196608;
  localparam integer BUDGET = 9437184;
  localparam integer ALLOCATIONS = BUDGET / LOGICAL_BYTES;
  logic clk = 0;
  logic rst_n = 0;
  logic transaction_begin = 0;
  logic candidate_accept = 0;
  logic [4:0] rank_in = 0;
  logic [11:0] confidence_in = 0;
  logic candidate_last = 0;
  integer cycle = 0;
  wire logical_valid;
  wire logical_ready = !STALL || (cycle % 7 >= 2);
  wire active;
  wire done;
  wire [31:0] credit;
  tile_descriptor_t gu;
  tile_descriptor_t down;
  tile_descriptor_t saved_gu;
  tile_descriptor_t saved_down;
  logic saved_stall = 0;
  logic [31:0] saved_credit;
  logic [31:0] confidence_words [0:CANDIDATES-1];
  logic [31:0] expected_words [0:ALLOCATIONS-1];
  integer depths [0:CANDIDATES-1];
  integer accepted = 0;
  integer covered = 0;
  string confidence_path;
  string expected_path;

  always #2.5 clk = ~clk;
  always @(posedge clk) cycle <= cycle + 1;

  candidate_prefetch_scheduler #(
    .MAX_CANDIDATES(CANDIDATES),
    .TILES_PER_EXPERT(TILES),
    .CHANNEL_COUNT(1),
    .GU_BYTE_RATIO(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .transaction_begin(transaction_begin),
    .stop_allocation(1'b0),
    .transaction_epoch(16'd2),
    .transaction_sequence(16'd1),
    .transaction_layer(3'd1),
    .speculative_budget_bytes(32'(BUDGET)),
    .channel_credit_bytes(32'(BUDGET)),
    .candidate_accept(candidate_accept),
    .candidate_epoch(16'd2),
    .candidate_layer(3'd1),
    .candidate_expert(10'(rank_in)),
    .candidate_rank(rank_in),
    .candidate_confidence(confidence_in),
    .candidate_last(candidate_last),
    .gu_base_addr(64'h10000000),
    .down_base_addr(64'h20000000),
    .expert_stride_bytes(32'd4718592),
    .physical_tile_bytes(32'(PHYSICAL_BYTES)),
    .logical_valid(logical_valid),
    .logical_ready(logical_ready),
    .logical_gu_descriptor(gu),
    .logical_down_descriptor(down),
    .allocation_active(active),
    .allocation_done(done),
    .remaining_credit_bytes(credit)
  );

  initial begin
    if (!$value$plusargs("CONFIDENCE=%s", confidence_path) ||
        !$value$plusargs("EXPECTED=%s", expected_path))
      $fatal(1, "missing deterministic golden input paths");
    $readmemh(confidence_path, confidence_words);
    $readmemh(expected_path, expected_words);
    for (integer e = 0; e < CANDIDATES; e = e + 1) depths[e] = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    transaction_begin = 1;
    @(negedge clk);
    transaction_begin = 0;
    for (integer e = 0; e < CANDIDATES; e = e + 1) begin
      candidate_accept = 1;
      rank_in = 5'(e);
      confidence_in = 12'(confidence_words[e]);
      candidate_last = e == CANDIDATES - 1;
      @(negedge clk);
    end
    candidate_accept = 0;
    candidate_last = 0;
    while (!done) begin
      @(posedge clk);
      if (saved_stall && (gu !== saved_gu || down !== saved_down || credit != saved_credit))
        $fatal(1, "descriptor or credit changed during backpressure");
      if (logical_valid && logical_ready) begin
        if (accepted >= ALLOCATIONS || {14'd0, gu.expert_id, gu.tile_id} !== expected_words[accepted])
          $fatal(1, "golden allocation mismatch index=%0d expert=%0d tile=%0d expected=%h",
                 accepted, gu.expert_id, gu.tile_id, expected_words[accepted]);
        if (gu.tile_id != depths[gu.expert_id] || gu.tile_id >= TILES ||
            gu.byte_count != PHYSICAL_BYTES || down.byte_count != PHYSICAL_BYTES ||
            gu.matrix_kind != MATRIX_GU || down.matrix_kind != MATRIX_DOWN ||
            gu.expert_id != down.expert_id || gu.tile_id != down.tile_id ||
            gu.request_class != REQUEST_SPECULATIVE || gu.memory_channel != 0 ||
            gu.external_addr != 64'h10000000 + gu.expert_id * 4718592 + gu.tile_id * PHYSICAL_BYTES ||
            down.external_addr != 64'h20000000 + gu.expert_id * 4718592 + gu.tile_id * PHYSICAL_BYTES)
          $fatal(1, "prefix/byte/address/descriptor contract failed");
        depths[gu.expert_id] = depths[gu.expert_id] + 1;
        $display("WIDE_ALLOC index=%0d cycle=%0d expert=%0d tile=%0d bytes=%0d",
                 accepted, cycle, gu.expert_id, gu.tile_id, LOGICAL_BYTES);
        accepted = accepted + 1;
      end
      saved_stall = logical_valid && !logical_ready;
      saved_gu = gu;
      saved_down = down;
      saved_credit = credit;
      @(negedge clk);
    end
    if (accepted != ALLOCATIONS || credit != 0 || active)
      $fatal(1, "allocation/credit conservation failed");
    for (integer e = 0; e < CANDIDATES; e = e + 1) begin
      if (depths[e] > 0) covered = covered + 1;
      $display("WIDE_DEPTH expert=%0d depth=%0d", e, depths[e]);
    end
    if (covered != CANDIDATES) $fatal(1, "expected wide candidate coverage");
    $display("PASS: wide HardwareGreedy-v1 candidates=%0d covered=%0d allocations=%0d bytes=%0d stall=%0d",
             CANDIDATES, covered, accepted, accepted * LOGICAL_BYTES, STALL);
    $finish;
  end

  initial begin
    repeat (50000) @(posedge clk);
    $fatal(1, "wide allocation watchdog expired");
  end
endmodule
