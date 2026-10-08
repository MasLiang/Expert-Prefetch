`timescale 1ns/1ps
/** Deferred classification and maximum-byte conservation across the counter pipeline. */
module tb_byte_finalizer_pipeline #(
  parameter integer ENTRIES = 3
);
  import tile_prefetch_package::*;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n = 0;
  logic capture_valid = 0;
  logic burst_done = 0;
  logic soft_flush_valid = 0;
  descriptor_token_t token;
  logic [63:0] useful_bytes;
  logic [63:0] wasted_bytes;
  logic [63:0] cancelled_bytes;
  localparam integer CLASSIFICATION_TO_COUNTER_CYCLES = ENTRIES > 8 ? 3 : 2;
  localparam logic [63:0] EXPECTED_BYTES = 64'(ENTRIES) * 64'hffffffff;

  speculative_byte_finalizer #(
    .ENTRIES(ENTRIES)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .soft_flush_valid(soft_flush_valid),
    .soft_flush_epoch(16'd1),
    .capture_valid(capture_valid),
    .capture_token(token),
    .capture_bytes(32'hffffffff),
    .capture_epoch(16'd1),
    .burst_done(burst_done),
    .burst_token(token),
    .burst_bytes(32'hffffffff),
    .dma_done(1'b1),
    .route_valid(1'b0),
    .route_token(token),
    .route_truth(1'b0),
    .cancelled_retired(1'b0),
    .retired_token(token),
    .useful_bytes(useful_bytes),
    .wasted_bytes(wasted_bytes),
    .cancelled_bytes(cancelled_bytes)
  );

  initial begin
    token = '0;
    repeat (2) @(negedge clk);
    rst_n = 1;
    for (integer i = 0; i < ENTRIES; i = i + 1) begin
      token.request_id = 16'(i);
      token.generation = 1;
      capture_valid = 1;
      @(negedge clk);
      capture_valid = 0;
      burst_done = 1;
      @(negedge clk);
      burst_done = 0;
      if (useful_bytes != 0 || wasted_bytes != 0 || cancelled_bytes != 0)
        $fatal(1, "DMA before truth was classified early");
    end
    // 同一拍分类所有记录；宽度覆盖单项32-bit上界，精确检查流水延迟。
    soft_flush_valid = 1;
    @(negedge clk);
    soft_flush_valid = 0;
    if (wasted_bytes != 0) $fatal(1, "flush bypassed deferred pipeline");
    // 旧记录分类/退休的同一拍重用entry0；快照必须保留旧generation字节。
    token.request_id = 0;
    token.generation = 2;
    capture_valid = 1;
    repeat (CLASSIFICATION_TO_COUNTER_CYCLES-1) begin
      @(negedge clk);
      capture_valid = 0;
      if (wasted_bytes != 0) $fatal(1, "delta stage published early");
    end
    @(negedge clk);
    if (wasted_bytes != EXPECTED_BYTES || useful_bytes != 0 || cancelled_bytes != 0)
      $fatal(1, "full-width finalizer sum mismatch");
    repeat (6) begin
      @(negedge clk);
      if (wasted_bytes != EXPECTED_BYTES) $fatal(1, "duplicate finalization");
    end
    burst_done = 1;
    @(negedge clk);
    burst_done = 0;
    soft_flush_valid = 1;
    @(negedge clk);
    soft_flush_valid = 0;
    repeat (CLASSIFICATION_TO_COUNTER_CYCLES) @(negedge clk);
    if (wasted_bytes != EXPECTED_BYTES + 64'hffffffff || useful_bytes != 0 || cancelled_bytes != 0)
      $fatal(1, "generation reuse lost or duplicated a finalized contribution");
    $display("PASS: deferred byte pipeline ENTRIES=%0d latency=%0d and 64-bit conservation", ENTRIES, CLASSIFICATION_TO_COUNTER_CYCLES);
    $finish;
  end
  initial begin
    #10000;
    $fatal(1, "finalizer pipeline timeout");
  end
endmodule
