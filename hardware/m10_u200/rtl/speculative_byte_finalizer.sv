`timescale 1ns/1ps
/** Deferred byte classification: completed DMA before Router stays unclassified.
 * Counters use actual completed payload bytes, including partially issued cancel.
 */
module speculative_byte_finalizer #(
  parameter integer ENTRIES = 64
) (
  input logic clk,
  input logic rst_n,
  input logic soft_flush_valid,
  input logic [15:0] soft_flush_epoch,
  input logic capture_valid,
  input tile_prefetch_package::descriptor_token_t capture_token,
  input logic [31:0] capture_bytes,
  input logic [15:0] capture_epoch,
  input logic burst_done,
  input tile_prefetch_package::descriptor_token_t burst_token,
  input logic [31:0] burst_bytes,
  input logic dma_done,
  input logic route_valid,
  input tile_prefetch_package::descriptor_token_t route_token,
  input logic route_truth,
  input logic cancelled_retired,
  input tile_prefetch_package::descriptor_token_t retired_token,
  output logic [63:0] useful_bytes,
  output logic [63:0] wasted_bytes,
  output logic [63:0] cancelled_bytes
);
  logic live [0:ENTRIES-1];
  logic [7:0] generation [0:ENTRIES-1];
  logic [15:0] epoch [0:ENTRIES-1];
  logic [31:0] allocated [0:ENTRIES-1];
  logic [31:0] transferred [0:ENTRIES-1];
  logic resolved [0:ENTRIES-1];
  logic useful [0:ENTRIES-1];
  logic terminal [0:ENTRIES-1];
  // 每项最多 32-bit bytes；所有 entry 的增量和只需要 32+ceil(log2(N)) bits。
  // N<=8：分类 -> pending -> 64-bit counter；N>8：分类 -> group -> pending -> counter。
  // 每组至多8项，采用平衡加法树。分类当拍退休entry并快照增量，允许表项立即复用。
  localparam integer DELTA_W = 32 + $clog2(ENTRIES);
  localparam integer GROUP_COUNT = (ENTRIES + 7) / 8;
  localparam integer GROUP_LEAVES = 1 << $clog2(GROUP_COUNT);
  wire [DELTA_W-1:0] useful_tree [1:2*GROUP_LEAVES-1] /* verilator split_var */;
  wire [DELTA_W-1:0] wasted_tree [1:2*GROUP_LEAVES-1] /* verilator split_var */;
  wire [DELTA_W-1:0] cancelled_tree [1:2*GROUP_LEAVES-1] /* verilator split_var */;
  logic [DELTA_W-1:0] useful_pending;
  logic [DELTA_W-1:0] wasted_pending;
  logic [DELTA_W-1:0] cancelled_pending;

  for (genvar group = 0; group < GROUP_COUNT; group = group + 1) begin : byte_group
    localparam integer COUNT = ENTRIES-group*8 < 8 ? ENTRIES-group*8 : 8;
    localparam integer WIDTH = 32 + $clog2(COUNT);
    localparam integer LEAVES = 1 << $clog2(COUNT);
    wire [WIDTH-1:0] u_tree [1:2*LEAVES-1] /* verilator split_var */;
    wire [WIDTH-1:0] w_tree [1:2*LEAVES-1] /* verilator split_var */;
    wire [WIDTH-1:0] c_tree [1:2*LEAVES-1] /* verilator split_var */;
    for (genvar leaf = 0; leaf < LEAVES; leaf = leaf + 1) begin : entry_delta
      if (leaf < COUNT) begin : present
        localparam integer INDEX = group*8+leaf;
        wire classify = live[INDEX] && resolved[INDEX] && terminal[INDEX];
        assign u_tree[LEAVES+leaf] = classify && useful[INDEX] ? WIDTH'(transferred[INDEX]) : '0;
        assign w_tree[LEAVES+leaf] = classify && !useful[INDEX] ? WIDTH'(transferred[INDEX]) : '0;
        assign c_tree[LEAVES+leaf] = classify && !useful[INDEX] ?
          WIDTH'(allocated[INDEX])-WIDTH'(transferred[INDEX]) : '0;
      end else begin : padding
        assign u_tree[LEAVES+leaf] = '0;
        assign w_tree[LEAVES+leaf] = '0;
        assign c_tree[LEAVES+leaf] = '0;
      end
    end
    for (genvar node = 1; node < LEAVES; node = node + 1) begin : reduction
      assign u_tree[node] = u_tree[2*node] + u_tree[2*node+1];
      assign w_tree[node] = w_tree[2*node] + w_tree[2*node+1];
      assign c_tree[node] = c_tree[2*node] + c_tree[2*node+1];
    end
    if (ENTRIES > 8) begin : snapshot
      logic [WIDTH-1:0] u_pending;
      logic [WIDTH-1:0] w_pending;
      logic [WIDTH-1:0] c_pending;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) u_pending <= 0;
        else u_pending <= u_tree[1];
      end
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) w_pending <= 0;
        else w_pending <= w_tree[1];
      end
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) c_pending <= 0;
        else c_pending <= c_tree[1];
      end
      assign useful_tree[GROUP_LEAVES+group] = DELTA_W'(u_pending);
      assign wasted_tree[GROUP_LEAVES+group] = DELTA_W'(w_pending);
      assign cancelled_tree[GROUP_LEAVES+group] = DELTA_W'(c_pending);
    end else begin : bypass
      assign useful_tree[GROUP_LEAVES+group] = DELTA_W'(u_tree[1]);
      assign wasted_tree[GROUP_LEAVES+group] = DELTA_W'(w_tree[1]);
      assign cancelled_tree[GROUP_LEAVES+group] = DELTA_W'(c_tree[1]);
    end
  end
  for (genvar group = GROUP_COUNT; group < GROUP_LEAVES; group = group + 1) begin : group_padding
    assign useful_tree[GROUP_LEAVES+group] = '0;
    assign wasted_tree[GROUP_LEAVES+group] = '0;
    assign cancelled_tree[GROUP_LEAVES+group] = '0;
  end
  for (genvar node = 1; node < GROUP_LEAVES; node = node + 1) begin : group_reduction
    assign useful_tree[node] = useful_tree[2*node] + useful_tree[2*node+1];
    assign wasted_tree[node] = wasted_tree[2*node] + wasted_tree[2*node+1];
    assign cancelled_tree[node] = cancelled_tree[2*node] + cancelled_tree[2*node+1];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) useful_pending <= 0;
    else useful_pending <= useful_tree[1];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wasted_pending <= 0;
    else wasted_pending <= wasted_tree[1];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cancelled_pending <= 0;
    else cancelled_pending <= cancelled_tree[1];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      useful_bytes <= 0;
      wasted_bytes <= 0;
      cancelled_bytes <= 0;
      for (integer i = 0; i < ENTRIES; i = i + 1) begin
        live[i] <= 0;
        generation[i] <= 0;
        epoch[i] <= 0;
        allocated[i] <= 0;
        transferred[i] <= 0;
        resolved[i] <= 0;
        useful[i] <= 0;
        terminal[i] <= 0;
      end
    end else begin
      useful_bytes <= useful_bytes + 64'(useful_pending);
      wasted_bytes <= wasted_bytes + 64'(wasted_pending);
      cancelled_bytes <= cancelled_bytes + 64'(cancelled_pending);
      for (integer i = 0; i < ENTRIES; i = i + 1) begin
        if (live[i] && resolved[i] && terminal[i]) live[i] <= 0;
        if (soft_flush_valid && live[i] && epoch[i] == soft_flush_epoch) begin
          resolved[i] <= 1;
          useful[i] <= 0;
        end
      end
      if (capture_valid && 32'(capture_token.request_id) < ENTRIES) begin
        live[int'(capture_token.request_id)] <= 1;
        generation[int'(capture_token.request_id)] <= capture_token.generation;
        epoch[int'(capture_token.request_id)] <= capture_epoch;
        allocated[int'(capture_token.request_id)] <= capture_bytes;
        transferred[int'(capture_token.request_id)] <= 0;
        resolved[int'(capture_token.request_id)] <= 0;
        terminal[int'(capture_token.request_id)] <= 0;
      end
      if (burst_done && 32'(burst_token.request_id) < ENTRIES && live[int'(burst_token.request_id)] &&
          generation[int'(burst_token.request_id)] == burst_token.generation) begin
        transferred[int'(burst_token.request_id)] <= transferred[int'(burst_token.request_id)] + burst_bytes;
        if (dma_done) terminal[int'(burst_token.request_id)] <= 1;
      end
      if (route_valid && 32'(route_token.request_id) < ENTRIES && live[int'(route_token.request_id)] &&
          generation[int'(route_token.request_id)] == route_token.generation) begin
        resolved[int'(route_token.request_id)] <= 1;
        useful[int'(route_token.request_id)] <= route_truth;
      end
      if (cancelled_retired && 32'(retired_token.request_id) < ENTRIES && live[int'(retired_token.request_id)] &&
          generation[int'(retired_token.request_id)] == retired_token.generation) begin
        terminal[int'(retired_token.request_id)] <= 1;
        resolved[int'(retired_token.request_id)] <= 1;
        useful[int'(retired_token.request_id)] <= 0;
      end
    end
  end
endmodule
