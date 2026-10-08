`timescale 1ns/1ps
/** Single-engine true Expert tile execution scoreboard and Q15 gated combine.
 * Physical descriptor ownership is recorded by OCM slot; compute is issued only
 * after Router truth and GU completion, without waiting for DOWN payload.
 * DOWN descriptor ownership is required. Tiles execute prefix order.
 * A compute request keeps its descriptors stable until completion; GU ownership
 * retires independently, and a reused generation survives the old tile's end.
 */
module tile_execution_scheduler #(
  parameter integer SLOT_COUNT = 128,
  parameter integer TOP_K_MAX = 8,
  parameter integer TILES_PER_EXPERT = 16,
  parameter integer H = 16,
  parameter integer GENERATION_W = 8,
  parameter bit PRESERVE_CONTEXTS = 0,
  parameter bit WAIT_FULL_EXPERT = 0
) (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic [15:0] context_epoch,
  input logic [2:0] context_layer,
  input logic flush_valid,
  input logic [15:0] flush_epoch,
  input logic capture_valid,
  input tile_prefetch_package::tile_descriptor_t capture_descriptor,
  input logic [GENERATION_W-1:0] capture_generation,
  input logic dma_done,
  input tile_prefetch_package::tile_descriptor_t dma_descriptor,
  input logic route_accept,
  input logic [9:0] route_expert,
  input logic [15:0] route_weight_q15,
  input logic route_last,
  output logic compute_req_valid,
  input logic compute_req_ready,
  output tile_prefetch_package::tile_descriptor_t gu_descriptor,
  output tile_prefetch_package::tile_descriptor_t down_descriptor,
  output logic [GENERATION_W-1:0] gu_generation,
  output logic [GENERATION_W-1:0] down_generation,
  input logic compute_done,
  input logic gu_release_valid,
  input logic [H*32-1:0] partial_y,
  output logic layer_done,
  output logic [H*32-1:0] layer_y
);
  import tile_prefetch_package::*;
  localparam integer SLOT_W = SLOT_COUNT > 1 ? $clog2(SLOT_COUNT) : 1;
  localparam integer EXPERT_W = TOP_K_MAX > 1 ? $clog2(TOP_K_MAX) : 1;
  localparam integer COUNT_W = $clog2(TOP_K_MAX + 1);
  localparam integer TILE_W = $clog2(TILES_PER_EXPERT + 1);
  localparam integer SLOT_LEAVES = 2 ** $clog2(SLOT_COUNT);
  localparam integer EXPERT_LEAVES = 2 ** $clog2(TOP_K_MAX);
  logic [$bits(tile_descriptor_t)-1:0] descriptor_mem [0:SLOT_COUNT-1];
  logic [GENERATION_W-1:0] generation_mem [0:SLOT_COUNT-1];
  logic valid_mem [0:SLOT_COUNT-1];
  logic ready_mem [0:SLOT_COUNT-1];
  logic [9:0] true_ids [0:TOP_K_MAX-1];
  logic [15:0] true_weights [0:TOP_K_MAX-1];
  logic [COUNT_W-1:0] truth_count;
  logic [EXPERT_W-1:0] expert_index;
  logic [TILE_W-1:0] tile_index;
  logic truth_ready;
  logic running;
  logic issue_pending;
  logic select_for_issue;
  logic search_stale;
  tile_descriptor_t gu_hold;
  tile_descriptor_t down_hold;
  logic [GENERATION_W-1:0] gu_gen_hold;
  logic [GENERATION_W-1:0] down_gen_hold;
  tile_descriptor_t selected_gu;
  tile_descriptor_t selected_down;
  logic [GENERATION_W-1:0] selected_gu_generation;
  logic [GENERATION_W-1:0] selected_down_generation;
  logic gu_found;
  logic down_found;
  logic [H*32-1:0] expert_sums [0:TOP_K_MAX-1];
  logic [TILE_W-1:0] next_tiles [0:TOP_K_MAX-1];
  wire [TILE_W-1:0] search_tiles [0:TOP_K_MAX-1];
  logic [COUNT_W-1:0] completed_experts;
  logic [EXPERT_W-1:0] selected_expert_index;
  logic [TILE_W-1:0] selected_tile_index;
  // Stage one reduces only slot indices, not 205-bit descriptors. Balanced
  // trees preserve the old highest-slot tie break without a linear mux chain.
  logic [SLOT_W:0] gu_match [0:TOP_K_MAX-1];
  logic [SLOT_W:0] down_match [0:TOP_K_MAX-1];
  logic [TILE_W-1:0] match_tile [0:TOP_K_MAX-1];
  logic [9:0] match_expert [0:TOP_K_MAX-1];
  wire [EXPERT_W:0] expert_tree [1:2*EXPERT_LEAVES-1] /* verilator split_var */;
  logic combine_valid1;
  logic combine_valid2;
  logic [15:0] combine_weight;
  logic signed [31:0] combine_sum [0:H-1];
  logic signed [48:0] combine_product [0:H-1];

  for (genvar expert = 0; expert < TOP_K_MAX; expert = expert + 1) begin : match_pipeline
    wire [2*TILES_PER_EXPERT-1:0] full_ready;
    if (WAIT_FULL_EXPERT) begin : full_expert_gate
      for (genvar tile = 0; tile < TILES_PER_EXPERT; tile = tile + 1) begin : tile_ready
        for (genvar matrix = 0; matrix < 2; matrix = matrix + 1) begin : matrix_ready
          wire [SLOT_COUNT-1:0] slot_matches;
          for (genvar slot = 0; slot < SLOT_COUNT; slot = slot + 1) begin : slot_ready
            wire tile_descriptor_t entry = descriptor_mem[slot];
            assign slot_matches[slot] = valid_mem[slot] && ready_mem[slot] &&
              entry.expert_id == true_ids[expert] && int'(entry.tile_id) == tile &&
              int'(entry.matrix_kind) == matrix &&
              (!PRESERVE_CONTEXTS || (entry.epoch_id == context_epoch && entry.layer_id == {13'd0,context_layer}));
          end
          assign full_ready[2*tile+matrix] = |slot_matches;
        end
      end
    end else begin : streaming_gate
      assign full_ready = '1;
    end
    // While this expert runs, register its successor prefix. No request or
    // ownership changes until retirement advances next_tiles to this prefix.
    assign search_tiles[expert] = next_tiles[expert] + TILE_W'(running && int'(expert_index) == expert);
    wire [SLOT_W:0] gu_tree [1:2*SLOT_LEAVES-1] /* verilator split_var */;
    wire [SLOT_W:0] down_tree [1:2*SLOT_LEAVES-1] /* verilator split_var */;
    for (genvar slot = 0; slot < SLOT_LEAVES; slot = slot + 1) begin : leaves
      if (slot < SLOT_COUNT) begin : active
        wire tile_descriptor_t entry;
        wire eligible;
        assign entry = descriptor_mem[slot];
        assign eligible = expert < int'(truth_count) &&
          int'(search_tiles[expert]) < TILES_PER_EXPERT && valid_mem[slot] &&
          entry.expert_id == true_ids[expert] && int'(entry.tile_id) == int'(search_tiles[expert]) &&
          (!PRESERVE_CONTEXTS || (entry.epoch_id==context_epoch && entry.layer_id=={13'd0,context_layer}));
        assign gu_tree[SLOT_LEAVES+slot] = {eligible && ready_mem[slot] && entry.matrix_kind == MATRIX_GU, SLOT_W'(slot)};
        // DOWN descriptor/ownership must exist, but its DMA may finish during GU.
        assign down_tree[SLOT_LEAVES+slot] = {eligible && entry.matrix_kind == MATRIX_DOWN, SLOT_W'(slot)};
      end else begin : padding
        assign gu_tree[SLOT_LEAVES+slot] = '0;
        assign down_tree[SLOT_LEAVES+slot] = '0;
      end
    end
    for (genvar node = 1; node < SLOT_LEAVES; node = node + 1) begin : reduction
      assign gu_tree[node] = gu_tree[2*node+1][SLOT_W] ? gu_tree[2*node+1] : gu_tree[2*node];
      assign down_tree[node] = down_tree[2*node+1][SLOT_W] ? down_tree[2*node+1] : down_tree[2*node];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) gu_match[expert] <= '0;
      else if (clear) gu_match[expert] <= '0;
      else gu_match[expert] <= gu_tree[1];
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) down_match[expert] <= '0;
      else if (clear) down_match[expert] <= '0;
      else down_match[expert] <= down_tree[1];
    end
    always_ff @(posedge clk) begin
      match_tile[expert] <= search_tiles[expert];
    end
    always_ff @(posedge clk) begin
      match_expert[expert] <= true_ids[expert];
    end
    // A registered search is a snapshot, not a reservation. Revalidate it
    // after completion, slot reuse or clear before forming a stable request.
    assign expert_tree[EXPERT_LEAVES+expert] = {
      gu_match[expert][SLOT_W] && down_match[expert][SLOT_W] &&
      (next_tiles[expert] != 0 || (&full_ready)) &&
      expert < int'(truth_count) && match_tile[expert] == next_tiles[expert] &&
      match_expert[expert] == true_ids[expert] &&
      valid_mem[gu_match[expert][SLOT_W-1:0]] && ready_mem[gu_match[expert][SLOT_W-1:0]] &&
      valid_mem[down_match[expert][SLOT_W-1:0]] &&
      !(capture_valid && (int'(capture_descriptor.ocm_slot) == int'(gu_match[expert][SLOT_W-1:0]) ||
                          int'(capture_descriptor.ocm_slot) == int'(down_match[expert][SLOT_W-1:0]))),
      EXPERT_W'(expert)};
  end
  for (genvar expert = TOP_K_MAX; expert < EXPERT_LEAVES; expert = expert + 1) begin : expert_padding
    assign expert_tree[EXPERT_LEAVES+expert] = '0;
  end
  for (genvar node = 1; node < EXPERT_LEAVES; node = node + 1) begin : expert_reduction
    // Lowest Router-order eligible expert wins; both slots remain owned.
    assign expert_tree[node] = expert_tree[2*node][EXPERT_W] ? expert_tree[2*node] : expert_tree[2*node+1];
  end

  assign gu_found = expert_tree[1][EXPERT_W];
  assign down_found = gu_found;
  assign selected_expert_index = expert_tree[1][EXPERT_W-1:0];
  assign selected_tile_index = next_tiles[selected_expert_index];
  assign selected_gu = descriptor_mem[gu_match[selected_expert_index][SLOT_W-1:0]];
  assign selected_down = descriptor_mem[down_match[selected_expert_index][SLOT_W-1:0]];
  assign selected_gu_generation = generation_mem[gu_match[selected_expert_index][SLOT_W-1:0]];
  assign selected_down_generation = generation_mem[down_match[selected_expert_index][SLOT_W-1:0]];
  assign select_for_issue = truth_ready && !running && !issue_pending && !clear && !search_stale &&
      !combine_valid1 && !combine_valid2 && gu_found && down_found;
  assign compute_req_valid = issue_pending && !clear;
  assign gu_descriptor = gu_hold;
  assign down_descriptor = down_hold;
  assign gu_generation = gu_gen_hold;
  assign down_generation = down_gen_hold;

  // External mutations still need a refresh to preserve live ready-first
  // priority. Retirement alone is covered by successor-prefix lookahead;
  // a cold/short execution remains blocked by match_tile revalidation.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) search_stale <= 1'b1;
    else search_stale <= clear || capture_valid || dma_done || route_accept;
  end

  // Payload is meaningful only while valid_mem is set. Do not asynchronously
  // reset descriptor/generation storage; this permits normal RAM inference.
  always_ff @(posedge clk) begin
    if (rst_n && (!clear || PRESERVE_CONTEXTS) && capture_valid && int'(capture_descriptor.ocm_slot) < SLOT_COUNT)
      descriptor_mem[int'(capture_descriptor.ocm_slot)] <= capture_descriptor;
  end
  always_ff @(posedge clk) begin
    if (rst_n && (!clear || PRESERVE_CONTEXTS) && capture_valid && int'(capture_descriptor.ocm_slot) < SLOT_COUNT)
      generation_mem[int'(capture_descriptor.ocm_slot)] <= capture_generation;
  end

  // ready-first 扫描与 OCM/compute admission 之间设置稳定请求寄存级。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) issue_pending <= 0;
    else if (clear || (compute_req_valid && compute_req_ready)) issue_pending <= 0;
    else if (select_for_issue) issue_pending <= 1;
  end

  // Expert sum、Q15 乘法、layer 累加分三个时钟边沿，完成脉冲在最终写回后产生。
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) combine_valid1 <= 0;
    else if (clear) combine_valid1 <= 0;
    else combine_valid1 <= running && compute_done && int'(tile_index) == TILES_PER_EXPERT-1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) combine_valid2 <= 0;
    else if (clear) combine_valid2 <= 0;
    else combine_valid2 <= combine_valid1;
  end
  always_ff @(posedge clk) begin
    if (running && compute_done && int'(tile_index) == TILES_PER_EXPERT-1)
      combine_weight <= true_weights[expert_index];
  end
  for (genvar row = 0; row < H; row = row + 1) begin : combine_pipeline
    always_ff @(posedge clk) begin
      if (running && compute_done && int'(tile_index) == TILES_PER_EXPERT-1)
        combine_sum[row] <= $signed(expert_sums[expert_index][row*32 +: 32]) +
          $signed(partial_y[row*32 +: 32]);
    end
    always_ff @(posedge clk) begin
      if (combine_valid1) combine_product[row] <= combine_sum[row] * $signed({1'b0, combine_weight});
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) truth_count <= '0;
    else if (clear) truth_count <= '0;
    else if (route_accept && int'(truth_count) < TOP_K_MAX) truth_count <= truth_count + 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) completed_experts <= '0;
    else if (clear) completed_experts <= '0;
    else if (combine_valid2) completed_experts <= completed_experts + 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) expert_index <= '0;
    else if (clear) expert_index <= '0;
    else if (select_for_issue) expert_index <= selected_expert_index;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) tile_index <= '0;
    else if (clear) tile_index <= '0;
    else if (select_for_issue) tile_index <= selected_tile_index;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) truth_ready <= 1'b0;
    else if (clear || (combine_valid2 && completed_experts + 1'b1 == truth_count)) truth_ready <= 1'b0;
    else if (route_accept && int'(truth_count) < TOP_K_MAX && route_last) truth_ready <= 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) running <= 1'b0;
    else if (clear || (running && compute_done)) running <= 1'b0;
    else if (compute_req_valid && compute_req_ready) running <= 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_hold <= '0;
    else if (select_for_issue) gu_hold <= selected_gu;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_hold <= '0;
    else if (select_for_issue) down_hold <= selected_down;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_gen_hold <= '0;
    else if (select_for_issue) gu_gen_hold <= selected_gu_generation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_gen_hold <= '0;
    else if (select_for_issue) down_gen_hold <= selected_down_generation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) layer_done <= 1'b0;
    else layer_done <= !clear && combine_valid2 && completed_experts + 1'b1 == truth_count;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) layer_y <= '0;
    else if (clear) layer_y <= '0;
    else if (combine_valid2)
      for (integer row = 0; row < H; row = row + 1)
        layer_y[row*32 +: 32] <= $signed(layer_y[row*32 +: 32]) + 32'(combine_product[row] >>> 15);
  end
  for (genvar slot = 0; slot < SLOT_COUNT; slot = slot + 1) begin : slot_state
    wire tile_descriptor_t entry;
    wire retire_gu;
    wire retire_down;
    assign entry = descriptor_mem[slot];
    assign retire_gu = running && (gu_release_valid === 1'b1 || compute_done) &&
      valid_mem[slot] && int'(gu_hold.ocm_slot) == slot &&
      generation_mem[slot] == gu_gen_hold && entry.epoch_id == gu_hold.epoch_id;
    assign retire_down = running && compute_done && valid_mem[slot] &&
      int'(down_hold.ocm_slot) == slot && generation_mem[slot] == down_gen_hold &&
      entry.epoch_id == down_hold.epoch_id;
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) valid_mem[slot] <= 1'b0;
      else if ((!PRESERVE_CONTEXTS && clear) ||
        (PRESERVE_CONTEXTS && flush_valid && entry.epoch_id==flush_epoch)) valid_mem[slot] <= 1'b0;
      else if (capture_valid && int'(capture_descriptor.ocm_slot) == slot) valid_mem[slot] <= 1'b1;
      else if (retire_gu || retire_down) valid_mem[slot] <= 1'b0;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) ready_mem[slot] <= 1'b0;
      else if ((!PRESERVE_CONTEXTS && clear) ||
        (PRESERVE_CONTEXTS && flush_valid && entry.epoch_id==flush_epoch)) ready_mem[slot] <= 1'b0;
      else if (capture_valid && int'(capture_descriptor.ocm_slot) == slot) ready_mem[slot] <= 1'b0;
      else if (retire_gu || retire_down) ready_mem[slot] <= 1'b0;
      else if (dma_done && int'(dma_descriptor.ocm_slot) == slot) ready_mem[slot] <= 1'b1;
    end
  end
  for (genvar expert = 0; expert < TOP_K_MAX; expert = expert + 1) begin : expert_state
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) true_ids[expert] <= '0;
      else if (!clear && route_accept && int'(truth_count) == expert) true_ids[expert] <= route_expert;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) true_weights[expert] <= '0;
      else if (!clear && route_accept && int'(truth_count) == expert) true_weights[expert] <= route_weight_q15;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) next_tiles[expert] <= '0;
      else if (clear) next_tiles[expert] <= '0;
      else if (running && compute_done && int'(expert_index) == expert) next_tiles[expert] <= next_tiles[expert] + 1'b1;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) expert_sums[expert] <= '0;
      else if (clear || (running && compute_done && int'(expert_index) == expert && int'(tile_index) == TILES_PER_EXPERT-1)) expert_sums[expert] <= '0;
      else if (running && compute_done && int'(expert_index) == expert)
        for (integer row = 0; row < H; row = row + 1)
          expert_sums[expert][row*32 +: 32] <= $signed(expert_sums[expert][row*32 +: 32]) + $signed(partial_y[row*32 +: 32]);
    end
  end
endmodule
