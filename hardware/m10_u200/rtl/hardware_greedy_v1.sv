`timescale 1ns/1ps
/**
 * Frozen M10 HardwareGreedy-v1 combinational winner selector.
 *
 * Every input row represents only an Expert's next prefix-legal logical F tile.
 * The sequential candidate-table owner updates depths, credits and loads only
 * after its allocation_valid/allocation_ready handshake completes.
 */
module hardware_greedy_v1 #(
  parameter integer CANDIDATE_COUNT = 16,
  parameter integer EXPERT_W = 10,
  parameter integer RANK_W = 5,
  parameter integer DEPTH_W = 8,
  parameter integer CONF_W = 12,
  parameter integer BYTE_W = 32,
  parameter integer CHANNEL_COUNT = 4,
  parameter integer CHANNEL_W = (CHANNEL_COUNT <= 1) ? 1 : $clog2(CHANNEL_COUNT),
  parameter integer MAX_ALLOCATIONS = 256,
  parameter integer CHLOAD_W = $clog2(MAX_ALLOCATIONS + 1),
  parameter integer INDEX_W = (CANDIDATE_COUNT <= 1) ? 1 : $clog2(CANDIDATE_COUNT),
  parameter integer PRIORITY_SHIFT_PER_DEPTH = 1,
  // 仅在调用者保证所有行 logical bytes 相同时启用。credit 检查仍使用真实 bytes；
  // 分数比较中约去共同的正 byte 因子，避免不必要的大位宽乘法链。
  parameter bit UNIFORM_TILE_BYTES = 1'b0,
  parameter bit REGISTERED_PIPELINE = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic select_start,
  input logic cancel_selection,
  output logic selection_done,
  input logic [CANDIDATE_COUNT-1:0] candidate_valid,
  input logic [CANDIDATE_COUNT*EXPERT_W-1:0] candidate_expert_id,
  input logic [CANDIDATE_COUNT*RANK_W-1:0] candidate_rank,
  input logic [CANDIDATE_COUNT*CONF_W-1:0] candidate_confidence_q12,
  input logic [CANDIDATE_COUNT*DEPTH_W-1:0] candidate_next_depth,
  input logic [CANDIDATE_COUNT*BYTE_W-1:0] candidate_logical_bytes,
  input logic [BYTE_W-1:0] global_credit_bytes,
  input logic [CHANNEL_COUNT*BYTE_W-1:0] channel_credit_bytes,
  input logic [CHANNEL_COUNT*CHLOAD_W-1:0] channel_load_allocations,
  output logic winner_valid,
  output logic [INDEX_W-1:0] winner_index,
  output logic [CHANNEL_W-1:0] winner_channel
);
  localparam integer PRIO_W = CONF_W;
  localparam integer N_W = CONF_W + PRIO_W;
  localparam integer D_W = BYTE_W + CHLOAD_W + 1;
  localparam integer CMP_W = N_W + D_W;
  localparam integer LEAVES = 2 ** $clog2(CANDIDATE_COUNT);
  localparam integer PIPE_STAGES = $clog2(CANDIDATE_COUNT) + 1;
  localparam integer PAYLOAD_W = 1+INDEX_W+CHANNEL_W+EXPERT_W+RANK_W+DEPTH_W+N_W+D_W;
  if (REGISTERED_PIPELINE) begin : valid_stages
    logic [PIPE_STAGES-1:0] valid_pipeline;
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) valid_pipeline <= '0;
      else if (cancel_selection) valid_pipeline <= '0;
      else valid_pipeline <= (valid_pipeline << 1) | PIPE_STAGES'(select_start);
    end
    assign selection_done = valid_pipeline[PIPE_STAGES-1];
  end else begin : combinational_valid
    assign selection_done = select_start;
  end
  // 每个数组元素是独立树节点。split_var 让 Verilator 保留元素级 DAG，
  // 而不是将父子节点折叠成一个自依赖数组；不关闭任何 lint 告警。
  wire node_valid [1:2*LEAVES-1] /* verilator split_var */;
  wire [INDEX_W-1:0] node_index [1:2*LEAVES-1] /* verilator split_var */;
  wire [CHANNEL_W-1:0] node_channel [1:2*LEAVES-1] /* verilator split_var */;
  wire [EXPERT_W-1:0] node_expert [1:2*LEAVES-1] /* verilator split_var */;
  wire [RANK_W-1:0] node_rank [1:2*LEAVES-1] /* verilator split_var */;
  wire [DEPTH_W-1:0] node_depth [1:2*LEAVES-1] /* verilator split_var */;
  wire [N_W-1:0] node_numerator [1:2*LEAVES-1] /* verilator split_var */;
  wire [D_W-1:0] node_denominator [1:2*LEAVES-1] /* verilator split_var */;

  initial begin
    if (CANDIDATE_COUNT < 1 || CHANNEL_COUNT < 1 ||
        (CHANNEL_COUNT & (CHANNEL_COUNT - 1)) != 0)
      $error("CHANNEL_COUNT must be a power of two in HardwareGreedy-v1");
  end

  // 平衡 tournament：比较级数为 ceil(log2(N))，不再逐行串联 N-1 个乘法比较。
  // 无有效行时输出固定为零；完全相同的 key 保留左侧（较早输入行）。
  for (genvar row = 0; row < LEAVES; row = row + 1) begin : leaves
    if (row < CANDIDATE_COUNT) begin : real_row
      wire [CONF_W-1:0] confidence = candidate_confidence_q12[row*CONF_W +: CONF_W];
      wire [BYTE_W-1:0] bytes_value = candidate_logical_bytes[row*BYTE_W +: BYTE_W];
      wire [EXPERT_W-1:0] expert = candidate_expert_id[row*EXPERT_W +: EXPERT_W];
      wire [DEPTH_W-1:0] depth = candidate_next_depth[row*DEPTH_W +: DEPTH_W];
      wire [CHANNEL_W-1:0] channel = CHANNEL_COUNT == 1 ? '0 : expert[CHANNEL_W-1:0];
      wire [CHLOAD_W:0] load_factor = {1'b0,
        channel_load_allocations[channel*CHLOAD_W +: CHLOAD_W]} + 1'b1;
      wire [31:0] shift_amount = 32'(depth) * PRIORITY_SHIFT_PER_DEPTH;
      wire [PRIO_W-1:0] priority_value = shift_amount >= PRIO_W ? PRIO_W'(1) :
        {PRIO_W{1'b1}} >> shift_amount;
      wire [PAYLOAD_W-1:0] next_payload = {
        (candidate_valid[row] && confidence != 0 && bytes_value != 0 && bytes_value <= global_credit_bytes && bytes_value <= channel_credit_bytes[channel*BYTE_W +: BYTE_W]),
        (INDEX_W'(row)),
        (channel),
        (expert),
        (candidate_rank[row*RANK_W +: RANK_W]),
        (depth),
        (N_W'(confidence) * N_W'(priority_value)),
        (UNIFORM_TILE_BYTES ? D_W'(load_factor) : D_W'(bytes_value) * D_W'(load_factor))};
      wire [PAYLOAD_W-1:0] payload;
      if (REGISTERED_PIPELINE) begin : registered_score
        logic [PAYLOAD_W-1:0] payload_hold;
        always_ff @(posedge clk) begin
          if (select_start) payload_hold <= next_payload;
        end
        assign payload = payload_hold;
      end else begin : direct_score
        assign payload = next_payload;
      end
      assign {node_valid[LEAVES+row], node_index[LEAVES+row], node_channel[LEAVES+row], node_expert[LEAVES+row], node_rank[LEAVES+row], node_depth[LEAVES+row], node_numerator[LEAVES+row], node_denominator[LEAVES+row]} = payload;
    end else begin : padding
      assign node_valid[LEAVES+row] = 1'b0;
      assign node_index[LEAVES+row] = '0;
      assign node_channel[LEAVES+row] = '0;
      assign node_expert[LEAVES+row] = '0;
      assign node_rank[LEAVES+row] = '0;
      assign node_depth[LEAVES+row] = '0;
      assign node_numerator[LEAVES+row] = '0;
      assign node_denominator[LEAVES+row] = D_W'(1);
    end
  end
  for (genvar node = 1; node < LEAVES; node = node + 1) begin : tournament
    wire [CMP_W-1:0] lhs = CMP_W'(node_numerator[2*node+1]) *
      CMP_W'(node_denominator[2*node]);
    wire [CMP_W-1:0] rhs = CMP_W'(node_numerator[2*node]) *
      CMP_W'(node_denominator[2*node+1]);
    wire choose_right = node_valid[2*node+1] && (!node_valid[2*node] || lhs > rhs ||
      (lhs == rhs && (node_rank[2*node+1] < node_rank[2*node] ||
       (node_rank[2*node+1] == node_rank[2*node] &&
        (node_depth[2*node+1] < node_depth[2*node] ||
         (node_depth[2*node+1] == node_depth[2*node] &&
          node_expert[2*node+1] < node_expert[2*node]))))));
    wire [PAYLOAD_W-1:0] next_payload = choose_right ?
      {node_valid[2*node+1], node_index[2*node+1], node_channel[2*node+1], node_expert[2*node+1], node_rank[2*node+1], node_depth[2*node+1], node_numerator[2*node+1], node_denominator[2*node+1]} :
      {node_valid[2*node], node_index[2*node], node_channel[2*node], node_expert[2*node], node_rank[2*node], node_depth[2*node], node_numerator[2*node], node_denominator[2*node]};
    wire [PAYLOAD_W-1:0] payload;
    if (REGISTERED_PIPELINE) begin : registered_compare
      logic [PAYLOAD_W-1:0] payload_hold;
      always_ff @(posedge clk) begin
        payload_hold <= next_payload;
      end
      assign payload = payload_hold;
    end else begin : direct_compare
      assign payload = next_payload;
    end
    assign {node_valid[node], node_index[node], node_channel[node], node_expert[node], node_rank[node], node_depth[node], node_numerator[node], node_denominator[node]} = payload;
  end
  assign winner_valid = node_valid[1];
  assign winner_index = node_valid[1] ? node_index[1] : '0;
  assign winner_channel = node_valid[1] ? node_channel[1] : '0;
endmodule
