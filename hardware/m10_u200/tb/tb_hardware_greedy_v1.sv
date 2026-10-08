`timescale 1ns/1ps
module tb_hardware_greedy_v1;
  logic [3:0] valid;
  logic [39:0] expert_id;
  logic [19:0] rank;
  logic [47:0] confidence;
  logic [31:0] depth;
  logic [127:0] bytes;
  logic [31:0] global_credit;
  logic [63:0] channel_credit;
  logic [11:0] channel_load;
  logic winner_valid;
  logic [1:0] winner_index;
  logic winner_channel;
  logic uniform_valid;
  logic [1:0] uniform_index;
  logic uniform_channel;

  hardware_greedy_v1 #(.CANDIDATE_COUNT(4),
    .CHANNEL_COUNT(2),
    .MAX_ALLOCATIONS(63)) dut (
    .candidate_valid(valid),
    .candidate_expert_id(expert_id),
    .candidate_rank(rank),
    .candidate_confidence_q12(confidence),
    .candidate_next_depth(depth),
    .candidate_logical_bytes(bytes),
    .global_credit_bytes(global_credit),
    .channel_credit_bytes(channel_credit),
    .channel_load_allocations(channel_load),
    .winner_valid(winner_valid),
    .winner_index(winner_index),
    .winner_channel
  );

  hardware_greedy_v1 #(
    .CANDIDATE_COUNT(4),
    .CHANNEL_COUNT(2),
    .MAX_ALLOCATIONS(63),
    .UNIFORM_TILE_BYTES(1'b1)
  ) uniform_dut (
    .candidate_valid(valid),
    .candidate_expert_id(expert_id),
    .candidate_rank(rank),
    .candidate_confidence_q12(confidence),
    .candidate_next_depth(depth),
    .candidate_logical_bytes(bytes),
    .global_credit_bytes(global_credit),
    .channel_credit_bytes(channel_credit),
    .channel_load_allocations(channel_load),
    .winner_valid(uniform_valid),
    .winner_index(uniform_index),
    .winner_channel(uniform_channel)
  );

  initial begin
    valid = 4'b1111;
    /* Field zero is LSB: expert 0 wins the initial equal-score tie by rank. */
    expert_id = {10'd3, 10'd2, 10'd1, 10'd0};
    rank = {5'd3, 5'd2, 5'd1, 5'd0};
    confidence = {12'd4095, 12'd4095, 12'd4095, 12'd4095};
    depth = '0;
    bytes = {32'd100, 32'd100, 32'd100, 32'd100};
    global_credit = 32'd100;
    channel_credit = {32'd100, 32'd100};
    channel_load = '0;
    #1;
    if (!winner_valid || winner_index != 0 || winner_channel != 0)
      $fatal(1, "HardwareGreedy-v1 deterministic tie or channel mapping failed");
    /* Expert 0 maps to channel 0 and is now disallowed by its channel credit. */
    channel_credit = {32'd100, 32'd0};
    #1;
    if (!winner_valid || winner_index != 1 || winner_channel != 1)
      $fatal(1, "HardwareGreedy-v1 per-channel credit filtering failed");
    // 确定性覆盖不同 confidence/depth/load/credit 和大 byte 值，证明共同因子
    // 约去没有改变任何 winner、tie-break 或 admission 结果。
    for (integer sample = 0; sample < 1024; sample = sample + 1) begin
      valid = sample;
      bytes = {4{32'h80000000 + sample}};
      global_credit = (sample % 7 == 0) ? 0 : 32'hffffffff;
      channel_credit[31:0] = (sample % 5 == 0) ? 0 : 32'hffffffff;
      channel_credit[63:32] = (sample % 3 == 0) ? 0 : 32'hffffffff;
      channel_load[5:0] = sample;
      channel_load[11:6] = sample * 7;
      for (integer row = 0; row < 4; row = row + 1) begin
        confidence[row*12 +: 12] = (sample * 331 + row * 997) % 4096;
        depth[row*8 +: 8] = (sample + row * 3) % 20;
      end
      #1;
      if ({uniform_valid, uniform_index, uniform_channel} !==
          {winner_valid, winner_index, winner_channel})
        $fatal(1, "uniform comparison mismatch sample=%0d", sample);
    end
    $display("PASS: HardwareGreedy-v1 selector");
    $finish;
  end
endmodule
