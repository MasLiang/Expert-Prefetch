`timescale 1ns/1ps
/** 平衡树对照独立串行参考，覆盖 padding、单 channel 和完整位宽乘法。 */
module tb_greedy_tree_equivalence #(
  parameter integer N = 5,
  parameter integer CHANNELS = 4,
  parameter bit PIPELINED = 0,
  parameter integer IW = N <= 1 ? 1 : $clog2(N),
  parameter integer CW = CHANNELS <= 1 ? 1 : $clog2(CHANNELS)
);
  logic [N-1:0] valid;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n = 0;
  logic select_start = 0;
  logic selection_done;
  logic [N*10-1:0] experts;
  logic [N*5-1:0] ranks;
  logic [N*12-1:0] confidence;
  logic [N*8-1:0] depth;
  logic [N*32-1:0] bytes;
  logic [31:0] global_credit;
  logic [CHANNELS*32-1:0] credits;
  logic [CHANNELS*6-1:0] loads;
  logic winner_valid;
  logic [IW-1:0] winner_index;
  logic [CW-1:0] winner_channel;
  logic expected_valid;
  integer expected_index;
  integer expected_channel;
  integer best_rank;
  integer best_depth;
  integer best_expert;
  integer channel;
  logic [23:0] numerator;
  logic [38:0] denominator;
  logic [23:0] best_numerator;
  logic [38:0] best_denominator;
  logic [62:0] lhs;
  logic [62:0] rhs;
  logic [11:0] priority_value;

  hardware_greedy_v1 #(
    .CANDIDATE_COUNT(N),
    .CHANNEL_COUNT(CHANNELS),
    .MAX_ALLOCATIONS(63),
    .REGISTERED_PIPELINE(PIPELINED)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .select_start(select_start),
    .cancel_selection(1'b0),
    .selection_done(selection_done),
    .candidate_valid(valid),
    .candidate_expert_id(experts),
    .candidate_rank(ranks),
    .candidate_confidence_q12(confidence),
    .candidate_next_depth(depth),
    .candidate_logical_bytes(bytes),
    .global_credit_bytes(global_credit),
    .channel_credit_bytes(credits),
    .channel_load_allocations(loads),
    .winner_valid(winner_valid),
    .winner_index(winner_index),
    .winner_channel(winner_channel)
  );

  initial begin
    repeat (2) @(negedge clk);
    rst_n = 1;
    for (integer sample = 0; sample < 1024; sample = sample + 1) begin
      global_credit = sample % 11 == 0 ? 0 : 32'hffffffff;
      for (integer c = 0; c < CHANNELS; c = c + 1) begin
        credits[c*32 +: 32] = (sample + c) % 7 == 0 ? 0 : 32'hffffffff;
        loads[c*6 +: 6] = 6'(sample * 17 + c * 3);
      end
      expected_valid = 0;
      expected_index = 0;
      expected_channel = 0;
      best_rank = 0;
      best_depth = 0;
      best_expert = 0;
      best_numerator = 0;
      best_denominator = 1;
      for (integer row = 0; row < N; row = row + 1) begin
        valid[row] = (sample + row) % 5 != 0;
        experts[row*10 +: 10] = 10'(row * 7);
        ranks[row*5 +: 5] = 5'((N-row) % 4);
        confidence[row*12 +: 12] = sample % 16 == 0 ? 12'd4095 :
          12'(sample * 331 + row * 997);
        depth[row*8 +: 8] = sample % 16 == 0 ? 0 : 8'((sample + row * 3) % 20);
        bytes[row*32 +: 32] = sample % 16 == 0 ? 32'd100 :
          32'hffffffff - 32'(sample * 11 + row * 311);
        channel = CHANNELS == 1 ? 0 : int'(experts[row*10 +: 10]) % CHANNELS;
        priority_value = depth[row*8 +: 8] >= 12 ? 12'd1 :
          12'hfff >> depth[row*8 +: 8];
        numerator = 24'(confidence[row*12 +: 12]) * 24'(priority_value);
        denominator = 39'(bytes[row*32 +: 32]) * (39'(loads[channel*6 +: 6]) + 1);
        lhs = 63'(numerator) * 63'(best_denominator);
        rhs = 63'(best_numerator) * 63'(denominator);
        if (valid[row] && confidence[row*12 +: 12] != 0 && bytes[row*32 +: 32] != 0 &&
            bytes[row*32 +: 32] <= global_credit &&
            bytes[row*32 +: 32] <= credits[channel*32 +: 32] &&
            (!expected_valid || lhs > rhs || (lhs == rhs &&
             (int'(ranks[row*5 +: 5]) < best_rank ||
              (int'(ranks[row*5 +: 5]) == best_rank &&
               (int'(depth[row*8 +: 8]) < best_depth ||
                (int'(depth[row*8 +: 8]) == best_depth &&
                 int'(experts[row*10 +: 10]) < best_expert))))))) begin
          expected_valid = 1;
          expected_index = row;
          expected_channel = channel;
          best_numerator = numerator;
          best_denominator = denominator;
          best_rank = int'(ranks[row*5 +: 5]);
          best_depth = int'(depth[row*8 +: 8]);
          best_expert = int'(experts[row*10 +: 10]);
        end
      end
      if (PIPELINED) begin
        @(negedge clk);
        select_start = 1;
        @(negedge clk);
        select_start = 0;
        wait (selection_done);
      end
      #1;
      if (winner_valid !== expected_valid || winner_index !== IW'(expected_index) ||
          winner_channel !== CW'(expected_channel))
        $fatal(1, "tree/reference mismatch N=%0d channels=%0d sample=%0d", N, CHANNELS, sample);
    end
    $display("PASS: tree/reference equivalence N=%0d channels=%0d pipeline=%0d (1024 cases)", N, CHANNELS, PIPELINED);
    $finish;
  end
endmodule
