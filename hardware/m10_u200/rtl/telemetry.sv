`timescale 1ns/1ps
/** Wraparound 64-bit hardware counters; reset establishes a fresh sequence. */
module telemetry (
  input logic clk, input logic rst_n, input logic sequence_active,
  input logic layer_done, input logic speculative_alloc, input logic speculative_issue,
  input logic demand_issue, input logic speculative_useful, input logic speculative_wasted,
  input logic speculative_cancel, input logic inflight_wasted, input logic demand_stall,
  input logic compute_wait_gu, input logic compute_wait_down, input logic scheduler_active,
  input logic allocation, input logic commit, input logic cancel,
  input logic [31:0] bytes, input logic [31:0] ocm_occupancy,
  input logic [31:0] issue_bytes,
  output logic [63:0] total_cycles, output logic [63:0] completed_layers,
  output logic [63:0] speculative_allocated_bytes, speculative_issued_bytes,
  output logic [63:0] demand_issued_bytes, useful_speculative_bytes,
  output logic [63:0] wasted_speculative_bytes, cancelled_before_issue_bytes,
  output logic [63:0] inflight_wasted_bytes, demand_stall_cycles,
  output logic [63:0] compute_wait_gu_cycles, compute_wait_down_cycles,
  output logic [63:0] ocm_peak_occupancy, scheduler_cycles, allocation_count,
  output logic [63:0] commit_count, cancel_count
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      total_cycles <= 0;
      completed_layers <= 0;
      speculative_allocated_bytes <= 0;
      speculative_issued_bytes <= 0;
      demand_issued_bytes <= 0;
      useful_speculative_bytes <= 0;
      wasted_speculative_bytes <= 0;
      cancelled_before_issue_bytes <= 0;
      inflight_wasted_bytes <= 0;
      demand_stall_cycles <= 0;
      compute_wait_gu_cycles <= 0;
      compute_wait_down_cycles <= 0;
      ocm_peak_occupancy <= 0;
      scheduler_cycles <= 0;
      allocation_count <= 0;
      commit_count <= 0;
      cancel_count <= 0;
    end else begin
      if (sequence_active) total_cycles <= total_cycles + 1;
      if (layer_done) completed_layers <= completed_layers + 1;
      if (speculative_alloc) speculative_allocated_bytes <= speculative_allocated_bytes + 64'(bytes);
      if (speculative_issue) speculative_issued_bytes <= speculative_issued_bytes + 64'(issue_bytes);
      if (demand_issue) demand_issued_bytes <= demand_issued_bytes + 64'(issue_bytes);
      if (speculative_useful) useful_speculative_bytes <= useful_speculative_bytes + 64'(bytes);
      if (speculative_wasted) wasted_speculative_bytes <= wasted_speculative_bytes + 64'(bytes);
      if (speculative_cancel) cancelled_before_issue_bytes <= cancelled_before_issue_bytes + 64'(bytes);
      if (inflight_wasted) inflight_wasted_bytes <= inflight_wasted_bytes + 64'(bytes);
      if (demand_stall) demand_stall_cycles <= demand_stall_cycles + 1;
      if (compute_wait_gu) compute_wait_gu_cycles <= compute_wait_gu_cycles + 1;
      if (compute_wait_down) compute_wait_down_cycles <= compute_wait_down_cycles + 1;
      if (scheduler_active) scheduler_cycles <= scheduler_cycles + 1;
      if (allocation) allocation_count <= allocation_count + 1;
      if (commit) commit_count <= commit_count + 1;
      if (cancel) cancel_count <= cancel_count + 1;
      if (64'(ocm_occupancy) > ocm_peak_occupancy) ocm_peak_occupancy <= 64'(ocm_occupancy);
    end
  end
endmodule
