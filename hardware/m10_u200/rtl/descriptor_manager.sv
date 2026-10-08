`timescale 1ns/1ps
/**
 * 逻辑 F-tile 的原子 admission manager。
 *
 * 一个 logical allocation 固定展开为 GU 后 DOWN 两个 physical descriptor。
 * 在两个 slot、两个 DescriptorTable entry 和两个 FIFO credit 同时可保留前，
 * logical_ready 必须保持低，因此 scheduler 不会因为半个 logical tile 而推进。
 * 已 live 的 speculative pair 遇到 demand 时走 promotion，不复制 descriptor。
 */
module descriptor_manager (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic logical_valid,
  input tile_prefetch_package::tile_descriptor_t logical_gu_descriptor,
  input tile_prefetch_package::tile_descriptor_t logical_down_descriptor,
  input logic existing_pair_live,
  input logic existing_pair_is_speculative,
  output logic logical_ready,
  input logic gu_slot_reserve_ready,
  input logic down_slot_reserve_ready,
  input logic descriptor_pair_reserve_ready,
  input logic queue_pair_reserve_ready,
  input logic promotion_ready,
  output logic reserve_pair_valid,
  output logic promotion_valid,
  output logic gu_valid,
  output tile_prefetch_package::tile_descriptor_t gu_descriptor,
  input logic gu_ready,
  output logic down_valid,
  output tile_prefetch_package::tile_descriptor_t down_descriptor,
  input logic down_ready
);
  import tile_prefetch_package::*;
  typedef enum logic [1:0] {WAIT_LOGICAL, EMIT_GU, EMIT_DOWN} manager_state_t;
  manager_state_t state;
  tile_descriptor_t gu_hold, down_hold;
  logic is_demand;

  always_comb begin
    is_demand = logical_gu_descriptor.request_class == REQUEST_DEMAND;
    reserve_pair_valid = 1'b0;
    promotion_valid = 1'b0;
    gu_valid = state == EMIT_GU;
    down_valid = state == EMIT_DOWN;
    gu_descriptor = gu_hold;
    down_descriptor = down_hold;
    logical_ready = 1'b0;
    if (state == WAIT_LOGICAL) begin
      if (existing_pair_live) begin
        /* A speculative resident/queued pair becomes demand-priority without
           allocating a duplicate. A speculative duplicate is simply ignored. */
        if (is_demand && existing_pair_is_speculative) begin
          logical_ready = promotion_ready;
          promotion_valid = logical_valid && promotion_ready;
        end else begin
          logical_ready = 1'b1;
        end
      end else begin
        logical_ready = gu_slot_reserve_ready && down_slot_reserve_ready &&
          descriptor_pair_reserve_ready && queue_pair_reserve_ready;
        reserve_pair_valid = logical_valid && logical_ready;
      end
    end
    // No new ownership may be created on the abort edge.
    if (clear) begin
      reserve_pair_valid = 1'b0;
      promotion_valid = 1'b0;
      gu_valid = 1'b0;
      down_valid = 1'b0;
      logical_ready = 1'b0;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= WAIT_LOGICAL;
      gu_hold <= '0;
      down_hold <= '0;
    end else if (clear) begin
      state <= WAIT_LOGICAL;
      gu_hold <= '0;
      down_hold <= '0;
    end else begin
      case (state)
        WAIT_LOGICAL: begin
          if (logical_valid && logical_ready && !existing_pair_live) begin
            gu_hold <= logical_gu_descriptor;
            down_hold <= logical_down_descriptor;
            state <= EMIT_GU;
          end
        end
        EMIT_GU: if (gu_valid && gu_ready) state <= EMIT_DOWN;
        EMIT_DOWN: if (down_valid && down_ready) state <= WAIT_LOGICAL;
        default: state <= WAIT_LOGICAL;
      endcase
    end
  end
endmodule
