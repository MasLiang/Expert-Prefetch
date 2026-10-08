`timescale 1ns/1ps
/**
 * Compute backend 的控制面 ABI；真实模式使用 backend stage-done，
 * timer 模式仅用于旧模块回归。
 *
 * 一个请求代表同一 logical F-tile 的 GU/DOWN slot pair。控制器只在对应 OCM
 * metadata 已 committed/resident 时启动各阶段。GU 完成后独立释放 GU slot，
 * DOWN 完成后只释放 DOWN slot；partial-output 使用 backend 捕获的值。
 */
module tile_compute_controller #(
  parameter integer GENERATION_W = 8,
  parameter bit EXTERNAL_COMPUTE = 1'b0
) (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic compute_req_valid,
  output logic compute_req_ready,
  input tile_prefetch_package::tile_descriptor_t compute_gu_descriptor,
  input logic [GENERATION_W-1:0] compute_gu_slot_generation,
  input tile_prefetch_package::tile_descriptor_t compute_down_descriptor,
  input logic [GENERATION_W-1:0] compute_down_slot_generation,
  input logic gu_resident,
  input logic down_resident,
  input logic [31:0] gu_cycles,
  input logic [31:0] nonlinear_cycles,
  input logic [31:0] down_cycles,
  input logic backend_gu_done,
  input logic backend_nonlinear_done,
  input logic backend_down_done,
  output logic busy,
  output logic done,
  output logic waiting_gu,
  output logic waiting_down,
  output logic gu_compute_start,
  output logic down_compute_start,
  output logic consumer_gu_release_valid,
  output logic consumer_release_valid,
  output logic [15:0] consumer_release_gu_slot,
  output logic [15:0] consumer_release_down_slot,
  output logic [15:0] consumer_release_epoch,
  output logic [GENERATION_W-1:0] consumer_release_gu_generation,
  output logic [GENERATION_W-1:0] consumer_release_down_generation
);
  import tile_prefetch_package::*;
  typedef enum logic [2:0] {IDLE, WAIT_GU, GU, NONLINEAR, WAIT_DOWN, DOWN} phase_t;
  phase_t phase;
  logic [31:0] remaining;
  tile_descriptor_t gu_hold, down_hold;
  logic [GENERATION_W-1:0] gu_generation_hold, down_generation_hold;
  logic request_fire;
  logic gu_finished;
  logic nonlinear_finished;
  logic down_finished;
  logic launch_gu;
  logic launch_down;
  logic retire;

  assign compute_req_ready = !clear && phase == IDLE;
  assign request_fire = compute_req_valid && compute_req_ready;
  assign gu_finished = EXTERNAL_COMPUTE ? backend_gu_done : remaining <= 1;
  assign nonlinear_finished = EXTERNAL_COMPUTE ? backend_nonlinear_done : remaining <= 1;
  assign down_finished = EXTERNAL_COMPUTE ? backend_down_done : remaining <= 1;
  // Starts remain registered: descriptor/generation capture is visible before
  // the backend consumes its pulse. A ready stage need not visit a wait state.
  assign launch_gu = gu_resident && ((phase == IDLE && request_fire) || phase == WAIT_GU);
  assign launch_down = down_resident && ((phase == NONLINEAR && nonlinear_finished) || phase == WAIT_DOWN);
  assign retire = phase == DOWN && down_finished;
  assign consumer_release_gu_slot = gu_hold.ocm_slot;
  assign consumer_release_down_slot = down_hold.ocm_slot;
  assign consumer_release_epoch = gu_hold.epoch_id;
  assign consumer_release_gu_generation = gu_generation_hold;
  assign consumer_release_down_generation = down_generation_hold;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) phase <= IDLE;
    else if (clear) phase <= IDLE;
    else begin
      case (phase)
        IDLE: if (request_fire) begin
          if (gu_resident) phase <= GU;
          else phase <= WAIT_GU;
        end
        WAIT_GU: if (gu_resident) phase <= GU;
        GU: if (gu_finished) phase <= NONLINEAR;
        NONLINEAR: if (nonlinear_finished) begin
          if (down_resident) phase <= DOWN;
          else phase <= WAIT_DOWN;
        end
        WAIT_DOWN: if (down_resident) phase <= DOWN;
        DOWN: if (down_finished) phase <= IDLE;
        default: phase <= IDLE;
      endcase
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) remaining <= '0;
    else if (clear) remaining <= '0;
    else if (launch_gu) remaining <= gu_cycles;
    else if (phase == GU && gu_finished) remaining <= nonlinear_cycles;
    else if (launch_down) remaining <= down_cycles;
    else if ((phase == GU && !gu_finished) || (phase == NONLINEAR && !nonlinear_finished) ||
             (phase == DOWN && !down_finished)) remaining <= remaining - 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_hold <= '0;
    else if (clear) gu_hold <= '0;
    else if (request_fire) gu_hold <= compute_gu_descriptor;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_hold <= '0;
    else if (clear) down_hold <= '0;
    else if (request_fire) down_hold <= compute_down_descriptor;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_generation_hold <= '0;
    else if (clear) gu_generation_hold <= '0;
    else if (request_fire) gu_generation_hold <= compute_gu_slot_generation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_generation_hold <= '0;
    else if (clear) down_generation_hold <= '0;
    else if (request_fire) down_generation_hold <= compute_down_slot_generation;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) busy <= 1'b0;
    else if (clear || retire) busy <= 1'b0;
    else if (request_fire) busy <= 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) done <= 1'b0;
    else if (clear) done <= 1'b0;
    else done <= retire;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) waiting_gu <= 1'b0;
    else if (clear) waiting_gu <= 1'b0;
    else waiting_gu <= phase == WAIT_GU && !gu_resident;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) waiting_down <= 1'b0;
    else if (clear) waiting_down <= 1'b0;
    else waiting_down <= phase == WAIT_DOWN && !down_resident;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gu_compute_start <= 1'b0;
    else if (clear) gu_compute_start <= 1'b0;
    else gu_compute_start <= launch_gu;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) down_compute_start <= 1'b0;
    else if (clear) down_compute_start <= 1'b0;
    else down_compute_start <= launch_down;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) consumer_gu_release_valid <= 1'b0;
    else if (clear) consumer_gu_release_valid <= 1'b0;
    else consumer_gu_release_valid <= phase == GU && gu_finished;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) consumer_release_valid <= 1'b0;
    else if (clear) consumer_release_valid <= 1'b0;
    else consumer_release_valid <= retire;
  end
endmodule
