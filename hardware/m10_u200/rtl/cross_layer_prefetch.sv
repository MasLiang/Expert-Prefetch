`timescale 1ns/1ps
/** 有界future context队列：完整预测立即参与共享allocator调度，不等待前层权重。
 * 每层最多一个transaction；epoch在整个sequence内必须唯一。复用既有Greedy算法。
 */
module cross_layer_prefetch #(
  parameter integer LAYERS=8,
  parameter integer MAX_CANDIDATES=16,
  parameter integer EXPERTS=64,
  parameter integer TILES=16,
  parameter integer CHANNELS=1,
  parameter integer H=16,
  parameter integer GU_BYTE_RATIO=2,
  parameter integer F_TILE=16
) (
  input logic clk,
  input logic rst_n,
  input logic clear_sequence,
  input logic sequence_busy,
  input logic abort,
  input logic [2:0] current_layer,
  input logic current_begin,
  input logic [15:0] current_epoch,
  input logic [15:0] current_sequence,
  input logic [15:0] sequence_id,
  input logic begin_valid,
  output wire begin_ready,
  input logic [2:0] target_layer,
  input logic [15:0] target_epoch,
  input logic [15:0] target_sequence,
  input logic [4:0] candidate_count,
  input logic [31:0] budget_bytes,
  input logic [CHANNELS*32-1:0] channel_credit_bytes,
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic candidate_valid,
  output wire candidate_ready,
  input logic [9:0] candidate_expert,
  input logic [4:0] candidate_rank,
  input logic [11:0] candidate_confidence,
  input logic candidate_last,
  output wire logical_valid,
  input logic logical_ready,
  output tile_prefetch_package::tile_descriptor_t gu_descriptor,
  output tile_prefetch_package::tile_descriptor_t down_descriptor,
  output wire current_context_ok,
  output wire flush_valid,
  output wire [15:0] flush_epoch,
  input logic flush_ready,
  output logic error
);
  import tile_prefetch_package::*;
  typedef struct packed {
    logic valid;
    logic launched;
    logic [15:0] epoch;
    logic [15:0] sequence_id;
    logic [4:0] count;
    logic [31:0] budget;
    logic [CHANNELS*32-1:0] channels;
    logic [63:0] gu_base;
    logic [63:0] down_base;
    logic [31:0] stride;
  } context_t;
  logic [$bits(context_t)-1:0] contexts [0:LAYERS-1];
  logic [4:0] received [0:LAYERS-1];
  logic [EXPERTS-1:0] seen [0:LAYERS-1];
  logic [9:0] experts [0:LAYERS-1][0:MAX_CANDIDATES-1];
  logic [11:0] confidences [0:LAYERS-1][0:MAX_CANDIDATES-1];
  logic receiving;
  logic [2:0] rx_layer;
  typedef enum logic [1:0] {IDLE, LAUNCH, FEED, ALLOCATE} state_t;
  state_t state;
  logic [2:0] selected;
  logic [4:0] feed_index;
  logic sweeping;
  logic [3:0] sweep_index;
  wire context_t rx_context=contexts[int'(rx_layer)];
  wire context_t selected_context=contexts[int'(selected)];
  wire context_t target_context=contexts[int'(target_layer)];
  wire context_t current_context=contexts[int'(current_layer)];
  wire context_t sweep_context=int'(sweep_index)<LAYERS ? contexts[int'(sweep_index)] : '0;
  wire [LAYERS-1:0] epoch_collision;
  wire [LAYERS-1:0] current_collision;
  wire [LAYERS-1:0] eligible;
  wire [3:0] choice [0:LAYERS] /* verilator split_var */;
  wire scheduler_valid, scheduler_done, scheduler_active;
  wire scheduler_stop=abort || sweeping || !sequence_busy || (state!=IDLE && selected<=current_layer);
  wire rx_legal=int'(candidate_expert)<EXPERTS && candidate_rank==received[int'(rx_layer)] &&
    received[int'(rx_layer)]<rx_context.count && !seen[int'(rx_layer)][int'(candidate_expert)] &&
    candidate_last==(received[int'(rx_layer)]+1'b1==rx_context.count);
  wire begin_fire=begin_valid && begin_ready;
  wire candidate_fire=candidate_valid && candidate_ready;
  assign choice[LAYERS]=0;
  for(genvar layer=0;layer<LAYERS;layer=layer+1) begin : layer_context
    wire context_t ctx=contexts[layer];
    context_t ctx_next;
    assign epoch_collision[layer]=ctx.valid && ctx.epoch==target_epoch;
    assign current_collision[layer]=ctx.valid && layer!=int'(current_layer) &&
      ctx.epoch==current_epoch;
    if(layer>0) assign eligible[layer]=ctx.valid && !ctx.launched &&
      received[layer]==ctx.count && layer>int'(current_layer);
    else assign eligible[layer]=0;
    assign choice[layer]=eligible[layer] ? {1'b1,3'(layer)} : choice[layer+1];
    always_comb begin
      ctx_next=ctx;
      if(begin_fire && int'(target_layer)==layer) begin
        ctx_next.valid=1;
        ctx_next.launched=0;
        ctx_next.epoch=target_epoch;
        ctx_next.sequence_id=target_sequence;
        ctx_next.count=candidate_count;
        ctx_next.budget=budget_bytes;
        ctx_next.channels=channel_credit_bytes;
        ctx_next.gu_base=gu_base_addr;
        ctx_next.down_base=down_base_addr;
        ctx_next.stride=expert_stride_bytes;
      end else if(state==IDLE && choice[0][3] && !abort && !sweeping &&
                  int'(choice[0][2:0])==layer) ctx_next.launched=1;
      else if(current_begin && int'(current_layer)==layer) begin
        ctx_next.valid=1;
        ctx_next.epoch=current_epoch;
        ctx_next.sequence_id=current_sequence;
      end
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) contexts[layer]<='0;
      else if(clear_sequence) contexts[layer]<='0;
      else contexts[layer]<=ctx_next;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) received[layer]<=0;
      else if(clear_sequence || (begin_fire && int'(target_layer)==layer)) received[layer]<=0;
      else if(candidate_fire && int'(rx_layer)==layer) received[layer]<=received[layer]+1'b1;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) seen[layer]<='0;
      else if(clear_sequence || (begin_fire && int'(target_layer)==layer)) seen[layer]<='0;
      else if(candidate_fire && int'(rx_layer)==layer) seen[layer][int'(candidate_expert)]<=1;
    end
    for(genvar rank=0;rank<MAX_CANDIDATES;rank=rank+1) begin : candidate_metadata
      always_ff @(posedge clk) begin
        if(candidate_fire && int'(rx_layer)==layer && int'(candidate_rank)==rank)
          experts[layer][rank]<=candidate_expert;
      end
      always_ff @(posedge clk) begin
        if(candidate_fire && int'(rx_layer)==layer && int'(candidate_rank)==rank)
          confidences[layer][rank]<=candidate_confidence;
      end
    end
  end
  assign begin_ready=sequence_busy && !abort && !sweeping && !receiving && !error &&
    int'(target_layer)<LAYERS && target_layer>current_layer && target_sequence==sequence_id &&
    !(|epoch_collision) &&
    !target_context.valid && int'(candidate_count)<=MAX_CANDIDATES &&
    gu_base_addr[5:0]==0 && down_base_addr[5:0]==0 && expert_stride_bytes[5:0]==0 &&
    64'(expert_stride_bytes)>=64'(H*F_TILE*2*TILES);
  assign candidate_ready=receiving && !abort && !sweeping && !error && rx_legal;
  assign current_context_ok=!(|current_collision) && (!current_context.valid ||
    (current_context.epoch==current_epoch && current_context.sequence_id==current_sequence &&
      received[int'(current_layer)]==current_context.count));
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) error<=0;
    else if(clear_sequence || abort) error<=0;
    else if(receiving && candidate_valid && !rx_legal) error<=1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) receiving<=0;
    else if(clear_sequence || abort || sweeping || (candidate_fire && candidate_last)) receiving<=0;
    else if(begin_fire && candidate_count!=0) receiving<=1;
  end
  always_ff @(posedge clk) if(begin_fire) rx_layer<=target_layer;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) state<=IDLE;
    else if(clear_sequence || scheduler_stop) state<=IDLE;
    else case(state)
      IDLE: if(choice[0][3]) state<=LAUNCH;
      LAUNCH: if(selected_context.count==0) state<=IDLE; else state<=FEED;
      FEED: if(feed_index+1'b1==selected_context.count) state<=ALLOCATE;
      ALLOCATE: if(scheduler_done) state<=IDLE;
      default: state<=IDLE;
    endcase
  end
  always_ff @(posedge clk) begin
    if(state==IDLE && choice[0][3]) selected<=choice[0][2:0];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) feed_index<=0;
    else if(state==LAUNCH) feed_index<=0;
    else if(state==FEED) feed_index<=feed_index+1'b1;
  end
  assign logical_valid=scheduler_valid && state==ALLOCATE && !scheduler_stop;
  candidate_prefetch_scheduler #(
    .MAX_CANDIDATES(MAX_CANDIDATES),
    .TILES_PER_EXPERT(TILES),
    .CHANNEL_COUNT(CHANNELS),
    .GU_BYTE_RATIO(GU_BYTE_RATIO)
  ) allocator (
    .clk(clk),
    .rst_n(rst_n),
    .transaction_begin(state==LAUNCH && !scheduler_stop),
    .stop_allocation(scheduler_stop),
    .transaction_epoch(selected_context.epoch),
    .transaction_sequence(selected_context.sequence_id),
    .transaction_layer(selected),
    .speculative_budget_bytes(selected_context.budget),
    .channel_credit_bytes(selected_context.channels),
    .candidate_accept(state==FEED && !scheduler_stop),
    .candidate_epoch(selected_context.epoch),
    .candidate_layer(selected),
    .candidate_expert(experts[int'(selected)][int'(feed_index)]),
    .candidate_rank(feed_index),
    .candidate_confidence(confidences[int'(selected)][int'(feed_index)]),
    .candidate_last(feed_index+1'b1==selected_context.count),
    .gu_base_addr(selected_context.gu_base),
    .down_base_addr(selected_context.down_base),
    .expert_stride_bytes(selected_context.stride),
    .physical_tile_bytes(32'(H*F_TILE)),
    .logical_valid(scheduler_valid),
    .logical_ready(logical_ready && logical_valid),
    .logical_gu_descriptor(gu_descriptor),
    .logical_down_descriptor(down_descriptor),
    .allocation_active(scheduler_active),
    .allocation_done(scheduler_done),
    .remaining_credit_bytes()
  );
  assign flush_valid=sweeping && int'(sweep_index)<LAYERS && sweep_context.valid;
  assign flush_epoch=sweep_context.epoch;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) sweeping<=0;
    else if(clear_sequence) sweeping<=0;
    else if(abort) sweeping<=1;
    else if(int'(sweep_index)==LAYERS) sweeping<=0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) sweep_index<=0;
    else if(clear_sequence || (!sweeping && abort)) sweep_index<=0;
    else if(sweeping && int'(sweep_index)<LAYERS && (!sweep_context.valid || flush_ready))
      sweep_index<=sweep_index+1'b1;
  end
endmodule
