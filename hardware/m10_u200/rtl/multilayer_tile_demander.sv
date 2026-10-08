`timescale 1ns/1ps
/** 每层保存实际admitted pair bitmap；跨层预取不因当前层begin而丢失。 */
module multilayer_tile_demander #(
  parameter integer LAYERS=8,
  parameter integer EXPERTS=64,
  parameter integer TILES_PER_EXPERT=16,
  parameter integer GU_BYTE_RATIO=2,
  parameter integer TOP_K_MAX=8
) (
  input logic clk,
  input logic rst_n,
  input logic clear,
  input logic clear_sequence,
  input logic speculative_accept,
  input tile_prefetch_package::tile_descriptor_t speculative_descriptor,
  input tile_prefetch_package::tile_descriptor_t query_descriptor,
  output wire query_admitted,
  input logic route_accept,
  input logic [15:0] route_epoch,
  input logic [2:0] route_layer,
  input logic [9:0] route_expert,
  input logic start,
  input logic [15:0] sequence_id,
  input logic [63:0] gu_base_addr,
  input logic [63:0] down_base_addr,
  input logic [31:0] expert_stride_bytes,
  input logic [31:0] physical_tile_bytes,
  output wire logical_valid,
  input logic logical_ready,
  output tile_prefetch_package::tile_descriptor_t gu_descriptor,
  output tile_prefetch_package::tile_descriptor_t down_descriptor,
  output logic active,
  output logic done
);
  import tile_prefetch_package::*;
  logic [EXPERTS*TILES_PER_EXPERT-1:0] admitted [0:LAYERS-1];
  logic [9:0] true_ids [0:TOP_K_MAX-1];
  logic [3:0] truth_count, expert_index;
  logic [7:0] tile_index;
  logic [15:0] epoch_hold;
  logic [2:0] layer_hold;
  assign query_admitted=int'(query_descriptor.layer_id)<LAYERS && int'(query_descriptor.expert_id)<EXPERTS &&
    int'(query_descriptor.tile_id)<TILES_PER_EXPERT &&
    admitted[int'(query_descriptor.layer_id)][int'(query_descriptor.expert_id)*TILES_PER_EXPERT+
      int'(query_descriptor.tile_id)];
  wire skip=admitted[int'(layer_hold)][int'(true_ids[int'(expert_index)])*TILES_PER_EXPERT+int'(tile_index)];
  wire advance=active && (skip || logical_ready);
  wire last_tile=int'(tile_index)==TILES_PER_EXPERT-1;
  wire last_expert=expert_index+1'b1==truth_count;
  assign logical_valid=active && !skip && !clear;
  always_comb begin
    gu_descriptor='0;
    gu_descriptor.epoch_id=epoch_hold;
    gu_descriptor.layer_id={13'd0,layer_hold};
    gu_descriptor.sequence_id=sequence_id;
    gu_descriptor.expert_id=true_ids[int'(expert_index)];
    gu_descriptor.tile_id=tile_index;
    gu_descriptor.matrix_kind=MATRIX_GU;
    gu_descriptor.byte_count=physical_tile_bytes*GU_BYTE_RATIO;
    gu_descriptor.external_addr=gu_base_addr+64'(true_ids[int'(expert_index)])*expert_stride_bytes+
      64'(tile_index)*physical_tile_bytes*GU_BYTE_RATIO;
    gu_descriptor.request_class=REQUEST_DEMAND;
  end
  always_comb begin
    down_descriptor=gu_descriptor;
    down_descriptor.matrix_kind=MATRIX_DOWN;
    down_descriptor.byte_count=physical_tile_bytes;
    down_descriptor.external_addr=down_base_addr+64'(true_ids[int'(expert_index)])*expert_stride_bytes+
      64'(tile_index)*physical_tile_bytes;
  end
  for(genvar layer=0;layer<LAYERS;layer=layer+1) begin : allocation_bitmap
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) admitted[layer]<='0;
      else if(clear_sequence) admitted[layer]<='0;
      else if(speculative_accept && int'(speculative_descriptor.layer_id)==layer &&
        int'(speculative_descriptor.expert_id)<EXPERTS && int'(speculative_descriptor.tile_id)<TILES_PER_EXPERT)
        admitted[layer][int'(speculative_descriptor.expert_id)*TILES_PER_EXPERT+
          int'(speculative_descriptor.tile_id)]<=1;
    end
  end
  for(genvar index=0;index<TOP_K_MAX;index=index+1) begin : truth_entries
    always_ff @(posedge clk) begin
      if(!clear && route_accept && int'(truth_count)==index) true_ids[index]<=route_expert;
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) truth_count<=0;
    else if(clear) truth_count<=0;
    else if(route_accept && int'(truth_count)<TOP_K_MAX) truth_count<=truth_count+1'b1;
  end
  always_ff @(posedge clk) if(!clear && route_accept) epoch_hold<=route_epoch;
  always_ff @(posedge clk) if(!clear && route_accept) layer_hold<=route_layer;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) active<=0;
    else if(clear) active<=0;
    else if(start && truth_count!=0) active<=1;
    else if(advance && last_tile && last_expert) active<=0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) done<=0;
    else done<=!clear && advance && last_tile && last_expert;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) tile_index<=0;
    else if(clear || start || (advance && last_tile)) tile_index<=0;
    else if(advance) tile_index<=tile_index+1'b1;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) expert_index<=0;
    else if(clear || start) expert_index<=0;
    else if(advance && last_tile && !last_expert) expert_index<=expert_index+1'b1;
  end
endmodule
