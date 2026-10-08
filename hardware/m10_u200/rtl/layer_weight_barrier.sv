`timescale 1ns/1ps
/** 仅监测逐层真实GU/DOWN DMA完成，不作为预取门控；释放OCM不撤销完成历史。 */
module layer_weight_barrier #(
  parameter integer LAYERS=8,
  parameter integer EXPERTS=64,
  parameter integer TILES=16
) (
  input logic clk,
  input logic rst_n,
  input logic clear_sequence,
  input logic route_accept,
  input logic [2:0] route_layer,
  input logic [9:0] route_expert,
  input logic route_last,
  input logic dma_done,
  input tile_prefetch_package::tile_descriptor_t dma_descriptor,
  output wire [LAYERS-1:0] weights_done
);
  import tile_prefetch_package::*;
  for(genvar layer=0;layer<LAYERS;layer=layer+1) begin : layers
    logic resolved;
    logic [EXPERTS-1:0] truth;
    logic [EXPERTS*TILES*2-1:0] completed;
    wire [EXPERTS-1:0] expert_complete;
    for(genvar expert=0;expert<EXPERTS;expert=expert+1) begin : experts
      assign expert_complete[expert]=!truth[expert] || &completed[expert*TILES*2 +: TILES*2];
    end
    assign weights_done[layer]=resolved && (&expert_complete);
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) resolved<=0;
      else if(clear_sequence) resolved<=0;
      else if(route_accept && int'(route_layer)==layer && route_last) resolved<=1;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) truth<='0;
      else if(clear_sequence) truth<='0;
      else if(route_accept && int'(route_layer)==layer && int'(route_expert)<EXPERTS)
        truth[int'(route_expert)]<=1;
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) completed<='0;
      else if(clear_sequence) completed<='0;
      else if(dma_done && int'(dma_descriptor.layer_id)==layer &&
              int'(dma_descriptor.expert_id)<EXPERTS && int'(dma_descriptor.tile_id)<TILES)
        completed[int'(dma_descriptor.expert_id)*TILES*2+int'(dma_descriptor.tile_id)*2+
          int'(dma_descriptor.matrix_kind)]<=1;
    end
  end
endmodule
