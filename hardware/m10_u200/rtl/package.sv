`timescale 1ns/1ps
`ifndef TILE_PREFETCH_PACKAGE_SV
`define TILE_PREFETCH_PACKAGE_SV
package tile_prefetch_package;
  typedef enum logic [1:0] { POLICY_NO_PREFETCH, POLICY_FULL_EXPERT,
    POLICY_BUDGET_PROPORTIONAL, POLICY_HARDWARE_GREEDY } policy_mode_t;
  typedef enum logic { MATRIX_GU, MATRIX_DOWN } matrix_kind_t;
  typedef enum logic { REQUEST_SPECULATIVE, REQUEST_DEMAND } request_class_t;
  typedef enum logic [3:0] { META_FREE, META_QUEUED_SPEC, META_QUEUED_DEMAND,
    META_INFLIGHT_SPEC, META_INFLIGHT_DEMAND, META_RESIDENT_SPEC,
    META_COMMITTED, META_RUNNING, META_EVICTABLE, META_DROP_ON_COMPLETE } metadata_state_t;

  /** Immutable physical tile descriptor. Mutable status belongs in metadata. */
  typedef struct packed {
    logic [15:0] epoch_id;
    logic [15:0] layer_id;
    logic [15:0] sequence_id;
    logic [9:0] expert_id;
    logic [7:0] tile_id;
    matrix_kind_t matrix_kind;
    logic [63:0] external_addr;
    logic [31:0] byte_count;
    logic [3:0] ocm_bank;
    logic [15:0] ocm_slot;
    logic [3:0] memory_channel;
    request_class_t request_class;
    logic [4:0] candidate_rank;
    logic [11:0] confidence_q12;
  } tile_descriptor_t;

  /** A FIFO stores this identity token, never a mutable descriptor copy. */
  typedef struct packed {
    logic [15:0] request_id;
    logic [7:0] generation;
  } descriptor_token_t;

  function automatic logic same_identity(
    input tile_descriptor_t a, input logic [15:0] epoch_id,
    input logic [15:0] layer_id, input logic [9:0] expert_id,
    input logic [7:0] tile_id, input matrix_kind_t matrix_kind
  );
    same_identity = a.epoch_id == epoch_id && a.layer_id == layer_id &&
      a.expert_id == expert_id && a.tile_id == tile_id && a.matrix_kind == matrix_kind;
  endfunction
endpackage
`endif
