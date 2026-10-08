IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
BUILD := build
RTL := rtl/package.sv rtl/hardware_greedy_v1.sv rtl/control_transaction_parser.sv rtl/candidate_prefetch_scheduler.sv rtl/route_token_resolver.sv rtl/descriptor_table.sv rtl/descriptor_manager.sv rtl/partial_output_accumulator.sv rtl/ocm_payload_ram.sv rtl/sequence_controller.sv rtl/ocm_slot_ownership.sv rtl/burst_dma_channel.sv rtl/axi_read_adapter.sv rtl/tile_compute_controller.sv rtl/telemetry.sv rtl/tile_prefetch_dataflow_top.sv rtl/tile_prefetch_axi_top.sv rtl/tile_prefetch_predictive_axi_top.sv

.PHONY: switch-test switch-four-cases four-cases ddr4-validation rtl-test lint golden alignment alignment-default alignment-main alignment-control alignment-interface boundary-validation descriptor-boundary ingress-boundary control-sensitivity gu-early-test handoff-test scheduler-test stage-release-test mac-pipeline-test multilayer-test synth-ooc synth-default clean

switch-four-cases:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/compare_four_cases.py --config configs/hardware/u200_switch_four_case_validation.yaml --output results/m10_switch_four_cases $(ALIGNMENT_ARGS)

four-cases:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/compare_four_cases.py $(ALIGNMENT_ARGS)

ddr4-validation:
	python scripts/run_ddr4_validation.py $(ALIGNMENT_ARGS)

multilayer-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_multilayer.py $(ALIGNMENT_ARGS)

mac-pipeline-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_gu_early_start.py --baseline results/m10_gu_stage_release --output results/m10_mac_pipeline_current --require-stage-release --require-pipelined-service --require-cycle-reduction $(ALIGNMENT_ARGS)

stage-release-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_gu_early_start.py --baseline results/m10_scheduler_lookahead --output results/m10_gu_stage_release_current --require-stage-release $(ALIGNMENT_ARGS)

scheduler-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_gu_early_start.py --baseline results/m10_compute_handoff --output results/m10_scheduler_lookahead_current --require-cycle-reduction $(ALIGNMENT_ARGS)

handoff-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_gu_early_start.py --baseline results/m10_gu_only_start --output results/m10_compute_handoff_current --require-cycle-reduction $(ALIGNMENT_ARGS)

gu-early-test:
	cd ../.. && PYTHONPATH=src python hardware/m10_u200/scripts/validate_gu_early_start.py $(ALIGNMENT_ARGS)

RTL += rtl/int8_swiglu_tile_backend.sv rtl/true_tile_demander.sv rtl/tile_execution_scheduler.sv
RTL += rtl/speculative_byte_finalizer.sv
RTL += rtl/striped_ddr_read_adapter.sv rtl/router_weight_store.sv rtl/router_topk_softmax.sv
RTL += rtl/u200_moe_top.sv
RTL += rtl/moe_layer_chain.sv
RTL += rtl/layer_weight_barrier.sv rtl/multilayer_tile_demander.sv rtl/cross_layer_prefetch.sv

switch-test:
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -s tb_switch_relu_backend -o $(BUILD)/switch_backend $(RTL) tb/tb_switch_relu_backend.sv
	$(VVP) $(BUILD)/switch_backend
	@for geometry in "48 48" "16 32"; do set -- $$geometry; $(IVERILOG) -g2012 -s tb_mac_chunk_pipeline -Ptb_mac_chunk_pipeline.H=$$1 -Ptb_mac_chunk_pipeline.F_TILE=$$2 -Ptb_mac_chunk_pipeline.SWITCH_RELU=1 -o $(BUILD)/switch_mac $(RTL) tb/tb_mac_chunk_pipeline.sv && $(VVP) $(BUILD)/switch_mac || exit 1; done
	@for args in "" "-Ptb_u200_moe.BUDGET=0" "-Ptb_u200_moe.PREDICTED_EXPERT=5" "-Ptb_u200_moe.FLUSH_AT=2" "-Ptb_u200_moe.ERROR_BANK=2" "-Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=1" "-Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=2 -Ptb_u200_moe.QUANT_SHIFT=1" "-Ptb_u200_moe.LAYERS=8 -Ptb_u200_moe.FUTURE_MODE=1 -Ptb_u200_moe.REPEATS=2"; do $(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.SWITCH_RELU=1 $$args -o $(BUILD)/switch_system $(RTL) tb/tb_u200_moe.sv && $(VVP) $(BUILD)/switch_system || exit 1; done

rtl-test: switch-test
	@mkdir -p $(BUILD)
	@for phase in 0 1 2 3; do $(IVERILOG) -g2012 -s tb_active_dma_promotion -Ptb_active_dma_promotion.PHASE=$$phase -o $(BUILD)/active_promotion $(RTL) tb/tb_active_dma_promotion.sv && $(VVP) $(BUILD)/active_promotion || exit 1; done
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.ROUTE_K=1 -Ptb_u200_moe.BUDGET=0 -Ptb_u200_moe.WAIT_FULL_EXPERT=1 -Ptb_u200_moe.OCM_SLOTS_PER_BANK=4 -o $(BUILD)/whole_expert $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/whole_expert
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.ROUTE_K=1 -Ptb_u200_moe.BUDGET=0 -Ptb_u200_moe.WAIT_FULL_EXPERT=1 -Ptb_u200_moe.OCM_SLOTS_PER_BANK=4 -Ptb_u200_moe.FUTURE_MODE=1 -o $(BUILD)/whole_prefetch $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/whole_prefetch
	$(IVERILOG) -g2012 -s tb_cross_layer_prefetch -o $(BUILD)/cross_layer_prefetch $(RTL) tb/tb_cross_layer_prefetch.sv
	$(VVP) $(BUILD)/cross_layer_prefetch
	$(IVERILOG) -g2012 -s tb_moe_layer_chain -o $(BUILD)/moe_layer_chain $(RTL) tb/tb_moe_layer_chain.sv
	$(VVP) $(BUILD)/moe_layer_chain
	$(IVERILOG) -g2012 -s tb_mac_chunk_pipeline -o $(BUILD)/mac_chunk_pipeline $(RTL) tb/tb_mac_chunk_pipeline.sv
	$(VVP) $(BUILD)/mac_chunk_pipeline
	$(IVERILOG) -g2012 -s tb_mac_chunk_pipeline -Ptb_mac_chunk_pipeline.H=16 -Ptb_mac_chunk_pipeline.F_TILE=32 -o $(BUILD)/mac_chunk_asymmetric $(RTL) tb/tb_mac_chunk_pipeline.sv
	$(VVP) $(BUILD)/mac_chunk_asymmetric
	$(IVERILOG) -g2012 -s tb_gu_slot_reuse -o $(BUILD)/gu_slot_reuse $(RTL) tb/tb_gu_slot_reuse.sv
	$(VVP) $(BUILD)/gu_slot_reuse
	$(IVERILOG) -g2012 -s tb_execution_lookahead -o $(BUILD)/lookahead $(RTL) tb/tb_execution_lookahead.sv
	$(VVP) $(BUILD)/lookahead
	$(IVERILOG) -g2012 -s tb_execution_lookahead -Ptb_execution_lookahead.SLOTS=128 -o $(BUILD)/lookahead_default $(RTL) tb/tb_execution_lookahead.sv
	$(VVP) $(BUILD)/lookahead_default
	$(IVERILOG) -g2012 -s tb_compute_handoff -o $(BUILD)/compute_handoff $(RTL) tb/tb_compute_handoff.sv
	$(VVP) $(BUILD)/compute_handoff
	$(IVERILOG) -g2012 -s tb_gu_early_start -o $(BUILD)/gu_early $(RTL) tb/tb_gu_early_start.sv
	$(VVP) $(BUILD)/gu_early
	$(IVERILOG) -g2012 -s tb_execution_selection -o $(BUILD)/execution_selection $(RTL) tb/tb_execution_selection.sv
	$(VVP) $(BUILD)/execution_selection
	$(IVERILOG) -g2012 -s tb_execution_selection -Ptb_execution_selection.SLOTS=128 -Ptb_execution_selection.EXPERTS=8 -o $(BUILD)/execution_default $(RTL) tb/tb_execution_selection.sv
	$(VVP) $(BUILD)/execution_default
	$(IVERILOG) -g2012 -s tb_synchronous_flush -o $(BUILD)/synchronous_flush $(RTL) tb/tb_synchronous_flush.sv
	$(VVP) $(BUILD)/synchronous_flush
	$(IVERILOG) -g2012 -s tb_execution_readiness_race -o $(BUILD)/execution_race $(RTL) tb/tb_execution_readiness_race.sv
	$(VVP) $(BUILD)/execution_race
	$(IVERILOG) -g2012 -s tb_demander_flush_priority -o $(BUILD)/demander_flush $(RTL) tb/tb_demander_flush_priority.sv
	$(VVP) $(BUILD)/demander_flush
	$(IVERILOG) -g2012 -s tb_router_gemv -o $(BUILD)/router_gemv $(RTL) tb/tb_router_gemv.sv
	$(VVP) $(BUILD)/router_gemv
	$(IVERILOG) -g2012 -s tb_router_topk_softmax -o $(BUILD)/router_softmax $(RTL) tb/tb_router_topk_softmax.sv
	$(VVP) $(BUILD)/router_softmax
	$(IVERILOG) -g2012 -s tb_striped_ddr -o $(BUILD)/striped_ddr $(RTL) tb/tb_striped_ddr.sv
	$(VVP) $(BUILD)/striped_ddr
	$(IVERILOG) -g2012 -s tb_u200_moe -o $(BUILD)/u200_moe $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_moe
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -o $(BUILD)/u200_three_layers $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_three_layers
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.QUANT_SHIFT=1 -o $(BUILD)/u200_layers_quant $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_layers_quant
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=8 -Ptb_u200_moe.REPEATS=2 -o $(BUILD)/u200_eight_layers $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_eight_layers
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=1 -o $(BUILD)/u200_future_hit $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_hit
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=2 -o $(BUILD)/u200_future_wrong $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_wrong
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=1 -Ptb_u200_moe.DDR_GAPS=0 -o $(BUILD)/u200_future_fast $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_fast
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=1 -Ptb_u200_moe.ABORT_FUTURE=1 -o $(BUILD)/u200_future_abort $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_abort
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=8 -Ptb_u200_moe.FUTURE_MODE=1 -Ptb_u200_moe.REPEATS=2 -o $(BUILD)/u200_future_eight $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_eight
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=1 -Ptb_u200_moe.OCM_SLOTS_PER_BANK=4 -o $(BUILD)/u200_future_capacity $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_capacity
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=2 -Ptb_u200_moe.DESCRIPTOR_ENTRIES=2 -Ptb_u200_moe.FIFO_DEPTH=2 -Ptb_u200_moe.EXPECT_EARLY_OVERLAP=0 -o $(BUILD)/u200_future_min_table $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_min_table
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.LAYERS=3 -Ptb_u200_moe.FUTURE_MODE=2 -Ptb_u200_moe.DESCRIPTOR_ENTRIES=4 -Ptb_u200_moe.FIFO_DEPTH=2 -o $(BUILD)/u200_future_table_pressure $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_future_table_pressure
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.PREDICTED_EXPERT=5 -o $(BUILD)/u200_wrong $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_wrong
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.BUDGET=0 -o $(BUILD)/u200_demand $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_demand
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.FLUSH_AT=10 -o $(BUILD)/u200_flush $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_flush
	$(IVERILOG) -g2012 -s tb_u200_moe -Ptb_u200_moe.ERROR_BANK=2 -o $(BUILD)/u200_error $(RTL) tb/tb_u200_moe.sv
	$(VVP) $(BUILD)/u200_error
	$(IVERILOG) -g2012 -s tb_candidate_scheduler_backpressure -o $(BUILD)/candidate_backpressure $(RTL) tb/tb_candidate_scheduler_backpressure.sv
	$(VVP) $(BUILD)/candidate_backpressure
	$(IVERILOG) -g2012 -s tb_combine_pipeline -o $(BUILD)/combine_pipeline $(RTL) tb/tb_combine_pipeline.sv
	$(VVP) $(BUILD)/combine_pipeline
	$(IVERILOG) -g2012 -s tb_byte_finalizer_pipeline -o $(BUILD)/byte_pipeline $(RTL) tb/tb_byte_finalizer_pipeline.sv
	$(VVP) $(BUILD)/byte_pipeline
	$(IVERILOG) -g2012 -s tb_byte_finalizer_pipeline -Ptb_byte_finalizer_pipeline.ENTRIES=1 -o $(BUILD)/byte_pipeline_1 $(RTL) tb/tb_byte_finalizer_pipeline.sv
	$(VVP) $(BUILD)/byte_pipeline_1
	$(IVERILOG) -g2012 -s tb_byte_finalizer_pipeline -Ptb_byte_finalizer_pipeline.ENTRIES=9 -o $(BUILD)/byte_pipeline_9 $(RTL) tb/tb_byte_finalizer_pipeline.sv
	$(VVP) $(BUILD)/byte_pipeline_9
	$(IVERILOG) -g2012 -s tb_byte_finalizer_pipeline -Ptb_byte_finalizer_pipeline.ENTRIES=64 -o $(BUILD)/byte_pipeline_64 $(RTL) tb/tb_byte_finalizer_pipeline.sv
	$(VVP) $(BUILD)/byte_pipeline_64
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -o $(BUILD)/tree5 $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree5
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.N=1 -Ptb_greedy_tree_equivalence.CHANNELS=1 -o $(BUILD)/tree1 $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree1
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.N=3 -o $(BUILD)/tree3 $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree3
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.N=16 -o $(BUILD)/tree16 $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree16
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.PIPELINED=1 -o $(BUILD)/tree5_pipe $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree5_pipe
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.N=1 -Ptb_greedy_tree_equivalence.CHANNELS=1 -Ptb_greedy_tree_equivalence.PIPELINED=1 -o $(BUILD)/tree1_pipe $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree1_pipe
	$(IVERILOG) -g2012 -s tb_greedy_tree_equivalence -Ptb_greedy_tree_equivalence.N=16 -Ptb_greedy_tree_equivalence.PIPELINED=1 -o $(BUILD)/tree16_pipe $(RTL) tb/tb_greedy_tree_equivalence.sv
	$(VVP) $(BUILD)/tree16_pipe
	$(IVERILOG) -g2012 -s tb_hardware_greedy_v1 -o $(BUILD)/greedy_v1 $(RTL) tb/tb_hardware_greedy_v1.sv
	$(VVP) $(BUILD)/greedy_v1
	$(IVERILOG) -g2012 -s tb_descriptor_table -o $(BUILD)/descriptor_table $(RTL) tb/tb_descriptor_table.sv
	$(VVP) $(BUILD)/descriptor_table
	$(IVERILOG) -g2012 -s tb_ocm_slot_ownership -o $(BUILD)/ocm_slot_ownership $(RTL) tb/tb_ocm_slot_ownership.sv
	$(VVP) $(BUILD)/ocm_slot_ownership
	$(IVERILOG) -g2012 -s tb_ocm_admission_tree -o $(BUILD)/ocm_admission_tree $(RTL) tb/tb_ocm_admission_tree.sv
	$(VVP) $(BUILD)/ocm_admission_tree
	$(IVERILOG) -g2012 -s tb_ocm_admission_tree -Ptb_ocm_admission_tree.BANKS=1 -Ptb_ocm_admission_tree.SLOTS=1 -o $(BUILD)/ocm_admission_single $(RTL) tb/tb_ocm_admission_tree.sv
	$(VVP) $(BUILD)/ocm_admission_single
	$(IVERILOG) -g2012 -s tb_ocm_admission_tree -Ptb_ocm_admission_tree.BANKS=4 -Ptb_ocm_admission_tree.SLOTS=4 -Ptb_ocm_admission_tree.PAIR_PRIMARY=1 -o $(BUILD)/ocm_admission_pair $(RTL) tb/tb_ocm_admission_tree.sv
	$(VVP) $(BUILD)/ocm_admission_pair
	$(IVERILOG) -g2012 -s tb_burst_dma_channel -o $(BUILD)/burst_dma $(RTL) tb/tb_burst_dma_channel.sv
	$(VVP) $(BUILD)/burst_dma
	$(IVERILOG) -g2012 -s tb_descriptor_manager -o $(BUILD)/descriptor_manager $(RTL) tb/tb_descriptor_manager.sv
	$(VVP) $(BUILD)/descriptor_manager
	$(IVERILOG) -g2012 -s tb_descriptor_dma_path -o $(BUILD)/descriptor_dma_path $(RTL) tb/tb_descriptor_dma_path.sv
	$(VVP) $(BUILD)/descriptor_dma_path
	$(IVERILOG) -g2012 -s tb_tile_prefetch_dataflow_top -o $(BUILD)/dataflow_top $(RTL) tb/tb_tile_prefetch_dataflow_top.sv
	$(VVP) $(BUILD)/dataflow_top
	$(IVERILOG) -g2012 -s tb_tile_prefetch_axi_top -o $(BUILD)/axi_top $(RTL) tb/tb_tile_prefetch_axi_top.sv
	$(VVP) $(BUILD)/axi_top
	$(IVERILOG) -g2012 -s tb_route_token_resolver -o $(BUILD)/route_token_resolver $(RTL) tb/tb_route_token_resolver.sv
	$(VVP) $(BUILD)/route_token_resolver
	$(IVERILOG) -g2012 -s tb_tile_prefetch_predictive_axi_top -o $(BUILD)/predictive_axi_top $(RTL) tb/tb_tile_prefetch_predictive_axi_top.sv
	$(VVP) $(BUILD)/predictive_axi_top
	$(IVERILOG) -g2012 -s tb_int8_swiglu_tile_backend -o $(BUILD)/numeric $(RTL) tb/tb_int8_swiglu_tile_backend.sv
	$(VVP) $(BUILD)/numeric
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -o $(BUILD)/moe_e2e $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_e2e
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.ROUTER_DELAY=20 -o $(BUILD)/moe_early $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_early
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.PREDICTED_EXPERT=5 -o $(BUILD)/moe_wrong $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_wrong
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.PREDICTED_EXPERT=5 -Ptb_moe_end_to_end.ROUTER_DELAY=20 -o $(BUILD)/moe_wrong_early $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_wrong_early
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.BUDGET_BYTES=0 -o $(BUILD)/moe_no_prefetch $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_no_prefetch
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.FLUSH_DELAY=20 -o $(BUILD)/moe_flush $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_flush
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.FLUSH_DELAY=7 -o $(BUILD)/moe_flush_admission $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_flush_admission
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.FLUSH_DELAY=70 -o $(BUILD)/moe_flush_completion $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_flush_completion
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.FLUSH_DELAY=150 -o $(BUILD)/moe_flush_resident $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_flush_resident
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.PREDICTED_EXPERT=4 -o $(BUILD)/moe_ready_first $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_ready_first
	$(IVERILOG) -g2012 -s tb_sequence_controller -o $(BUILD)/sequence $(RTL) tb/tb_sequence_controller.sv
	$(VVP) $(BUILD)/sequence
	$(IVERILOG) -g2012 -s tb_moe_end_to_end -Ptb_moe_end_to_end.AXI_ERROR=1 -o $(BUILD)/moe_axi_error $(RTL) tb/tb_moe_end_to_end.sv
	$(VVP) $(BUILD)/moe_axi_error

lint:
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module u200_moe_top $(RTL)
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module u200_moe_top -GLAYER_COUNT=3 $(RTL)
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module u200_moe_top "-GWAIT_FULL_EXPERT=1'b1" $(RTL)
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module u200_moe_top "-GSWITCH_RELU=1'b1" -GLAYER_COUNT=3 $(RTL)
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module u200_moe_top "-GSWITCH_RELU=1'b1" "-GWAIT_FULL_EXPERT=1'b1" $(RTL)
	$(VERILATOR) --lint-only --timing --unroll-count 1024 -Wall -Wno-fatal --top-module tile_prefetch_predictive_axi_top $(RTL)
	$(VERILATOR) --lint-only --timing -Wall -Wno-fatal --top-module hardware_greedy_v1 rtl/hardware_greedy_v1.sv

synth-ooc:
	vivado -mode batch -nojournal -nolog -source scripts/synth_ooc.tcl

synth-default:
	vivado -mode batch -nojournal -nolog -source scripts/synth_ooc.tcl -tclargs ../../results/m10_mac_pipeline_current/synthesis --full-default

golden:
	PYTHONPATH=../../src python scripts/compare_rtl_vs_python.py
	python scripts/compare_swiglu_numeric.py

alignment:
	PYTHONPATH=../../src python scripts/compare_u200_alignment.py

# 当前 RTL 的主模拟器共同激励诊断；不是旧 M10AlignmentSimulator。
alignment-main:
	PYTHONPATH=../../src python scripts/compare_unified_alignment.py $(ALIGNMENT_ARGS)

alignment-control:
	PYTHONPATH=../../src python scripts/compare_unified_alignment.py --config ../../configs/hardware/u200_control_alignment.yaml $(ALIGNMENT_ARGS)

alignment-interface:
	PYTHONPATH=../../src python scripts/compare_unified_alignment.py --config ../../configs/hardware/u200_interface_alignment.yaml $(ALIGNMENT_ARGS)

boundary-validation:
	PYTHONPATH=../../src python scripts/run_boundary_validation.py $(ALIGNMENT_ARGS)

descriptor-boundary:
	PYTHONPATH=../../src python scripts/run_boundary_validation.py --config ../../configs/hardware/u200_descriptor_alignment.yaml --output ../../results/m10_descriptor_boundary $(ALIGNMENT_ARGS)

ingress-boundary:
	PYTHONPATH=../../src python scripts/run_boundary_validation.py --config ../../configs/hardware/u200_ingress_alignment.yaml --output ../../results/m10_ingress_boundary $(ALIGNMENT_ARGS)

control-sensitivity:
	PYTHONPATH=../../src python scripts/analyze_control_sensitivity.py

# 使用XSim完整默认配置。可传 ALIGNMENT_ARGS='--netlist /abs/path/funcsim.v'。
alignment-default:
	PYTHONPATH=../../src python scripts/compare_u200_default_alignment.py $(ALIGNMENT_ARGS)

clean:
	rm -rf $(BUILD) reports/*.json
