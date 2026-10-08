# 中文说明：只读打开既有 DDR example，导出独立完整尺寸验证，不保存原工程。
if {$argc != 7} {error "usage: output example_xpr H F FT experts layers"}
lassign $argv output example_xpr H F FT experts layers
set output [file normalize $output]
set root [file normalize [file join [file dirname [info script]] ..]]
open_project -read_only [file normalize $example_xpr]
save_project_as -exclude_run_results switch_ddr [file join $output project]
if {[get_property PART [current_project]] ne "xcu200-fsgd2104-2-e"} {error "wrong part"}
set receipt [open [file join $output ip_readback.txt] w]
puts $receipt "vivado=[version -short]"
foreach {key want} {CONFIG.C0.DDR4_MemoryType RDIMMs CONFIG.C0.DDR4_MemoryPart MTA18ASF2G72PZ-2G3 CONFIG.C0.DDR4_TimePeriod 833 CONFIG.C0.DDR4_AxiDataWidth 128 CONFIG.C0.DDR4_DataWidth 72 CONFIG.C0.DDR4_Ecc true CONFIG.Simulation_Mode BFM} {
    set got [get_property $key [get_ips ddr4_0]]
    puts $receipt "$key=$got"
    if {$got ne $want} {error "DDR configuration differs: $key=$got expected=$want"}
}
foreach {key want} {CONFIG.ADDR_WIDTH 34 CONFIG.DATA_WIDTH 128 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} {
    set got [get_property $key [get_ips ddr_axi_cdc]]
    puts $receipt "cdc.$key=$got"
    if {$got ne $want} {error "CDC configuration differs: $key=$got expected=$want"}
}
close $receipt
foreach path [concat [glob -directory [file join $root rtl] *.sv] [list [file join $root tb ddr4_bank_backend.sv] [file join $root tb tb_switch_full_size.sv]]] {
    if {[llength [get_files $path]] == 0} {add_files -fileset sim_1 $path}
}
set sim [get_filesets sim_1]
set_property top tb_switch_full_size $sim
set_property target_simulator XSim [current_project]
set_property verilog_define M10_SWITCH_DDR4 $sim
set_property -name xsim.compile.xvlog.more_options -value {-d M10_SWITCH_DDR4} -objects $sim
set_property xsim.elaborate.debug_level off $sim
set_property xsim.simulate.log_all_signals false $sim
set_property xsim.simulate.runtime 1100ms $sim
update_compile_order -fileset sim_1
set slot [expr {$H == 16 ? 512 : 131072}]
foreach {name whole prefetch} {A_whole_demand 1 0 B_stream_demand 0 0 C_whole_full_prefetch 1 1 D_stream_tile_prefetch 0 1} {
    set_property generic "H=$H F=$F FT=$FT EXPERTS=$experts LAYERS=$layers WHOLE=$whole SLOT_BYTES=$slot SLOTS_PER_BANK=32" $sim
    set_property -name xsim.simulate.xsim.more_options -value "-testplusarg PREFETCH=$prefetch -testplusarg PATTERN=0 -testplusarg TIMELINE=timeline.csv" -objects $sim
    export_simulation -simulator xsim -absolute_path -force -directory [file join $output cases $name]
}
puts "SWITCH_DDR_PREPARE_PASS output=$output"
close_project
