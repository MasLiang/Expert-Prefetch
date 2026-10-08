# 中文说明：独立 DDR 验证工程；不修改交付工程的 RTL、XDC 或 Vivado settings。
# Vivado 2023.2 / Tcl 8.5 兼容。新版 ipcfg helper 需要 try/Tcl 8.6，不能在此使用。
if {$argc != 1} {error "usage: -tclargs <absolute output directory>"}
set output [file normalize [lindex $argv 0]]
set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir $output
set example_xpr [file join $output example ddr4_0_ex ddr4_0_ex.xpr]
if {![file exists $example_xpr]} {
    create_project ddr_seed [file join $output seed] -part xcu200-fsgd2104-2-e
    source -notrace [file join $root configs ddr4_0.tcl]
    if {[llength [get_ips ddr4_0]] != 1} {error "DDR IP recreation failed"}
    generate_target all [get_ips ddr4_0]
    open_example_project -dir [file join $output example] [get_ips ddr4_0]
    close_project
}
open_project $example_xpr
set expected [list \
    CONFIG.C0.DDR4_MemoryType RDIMMs \
    CONFIG.C0.DDR4_MemoryPart MTA18ASF2G72PZ-2G3 \
    CONFIG.C0.DDR4_TimePeriod 833 \
    CONFIG.C0.DDR4_InputClockPeriod 3332 \
    CONFIG.C0.DDR4_DataWidth 72 \
    CONFIG.C0.DDR4_AxiSelection true \
    CONFIG.C0.DDR4_AxiDataWidth 128 \
    CONFIG.C0.DDR4_AxiNarrowBurst false \
    CONFIG.C0.DDR4_Ecc true \
    CONFIG.C0.DDR4_Mem_Add_Map ROW_COLUMN_BANK \
    CONFIG.C0.DDR4_AxiArbitrationScheme RD_PRI_REG \
    CONFIG.Simulation_Mode BFM]
set receipt [open [file join $output ip_readback.txt] w]
puts $receipt "vivado=[version -short]"
puts $receipt "part=[get_property PART [current_project]]"
foreach {key want} $expected {
    set got [get_property $key [get_ips ddr4_0]]
    puts $receipt "$key=$got"
    if {$got ne $want} {close $receipt; error "DDR_READBACK_FAIL $key want=$want got=$got"}
}
if {[llength [get_ips ddr_axi_cdc]] == 0} {
    source -notrace [file join $root configs ddr_axi_cdc.tcl]
}
foreach {key want} {CONFIG.ADDR_WIDTH 34 CONFIG.DATA_WIDTH 128 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} {
    set got [get_property $key [get_ips ddr_axi_cdc]]
    puts $receipt "cdc.$key=$got"
    if {$got ne $want} {close $receipt; error "CDC_READBACK_FAIL $key want=$want got=$got"}
}
close $receipt
generate_target all [get_ips ddr_axi_cdc]
# 相同 source 已存在时跳过，不能复制 RTL 以造成版本漂移。
foreach path [concat [glob -directory [file join $root rtl] *.sv] \
    [list [file join $root tb ddr4_bank_backend.sv] [file join $root tb tb_u200_moe.sv]]] {
    if {[llength [get_files $path]] == 0} {add_files -fileset sim_1 $path}
}
set_property top tb_u200_moe [get_filesets sim_1]
set_property verilog_define M10_DDR4_MODEL [get_filesets sim_1]
# 2023.2 export_simulation 必须显式将宏传给 xvlog，不能只依赖 xelab 的 -d。
set_property -name xsim.compile.xvlog.more_options -value {-d M10_DDR4_MODEL} -objects [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value {-testplusarg ALIGN_MAIN} -objects [get_filesets sim_1]
set_property xsim.simulate.runtime 1ms [get_filesets sim_1]
update_compile_order -fileset sim_1
foreach {name mode} {no_prefetch 0 hit 1 wrong 2} {
    # 禁用 bench 默认的当前层 predictor，三组只比较跨层预测；与主模型夹具一致。
    set_property generic "LAYERS=3 FUTURE_MODE=$mode BUDGET=0 DDR_GAPS=0" [get_filesets sim_1]
    export_simulation -simulator xsim -absolute_path -force -directory [file join $output cases $name]
}
puts "DDR_PREPARE_PASS output=$output"
close_project
