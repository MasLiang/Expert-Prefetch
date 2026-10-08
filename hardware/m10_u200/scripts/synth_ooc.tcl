# 中文说明：验证本机安装的 U200 device part，并进行真实计算顶层 OOC 综合。
# 该检查不绑定板级 XPFM，也不代表已完成板级 PnR 或上板测量。
set root [file normalize [file join [file dirname [info script]] ..]]
set output_dir [file join $root reports]
if {[llength $argv] > 0} {set output_dir [file normalize [lindex $argv 0]]}
set full_default 0
if {[llength $argv] > 1} {
  if {[lindex $argv 1] ne "--full-default"} {error "expected --full-default"}
  set full_default 1
}
file mkdir $output_dir
set candidates [get_parts -quiet xcu200-fsgd2104-2-e]
if {[llength $candidates] != 1} {
  error "BLOCKED: required installed U200 device part absent"
}
set part [lindex $candidates 0]
set manifest [open [file join $output_dir synthesis_source_hashes.txt] w]
puts $manifest [exec sha256sum {*}[lsort [glob [file join $root rtl *.sv]]]]
puts $manifest [exec sha256sum [file join $root scripts ooc_clock.xdc]]
puts $manifest "vivado=[version -short] part=$part full_default=$full_default"
close $manifest
read_verilog -sv [file join $root rtl package.sv]
foreach source [lsort [glob [file join $root rtl *.sv]]] {
  if {[file tail $source] ne "package.sv"} {read_verilog -sv $source}
}
read_xdc [file join $root scripts ooc_clock.xdc]
if {$full_default} {
  synth_design -top u200_moe_top -part $part -mode out_of_context
} else {
  synth_design -top u200_moe_top -part $part -mode out_of_context \
    -generic MAX_CANDIDATES=4 -generic TILES_PER_EXPERT=4 \
    -generic DESCRIPTOR_TABLE_ENTRIES=16 -generic NUM_OCM_BANKS=2 \
    -generic SLOTS_PER_BANK=4 -generic OCM_SLOT_BYTES=512 -generic DMA_FIFO_DEPTH=4
}
report_utilization -hierarchical -file [file join $output_dir synthesis_utilization.rpt]
report_timing_summary -file [file join $output_dir synthesis_timing.rpt]
report_timing -max_paths 20 -nworst 1 -file [file join $output_dir synthesis_critical_paths.rpt]
report_drc -file [file join $output_dir synthesis_drc.rpt]
write_checkpoint -force [file join $output_dir tile_prefetch_ooc.dcp]
puts "PASS: U200 installed-part OOC synthesis; platform/board validation remains pending"
