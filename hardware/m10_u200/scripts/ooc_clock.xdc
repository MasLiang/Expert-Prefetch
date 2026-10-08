# OOC 核心检查目标；外部 AXI/host IO 时序由未来板级 wrapper 单独约束。
create_clock -name core_clk -period 5.000 [get_ports clk]
