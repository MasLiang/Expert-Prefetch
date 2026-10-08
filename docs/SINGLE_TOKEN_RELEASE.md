# 单 token Simulator / U200 RTL 源码发布准备

## 发布范围

这是当前版本的源码快照，建议候选标识 `single-token-20261008`，并非已经创建的 Git tag。
保留 Python 主模型、配置、CLI、测试、实验复现入口和 U200 手写 RTL/Testbench/Tcl/XDC。
Simulator 主入口是 UnifiedSequenceSimulator；M10AlignmentSimulator 对当前RTL已失效。
RTL 支持真实 INT8 Router、外部 predictor、SwiGLU/Switch ReLU、单 token 多层链、
有限 OCM、四通道 AXI、GU 就绪启动及投机 commit/cancel。

保留 batch 相关源码以避免破坏回归，但 multi-token 新方向不纳入这个 release 的功能承诺。
不声称200MHz时序闭合、真实DDR周期校准、整LLM实现、板上部署或任意profile周期精确对齐。
大batch、小OCM的expert_locked活性限制见 MULTI_TOKEN_OPPORTUNITY_STUDY.md。

## 当前 Git 状态

2026-10-08检查：工作目录中存在一个只读空 `.git/`，没有有效仓库元数据。
`git rev-parse --show-toplevel`返回not a git repository；父目录也不是仓库。
当前环境对根目录 `.git` 仅允许读取，所以本轮没有尝试删除它或初始化覆盖它。
没有设置远端、创建commit/tag或上传任何文件。

## 文件组织

原始工作区保留不动；以下目录可提交：

```text
src/                       simulator
configs/                   模型/硬件/实验参数
examples/                  使用和实验入口
tests/                     Python回归
docs/                      规格、诊断、研究范围
scripts/                   release准备工具
hardware/m10_u200/
  rtl/                     手写RTL
  tb/                      Testbench
  configs/                 IP/RTL参数配方
  scripts/                 综合/仿真/比较脚本、XDC
  traces/                  固定小激励
  Makefile, README.md
results/<有限夹具目录>/     旧artifact测试的必需数据
```

不提交 `.aws/.codex/.agents`、缓存、波形、日志、build、Vivado生成工程或DCP/bitstream。
原 results约7.5GiB，hardware约748MiB；不会直接打包整个目录。
部分旧pytest用例依赖M7–M9 artifact和prefetch_dense/inputs.json，故保留明确白名单，
而不是简单忽略所有results后宣称完整测试仍能运行。
其他历史图/数据仍在本机，文档中的历史结果链接可能不包含在源码包中；
它们不是该发布包可重跑的功能测试依赖。

## 准备和验证

```bash
python scripts/prepare_single_token_release.py
cd dist/edge-moe-single-token-20261008
python -m pip install -e '.[test]'
pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test
```

Python项目声明3.11+；完整研究图复现另需matplotlib等实验依赖。
RTL功能回归需要Icarus Verilog，XSim/综合入口需要Vivado2023.2。
手写 RTL 和IP生成recipe作为版本源，不把生成 `.xpr` 作为版本源。
现有 `synth_ooc.tcl` 从脚本位置解析路径，读取所有RTL与ooc_clock.xdc：

```bash
vivado -mode batch -source hardware/m10_u200/scripts/synth_ooc.tcl
```

它执行OOC综合，不创建板级MIG/PCIe平台，也不执行PnR；默认缩小资源配置，
完整默认参数需查看Makefile的synth-default入口。发布整理没有新运行综合/PnR。

准备脚本生成未包含Git历史的独立源码目录、tar.gz和逐文件SHA256清单。
不会删除、移动或覆盖原工程，也不会修改simulator/RTL算法。

## 上传前需要确定的事项

1. 确认远端URL、可见性及代码/第三方材料公开权限。
2. 确认许可证；本轮不擅自分配MIT/Apache等许可。
3. 在可写的独立源码目录初始化仓库，审查文件后commit/push。
4. 验收完成后再创建正式tag/release；当前只准备候选快照。

例如在导出的源码目录（不是当前只读空 `.git` 的工作区）操作：

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Freeze single-token simulator and U200 RTL"
# git remote add origin <确认后的远端URL>
# git push -u origin main
# 验收后再创建与推送tag
```

不要将快照内保留的历史研究profile误称为U200板级实测或本次发布性能承诺。
