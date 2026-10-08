# 1 token version 验证记录（2026-10-08）

新增multi-token机会实验的YAML、驱动、测试、报告与总览/索引条目已移除。
源文件与结果归档到Git忽略的 `dist/multi-token-research-archive-20261008/`，
可从归档或上一条提交恢复；不会进入新源码包或Git提交。
历史BatchRoutingEvent/BatchPredictorOutput是主模型、CLI与RTL对齐的内部依赖，
单token以batch_size=1表示，未破坏它们。此版本不承诺multi-token RTL。

## 实际执行

- 清理后的工作区：`pytest -o addopts='' -q`，591 passed，14条依赖弃用warning。
- 本轮没有修改 `src/edge_moe_sim/` 或任何RTL/Testbench/XDC。
- 同一RTL源码此前独立快照的 `make -C hardware/m10_u200 rtl-test` 退出码0，89条PASS。
- 白名单排除凭据、Git元数据、Vivado生成工程、日志、波形、build/reports。

RTL日志有Icarus关于always_* constant select敏感列表的提示，以及仿真断言`$error`
不可综合的提示；不是零warning结果，不将功能回归通过当作Vivado综合/PnR验收。
本轮没有运行新的综合、时序实现或板上测试。

## 产物

```bash
python scripts/prepare_single_token_release.py
```

新快照为 `dist/edge-moe-single-token-20261008-clean/`，压缩包为同名tar.gz。
逐文件摘要见快照内RELEASE_MANIFEST.json。旧快照不覆盖，但不作为当前干净版本使用。

## Git与待决策事项

清理提交消息为 `1 token version`，origin为git@github.com:MasLiang/Expert-Prefetch.git。
未执行push或创建tag，许可证尚未选定。
