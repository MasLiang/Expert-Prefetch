# 单 token 发布候选验证记录（2026-10-08）

候选是源码快照，不是已上传的Git release。以下在独立导出目录运行，
没有依赖原工作区的7.5GiB完整results或Vivado build/reports。

## 实际执行

- `python scripts/prepare_single_token_release.py`：698个白名单文件；源码77871978byte。
- `pytest -o addopts='' -q`：599 passed，14条已有依赖弃用warning，12.34秒。
- `make -C hardware/m10_u200 rtl-test`：退出码0，日志89条PASS。
- 逐文件SHA256校验：测试后所有清单源文件保持不变。
- 压缩包检查：698个源文件+RELEASE_MANIFEST.json；不含凭据、Git元数据、build、reports、
  Vivado生成工程、日志、波形或checkpoint。

RTL日志有Icarus关于always_* constant select敏感列表的提示，以及仿真断言`$error`
不可综合的提示；不是零warning结果，不将功能回归通过当作Vivado综合/PnR验收。
本轮没有运行新的综合、时序实现或板上测试。

## 产物

源码目录：`dist/edge-moe-single-token-20261008/`。

压缩包：`dist/edge-moe-single-token-20261008.tar.gz`，8189075byte，约7.8MiB。

压缩包SHA256：

```text
62e926a273cdecb504d118646256f7430ede577aa0329d49720cffda4c009905
```

本记录在验证后添加，故不属于上述已冻结压缩包的698个文件；可随Git提交或作为release附件。
RTL完整日志位于本机 `/tmp/edge-moe-single-token-release-rtl.log`，不作为源码提交。

## Git与待决策事项

原工作区只读空`.git`不是有效仓库；没有初始化、commit、tag或push。
已准备更严格的.gitignore并修复根README的本机绝对链接。
公开/私有远端、许可证与正式tag名称尚未决定，不能宣称已完成上传。
独立源码目录保留原有工程结构，避免移动模块/文档破坏路径依赖。
