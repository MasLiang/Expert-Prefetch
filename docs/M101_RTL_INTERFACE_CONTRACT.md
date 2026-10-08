# M10.1 参数化 RTL 接口合同（v0.9 实现冻结）

本文件冻结 M10.1 控制面和数据面的四个缺失接口。它是 vendor-neutral 合同：U200
的 AXI/DDR wrapper 只能适配这些接口，不能改变其 ownership、时序或计数语义。

## 1. Candidate 到 logical F-tile descriptor

`control_transaction_parser` 接收 prediction transaction；candidate table 保存
`expert_id/rank/confidence/next_depth`。descriptor generator 的输入为成功 admission
的 `(epoch, layer, expert, depth, request_class)`，输出严格有序的两个模板：GU 后
DOWN。地址不在 RTL 中猜测，来自 host-provided address table：

```text
gu.external_addr   = gu_base[layer][expert]   + gu_offset[layer][expert][depth]
down.external_addr = down_base[layer][expert] + down_offset[layer][expert][depth]
```

每个模板包含精确 `byte_count[31:0]`，包括末尾不足一 tile 的情况。两个模板经
`descriptor_manager` 原子 admission 后才产生任何 DMA token。

## 2. Router truth 到 live request

Router parser 输出 `(epoch, layer, expert)`。key resolver 在 descriptor allocation
时保存 `Key=(epoch,layer,expert,tile_id,matrix_kind) -> (request_id,generation)`。
Router 命中以该 `(request_id,generation)` 调用顶层 `route_valid/route_token`：

- true：queued spec 进入 PromotedFIFO；inflight spec 仅 commit；resident spec 标记 committed；
- false：queued token tombstone；inflight 设置 `drop_on_complete`；
- stale/missing key：计入 stale counter，绝不修改 OCM/DMA 状态。

key resolver 必须是有界表；满时阻塞 allocation，不允许丢失 key。

## 3. DMA payload streaming ABI

DMA command 与 payload 分离。`burst_dma_channel` 负责 setup/payload-cycle timing 与
token 仲裁；适配器对每个 `dma_burst_start` 发出：

```text
cmd_valid/ready: request_id, generation, external_addr, burst_bytes
```

读响应使用独立接口：

```text
payload_valid/ready: request_id, generation, ocm_slot, slot_generation,
                     byte_offset, data[127:0], keep[15:0], last
```

`ocm_payload_ram` 的写保护必须在每个 payload beat 上检查 tag。DMA 计时与 payload
backpressure 分离；response `last` 退休前禁止回收 request 或 slot。

## 4. Compute 与 partial output ABI

compute request 是同一 F-tile 的 GU/DOWN physical pair。`tile_compute_controller`
在两者均 `COMMITTED` 后执行 `GU -> nonlinear -> DOWN`，并在 complete 后释放 pair。
partial output ABI：

```text
tile_partial_valid/ready: epoch, layer, expert, tile_id, partial_y[ACC_W]
```

默认 `ACC_W=32`，有符号二补码 wraparound；实际 GEMM backend 可以参数化替换它。
`partial_output_accumulator` 对每个 `(epoch,layer,expert)` 维护 bitmap，重复 tile
报告错误且不重复累加；required tiles 全部完成后运行 `combine_cycles`。

这些规则不绑定量化格式，也不代表真实 GEMM 数值精度模型。
