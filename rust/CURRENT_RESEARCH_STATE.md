# 当前 BPE 研究状态

当前推荐作速度候选的是 `experiments/radical/owned_integer_hash` 的 `--integer-hash ahash`；同二进制保留默认 `std` 作控制，不改已冻结实测源。它在唯一 owner 的频率/位置索引、grouped birth chain、inline posting 结构上只更换哈希构造器。训练仍与串行 greedy 的完整规则、频率和最终 token 相同，输入不依赖空格边界。

```sh
cargo build --manifest-path rust/experiments/radical/owned_integer_hash/Cargo.toml --release --locked --target-dir rust/target
rust/target/release/radical-owned-integer-hash --input rust/fixtures/ablation/en-4m-continuous.json --workers 4 --chunk-size 4096 --rules 3000 --min-frequency 2 --heap-policy lazy --integer-hash ahash
```

[最新有限复核](batch_results/radical-local-hash-v1/README.md)中，4 MiB/3000 规则 n=2 的 aHash 单核/四核中位数为英文 1.310/0.533 秒、中文 0.585/0.303 秒；相对 std 四格均改善约 1.43–1.46×。自身 1→4 仍仅 2.46×/1.93×，所以 Goal 继续，不能以对旧直接串行的 3.06×/3.25× 宣布多核目标完成。后两个比例还混合了哈希工程差异，CF/CF16 的同 aHash 控制已通过完整正确性门控，性能尚未测量。机器只提供六个可见 CPU，尚无几十核/双路证据。

## 下一步只推进这些问题

- **公平的直接串行常数基线**：[串行控制](experiments/radical/serial_integer_hash/DESIGN.md)已实现 CF、CF16 的 std/aHash 选择，通过 156 次完整 oracle 加 4 次预期域拒绝；下一步才做有限同窗口比较。不能把旧串行保留 std 的差距算成并行算法创新。
- **扩大精确批次**：[出生时邻接摘要设计](EXACT_BATCH_WIDENING_NEXT.md)及[独立反例审查](BIRTH_NEIGHBOR_REVIEW.md)。新旧候选使用较新 key 出生时的保守邻居掩码来排除实际重叠；每 key 额外 8 字节，饱和与额外读取可能使它不划算。[首轮8调用小测](batch_results/radical-neighbor-certificate-v1/README.md)已通过完整轨迹，英文批次 69→60、中文 87→82，但英文更慢、中文仅四核改善，且峰值内存上升；不默认整合。下一步是[首次查询时构建并缓存摘要](LAZY_NEIGHBOR_NEXT.md)：候选缓存的键数上界为 R+B≤2R，总额外 posting 扫描仍可约束为 O(N)；尚未实现。
- **复杂度边界**：[权重定位与 AA 整数排序方向](OWNER_ROUTE_NEXT.md)。历史 posting 总访问 O(N) 不等于完整训练线性；piece 二分、堆、AA 比较排序及每批管理有独立成本。原地整数排序只是设计，尚未实现。

## 已筛过，避免无证据重做

| 方向 | 当前决策及证据 |
|---|---|
| grouped birth + inline posting | 保留为结构基础，见[组合复核](batch_results/radical-layout-combo-v1/README.md)。|
| owner 内连续提交、直接旧 key 扣减 | 有潜力，但 n=2 未稳定替代原结构；[归档](batch_results/radical-planning-integrated-v1/README.md)。|
| 大 posting 分散填充、增加逻辑 owner | 额外组织成本尚无充分净收益，不默认叠加。|
| u16 端点 | 同容量语料载荷减半，但转换峰值、总 RSS 和速度不保证改善；词表超域自动回 u32，长度仍为 u32。[控制筛选](batch_results/radical-controlled-longscreen-v1/README.md)。|
| 持久路由缓存 | 成功减少初始化和扫描；相位重置已控制，英文/中文净速度方向不同，暂不默认。|
| 按规则上下文累计 | 大幅减少实际路由哈希更新，却没有一致 W4 收益；局部 scratch 头的后续小测也未获得一致收益。|
| 空间逐位置证书 | 旧 native 已有并行 extra-only，owner 有串行预算 probe；两者不是未探索的新点，不重复包装为新算法。|

先用 256 KiB 小筛选淘汰候选，只为具体疑问做有限 4 MiB 测量，最终收敛后再完整矩阵。所有已发布数据保留 source/binary/fixture 哈希、完整轨迹校验、进程 CPU 与训练 VmHWM 口径。另一个 tokenizer benchmark 工作区始终只读。外存训练尚未实现。
