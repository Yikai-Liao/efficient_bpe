# 当前 BPE 研究状态

当前推荐作速度候选的是 `experiments/radical/owned_integer_hash` 的 `--integer-hash ahash`；同二进制保留默认 `std` 作控制，不改已冻结实测源。它在唯一 owner 的频率/位置索引、grouped birth chain、inline posting 结构上只更换哈希构造器。训练仍与串行 greedy 的完整规则、频率和最终 token 相同，输入不依赖空格边界。

```sh
cargo build --manifest-path rust/experiments/radical/owned_integer_hash/Cargo.toml --release --locked --target-dir rust/target
rust/target/release/radical-owned-integer-hash --input rust/fixtures/ablation/en-4m-continuous.json --workers 4 --chunk-size 4096 --rules 3000 --min-frequency 2 --heap-policy lazy --integer-hash ahash
```

[最新有限复核](batch_results/radical-local-hash-v1/README.md)中，4 MiB/3000 规则 n=2 的 aHash 单核/四核中位数为英文 1.310/0.533 秒、中文 0.585/0.303 秒；相对 std 四格均改善约 1.43–1.46×。自身 1→4 仍仅 2.46×/1.93×，所以 Goal 继续，不能以对旧直接串行的 3.06×/3.25× 宣布多核目标完成。后两个比例还混合了哈希工程差异。[同 aHash 直接串行小测](batch_results/radical-serial-integer-quick-v1/README.md)已补齐：256 KiB n=1 中 CF32 checked 的英文/中文为 0.0386/0.0198 秒，同窗 owner 四核为 0.0314/0.0180 秒，差距明显缩小；尚不能外推 4 MiB 或作稳定排序。机器只提供六个可见 CPU，尚无几十核/双路证据。

## 下一步只推进这些问题

- **公平的直接串行常数基线**：[串行控制](experiments/radical/serial_integer_hash/DESIGN.md)已完成 22 调用同窗口小测，全部完整轨迹匹配。相同 backend/bounds 的八组 std→aHash 中七组更快，但 n=1 不作稳定排名；不能把旧串行保留 std 的差距算成并行算法创新。
- **扩大精确批次**：[出生时摘要](EXACT_BATCH_WIDENING_NEXT.md)之后已实现[按需摘要](experiments/radical/owned_lazy_neighbor/DESIGN.md)。它不增加每个 Entry 的字段，只为首次查询 key 扫一次 posting 并缓存；build 数≤R+B，访问数≤所有 retained 历史 posting。17 项 Rust 测试及 200/200 oracle 通过，14 次小测完整匹配。英文摘要扫描由 530,725 降至 32,801 个位置、中文由 149,188 降至 4,751；额外工作与内存降低，但 W4 调用没有一致优于 type 控制，不默认整合。详见[归档](batch_results/radical-lazy-neighbor-gate-v1/README.md)。
- **复杂度边界**：[原地 AA radix](experiments/radical/owned_aa_radix/DESIGN.md)已实现，排序固定 u32 域下最坏 O(H)、辅助数组栈载荷上界 24 KiB，无 H 长度缓冲；原控制仍是 Rayon 并行比较排序。13 项 Rust 测试、160/160 oracle 和 12 次小测通过。AA 密集两例中，W4 的 radix 排序阶段未胜过标准排序；保留复杂度选项，不作为速度默认。详见[归档](batch_results/radical-aa-radix-gate-v1/README.md)。
- **消除一次整批屏障**：[方向标记端点与批前邻居恢复](FUSED_ENDPOINT_SNAPSHOT_NEXT.md)是新的待验证设计，试图在每次匹配内融合规划与写入，不复制语料。需要独立交错模型、弱内存序证明及 tagged-two-pass 控制；目前不作性能承诺。

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
