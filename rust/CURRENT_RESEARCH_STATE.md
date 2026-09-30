# 当前 BPE 研究状态

当前推荐作速度候选的是 `experiments/radical/owned_integer_hash` 的 `--integer-hash ahash`；同二进制保留默认 `std` 作控制，不改已冻结实测源。它在唯一 owner 的频率/位置索引、grouped birth chain、inline posting 结构上只更换哈希构造器。训练仍与串行 greedy 的完整规则、频率和最终 token 相同，输入不依赖空格边界。

```sh
cargo build --manifest-path rust/experiments/radical/owned_integer_hash/Cargo.toml --release --locked --target-dir rust/target
rust/target/release/radical-owned-integer-hash --input rust/fixtures/ablation/en-4m-continuous.json --workers 4 --chunk-size 4096 --rules 3000 --min-frequency 2 --heap-policy lazy --integer-hash ahash
```

[最新有限复核](batch_results/radical-local-hash-v1/README.md)中，4 MiB/3000 规则 n=2 的 aHash 单核/四核中位数为英文 1.310/0.533 秒、中文 0.585/0.303 秒；相对 std 四格均改善约 1.43–1.46×。自身 1→4 仍仅 2.46×/1.93×，所以 Goal 继续，不能以对旧直接串行的 3.06×/3.25× 宣布多核目标完成。后两个比例还混合了哈希工程差异。[同 aHash 直接串行小测](batch_results/radical-serial-integer-quick-v1/README.md)已补齐：256 KiB n=1 中 CF32 checked 的英文/中文为 0.0386/0.0198 秒，同窗 owner 四核为 0.0314/0.0180 秒，差距明显缩小；尚不能外推 4 MiB 或作稳定排序。机器只提供六个可见 CPU，尚无几十核/双路证据。

上一轮小输入筛选收敛了三个 Rust 原型：[自适应切区、出生位置重放及计数内联报告](ADAPTIVE_REPLAY_REPORT.md)。28/27/30 项 lib 测试、286 次独立完整轨迹门控和 66 次轻量计时通过。adaptive 改善扫描均衡但没有通用净收益；原 replay 删掉 8 B BirthNode，却在自然语料增加总耗时。追加 replay-inline 后，第二窗口 W4 EN/ZH 为 50.78/21.28 ms，自身 W1→W4 仅 1.20×/1.51×，公平直接串行/W4 为 0.74×/1.12×。计数 heap 字节减少、总 HWM 未一致下降；没有达成 3×，不切换通用速度默认。后续实现与计时由主 agent 完成，不再使用 subagent。

已追加[16 MiB、32,000 实际合并的核心方案复核](batch_results/radical-full-v1/README.md)。原 1,000 调用矩阵在 135 条正式结果后停止，收敛后新增 28 次调用，用时 108.7 秒。主表使用同一补测窗口 n=2：英文旧 owner/自适应/出生链/inline 为 2.591/2.915/2.908/3.627 秒；中文为 1.641/1.577/1.594/1.987 秒。inline 与同二进制 chain 对照两者均慢约 25%，不是通用最优。英文旧 owner 本窗最快，中文 adaptive 与 chain 的范围交叠；相对同窗最快直接串行仅 1.94×/1.69×。内存高水位也未一致改善，不切换默认。全部 163 次独立正式调用完整指纹匹配，原窗口另有 135 次完整 trace 校验；原矩阵的零散结果仅保留为探索性数据。

## 下一步只推进这些问题

- **公平的直接串行常数基线**：[串行控制](experiments/radical/serial_integer_hash/DESIGN.md)已完成 22 调用同窗口小测，全部完整轨迹匹配。相同 backend/bounds 的八组 std→aHash 中七组更快，但 n=1 不作稳定排名；不能把旧串行保留 std 的差距算成并行算法创新。
- **扩大精确批次**：[出生时摘要](EXACT_BATCH_WIDENING_NEXT.md)之后已实现[按需摘要](experiments/radical/owned_lazy_neighbor/DESIGN.md)。它不增加每个 Entry 的字段，只为首次查询 key 扫一次 posting 并缓存；build 数≤R+B，访问数≤所有 retained 历史 posting。17 项 Rust 测试及 200/200 oracle 通过，14 次小测完整匹配。英文摘要扫描由 530,725 降至 32,801 个位置、中文由 149,188 降至 4,751；额外工作与内存降低，但 W4 调用没有一致优于 type 控制，不默认整合。详见[归档](batch_results/radical-lazy-neighbor-gate-v1/README.md)。
- **复杂度边界**：[原地 AA radix](experiments/radical/owned_aa_radix/DESIGN.md)已实现，排序固定 u32 域下最坏 O(H)、辅助数组栈载荷上界 24 KiB，无 H 长度缓冲；原控制仍是 Rayon 并行比较排序。13 项 Rust 测试、160/160 oracle 和 12 次小测通过。AA 密集两例中，W4 的 radix 排序阶段未胜过标准排序；保留复杂度选项，不作为速度默认。详见[归档](batch_results/radical-aa-radix-gate-v1/README.md)。
- **消除一次整批屏障**：[方向标记端点与批前邻居恢复](FUSED_ENDPOINT_SNAPSHOT_NEXT.md)已实现，14 项 Rust 测试、258 次完整 oracle 通过；同 binary 保留原/tagged 两阶段控制。[30 次小测及 20 次有限复核](ENDPOINT_BITMAP_REPORT.md)均保持精确，但 4 MiB n=2 未确认通用速度收益：融合 W1/W4 英文 1.245/0.549、中文 0.553/0.311 秒，自身扩展 2.27×/1.78×，四核结果波动大。临时起点数组消除成立，不默认切换。
- **密集 AA 不排序、不留 Plan Vec**：[位图方案](AA_DENSE_PARITY_NEXT.md)已实现并经交叉审查，13 项 Rust 测试、166 次 oracle 通过；只有 H≥ceil(N/16) 且容量守卫通过时建立一份 N 位 bitmap。unary 小测四核 HWM 4.71→3.56 MiB，速度方向仍分化；自然 EN 未启用 bitmap，不能将其同路径耗时波动算成收益。[word-cache 后续小测](batch_results/radical-aa-bitmap-cache-quick-v1/README.md)已将原子 OR 次数减少约 19–30 倍，四格 scatter 均下降，完整调用仍方向不一。
- **复用已存在的表**：[owner accumulator](OWNER_ACCUMULATOR_NEXT.md)已实现，12 项 Rust 测试及 240 次完整 oracle 通过。staged/fused-fresh/fused-reuse 三种同 binary 控制隔离连续提交与少一次汇总插入的效果；[12 次小测](batch_results/radical-reuse-accumulator-quick-v1/README.md)中 W4 英文/中文各跳过约 2.3 万/1.6 万入口，完整调用却无一致收益，暂不默认整合。
- **有条件组合**：[endpoint/bitmap 组合](experiments/radical/owned_endpoint_bitmap_combo/DESIGN.md)已通过 23 项 Rust 测试、485 次完整 oracle 和 16 次小测。AB 64 KiB 中非 AA 起点载荷归零、AA Plan 容量峰值从 524288 降至 65536 字节，两类节省可共存；组合 W1/W4 .00863/.00634 秒，仅 1.36×，n=1 不推为速度默认。EN/ZH 的 dense AA 为零，其 AA 开关是负控制。
- **新的并行调度**：[region 投影有序 posting](REGION_ORDERED_FUSION_NEXT.md)已通过 16 项 Rust 测试和 168 次完整 oracle；[8 次小测](batch_results/radical-region-fused-quick-v1/README.md)中 W4 英文 dynamic/region .03824/.04004 秒、中文 .02187/.02540 秒，没有净速度收益。region 自身 1→4 仅约 1.58×/1.19×。旧访问投影比值 3.69/2.59 包含 AA，而 AA 并不按同一 region 调度，不能直接称为实际阶段的扩展性上界；新微区实验单独记录非 AA 下界。保持独立原型，不组合为默认。
- **独占区域的端点访问**：[O(T) 边界快照](REGION_BOUNDARY_SNAPSHOT_REVIEW.md)已实现并通过 23 项 Rust 测试、171 次完整轨迹对照。每个 region 只读写自己的切片，远端读来自批前常数个 token 描述符，至多 2(T−1) 次越界端点写在 join 后回放；切点可穿过 token。AA 保留原路径，没有 N 大小的额外快照。[本轮报告](SNAPSHOT_PENDING_REPORT.md)与实验归档分别记录协议收益和计时边界。
- **直接汇总新键**：[pending owner Entry](PENDING_OWNER_ENTRY_NEXT.md)已实现并通过 14 项 Rust 测试、264 次完整轨迹对照。同 binary 保留 staged 和使用临时表的 direct-old 控制；Entry 不增宽，但删除临时表可能换来永久表的容量高水位。[本轮报告](SNAPSHOT_PENDING_REPORT.md)包含高阈值对照及 HashMap 公开 capacity 的测量纠正。
- **减少固定分区倾斜**：[微区](REGION_TASK_GRANULARITY_NEXT.md)已实现并通过 24 项 Rust 测试、92 次独立 oracle 和 8 组 k1 新旧对照。W4 k1→k4 改善了非 AA 访问均衡，却增加区间查询、局部聚合与调度成本；256 KiB n=2 英文 region .04315→.05442 秒、中文 .02156→.02537 秒，snapshot 也更慢。不继续盲扫任务数，见[三原型报告](MICRO_ATOMIC_ORDERED_REPORT.md)。
- **提前提交旧频率**：[producer 原子减频](PRODUCER_OLD_REDUCTION_REVIEW.md)已通过 17 项 Rust 测试和 85 次完整 oracle。它复用旧 route 记录作唯一退休标记，不增宽 Entry，owner 字典在共享减频阶段禁止结构变动。同窗 n=2 W4 owner→atomic 英文 .03322→.04130 秒、中文 .02394→.02248 秒，方向分化，不默认整合。
- **从出生来源维持全局有序**：[有序 posting](ORDERED_POSTING_REVIEW.md)已通过 25 项 Rust 测试、90 次完整 oracle 和非法模式检查。每个新 key 的出生由唯一规则方向产生，局部链片段反转后按 region 连接即可保持全局升序，无 N 大小副本。AA 初始排序确实被取消；自然语料 n=1 的微小净改善和 AB n=2 的交叠范围不足以确认速度优势。保留不变量作为后续切区和压缩基础，当前仍不切换默认。
- **工作量驱动的切点**：[按本批工作量选 cut](ADAPTIVE_ORDERED_CUTS_NEXT.md)已在 `owned_adaptive_cuts` 实现，28 项测试和 94 次独立完整轨迹门控通过。同 binary 小测 W4，非 AA 最大区域访问累计和 EN/ZH 降 2.25%/27.86%，完整调用与 HWM 没有一致收益。候选元数据有 8 MiB 选择守卫，但不是整次训练预算；不默认整合，详见[本轮报告](ADAPTIVE_REPLAY_REPORT.md)。
- **尚未实现的存储方向**：[u16 gap 压缩](ORDERED_POSTING_STORAGE_NEXT.md)保留完整 u32 位置域，但必须核算 checkpoint、对象头和转换峰值。[不可变 pair row](IMMUTABLE_PAIR_ROWS_REVIEW.md)暂因目录与集中更新成本保留在审查阶段；[磁盘 posting 分级存储](IMMUTABLE_POSTING_STORAGE_NEXT.md)仍未解决构建峰值及频率/heap 常驻。不能将这些设计当作已测得的提速。
- **删除出生链并分散填最终 posting**：[稳定语料重放出生位置](REPLAY_BIRTH_FILL_NEXT.md)已于 `owned_replay_birth` 实现，27 项 Rust 测试和 96 次 oracle 通过；raw `HEAD|fresh_id` 识别真实匹配，独占切片并行填充，物理 BirthNode 为零，额外累计历史扫描 O(N)。原 replay 的自然语料 W4 较慢、AB 较快，不能将节点去除等同于总峰值下降。
- **重放计数内联**：主 agent 依据上述成本追加 [`owned_replay_counts`](experiments/radical/owned_replay_counts/DESIGN.md)，30 项 Rust 测试及 96 次独立完整轨迹通过，同 binary chain/Vec replay/inline 的 26 调用小测。W4 计数 heap 峰值 EN/ZH 降约 65%/82%；inline EN 与 chain 相近、ZH 和 AB 较快，自身扩展仍仅 1.20×/1.51×。保留为 replay 的优先候选，不作为通用默认，见[归档](batch_results/radical-replay-counts-v1/README.md)。

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

先用小输入淘汰候选，正式复核只覆盖回答当前疑问所需的完整规模输入、核心模式和重复次数，提前计算总调用量，避免展开无必要的笛卡尔积矩阵。所有已发布数据保留 source/binary/fixture 哈希、完整轨迹校验、进程 CPU 与训练 VmHWM 口径。另一个 tokenizer benchmark 工作区始终只读。外存训练尚未实现。
