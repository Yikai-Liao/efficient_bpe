# v2: 只保留可入选 posting，取消全局 Plan flatten

这份 crate 是 `radical-birth-v1` 的独立副本，v1 字节不再改写。v2 第一组改动是两项可归因的内存/协调工作消除：

1. 旧 pair 频率只能下降。初始化计数后，低于 `min_frequency` 的 key 不分配 posting span；出生 pair 的净频率低于阈值也不入索引；已入索引的 key 降至阈值以下即退役。worker 仍报告所有边的局部 delta；owner/协调端对不存在的旧 key 负 delta 直接忽略。这保持规则选择不变。
2. 验证后的 Plan 继续留在每个 posting chunk 的 `Vec<Plan>`，不在协调线程复制成全局 `Vec<Plan>`。两个 O(G) 摘要遍历记录各 chunk 左侧最近非空 chunk 的最后 `right` 与右侧最近非空 chunk 的首 `pos`（G 是 chunk 数，空 chunk 可跨越）。worker 应用时在 chunk 内检查相邻 Plan，边界只看摘要，因此相邻 `ABAB` 或 AA run 仍只扣一次旧边，并生成最终 `Z/Z`。`chunk_summary_seconds` 单列，避免把残留的串行 O(G) 隐藏。

这两项尚未解决 v1 的串行初始计数、局部 delta 归并与出生 append。下一步结构如下，目标是在不复制全语料/全索引到每个 worker 的前提下转移这些工作：

| 阶段 | 所有权与并行任务 | 同步/空间代价 |
|---|---|---|
| 初始计数 | 文本连续块各自扫描其左端所属边；每块暂存 `key -> (occurrence count, weighted count)`，按 key hash 将局部汇总发给 owner | 暂存条目数至多初始边数，通常约 `workers × 热 key 数`；不是每 worker 一份完整索引 |
| span 前缀 | 每个 owner 合并该 key 各文本块的计数，判断是否达到阈值，计算 key 全局起点与每块独占子区间 | 每个 `(key,block)` 一条 cursor 元数据；跨块顺序由块号保证 |
| 初始填充 | 文本块再扫一次，只向自己已分配的 `(key,block)` 子区间写 `u32` 位置 | 子区间两两不交，避免每 occurrence 原子增量；使用 `Vec<AtomicU32>` 或有证明的 disjoint raw pointer 写 |
| 每轮规划/应用 | winner 的单一冻结 span 分块给 Rayon worker，Plan 留在 chunk，频次变化暂存 chunk 局部 | 一份 posting arena；热 pair 的位置工作不会集中到持有该 pair 的线程 |
| 频次归约/候选 | 每个 chunk 的 delta 按 key owner 发送；owner 并行更新自己的标量表与候选堆，协调者只归并 owner 的最高候选 | 路由与每 owner 字典可能仍消耗大量 CPU；需报消息/键数和 CPU:wall |
| 出生 posting | 新边按 owner 分桶；owner 各自按 `(key,pos)` 并行排序并追加到自己的 arena，key 的 span 当轮冻结 | 物理上是每 owner 一个 arena，但每条位置仍恰好存一次；独立 Vec 的 capacity 浪费需计入 |

另一种完全单 arena 方案是 owner 算完各 key 长度后进行一次全局 prefix，再让 worker 用独占子区间填充。它保持一个 `Vec<u32>`，但每轮全局 append 需要所有 owner 的长度和地址分配屏障；如果出生记录很少，屏障可能比串行 append 更贵。实现顺序应先测 v2 当前两项的实际收益，再决定 owner arena 与单 arena 哪个更值得。

正确性归纳依赖三个边界：初始块以**左端位置**拥有边，跨块边不漏；所有填充子区间按文本位置块号排列，所以 AA posting 仍全局有序；每轮 owner 更新完成后才取下一条全局候选，以 `(frequency desc, pair asc)` 保持 tie 与 fresh ID 顺序。worker 写 corpus 仍须在 Plan 全部完成后进行，AtomicU32 不能替代快照屏障。
