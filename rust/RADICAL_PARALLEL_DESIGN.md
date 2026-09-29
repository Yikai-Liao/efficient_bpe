# 全局出生 posting 原型：按位置块并行，索引只存一份

这是与 `parallel_pair_owned` / pipeline 主线独立的结构消融。代码在 `experiments/radical/`，没有注册进主 crate 的 ablation dispatch。它仍训练同一份 `Prepared` 连续语料和同一条精确贪心 rule 序列；用父 crate 的 `validate_prepared` 检查完整输入契约。输出的 `Rule(left,right,frequency)`、fresh ID 顺序和最终 token 序列必须与串行参照逐项相等。

## 核心不变量

每一轮新增 ID `Z`。这一轮出生的任何新邻边都包含 `Z`，而先前的旧 pair 不可能因此增加出现次数。同一 pair 只能在初始状态或它较新的端点出生的那一轮产生全部出现位置。以后它的出现只能失效。这允许把每个 pair 的历史位置写成一个**冻结的连续 span**，不用每个 worker 各有 `pair -> Vec<u32>`，也不用在后续轮次给旧 key 追加位置。

初始化做两遍顺序扫描：第一遍计算每个 key 的加权重叠频次和 occurrence 数，按 key 分配全局 `u32` posting arena 的区间；第二遍按文本位置填充，故每个 span 内位置天然升序。出生边先收集 `(key,pos)`，由 Rayon 并行排序，再按 key 追加一次。选中 pair 的 span 可以分为任意大小的连续任务块，worker 只借用同一份 arena。`chunk_size` 是可调参数；固定 pool 在训练全程复用。

非自配对 `AB` 的所有实际匹配互不重叠。worker 先只读稳定的 `AtomicU32` 端点语料，过滤历史 stale posting，生成 `Plan(pos,right,after,before,left_id,right_id,weight)`；全部 Plan 完成后才并行写。写入阶段只读 Plan 和 token ID，不再读语料，因此不会把不同线程刚写入的状态误认为批前状态。升序 Plan 中，若前一匹配的 `right` 等于当前匹配的 `before`，中间旧边只由前一匹配扣一次；若下一匹配的 `pos` 等于当前的 `after`，前一匹配生成最终 `(Z,Z)`。每个匹配的端点写集合不交叠。

`AA` 的重叠 occurrence 也在同一个有序 span。先并行过滤有效位置，每个 chunk 报告 run 首尾与末尾奇偶；协调者只处理 chunk 摘要，再并行按左到右奇偶选择不重叠匹配。实现复用 `experiments/aa_parity.rs`。选择后与非自配对共用稳定 Plan 与并行写路径。piece 边界的零哨兵形成大于 token 长度的物理间隔，不会把不同 piece 的 AA run 拼在一起。加权频次在任何替换前由初始邻边累积，AA 的重叠边都计数；应用时只选左到右不重叠子集。词表新长度用 `u32` 检查加法，所有位置与 posting 都用 `u32`，没有 `u8` 长度限制。

## 频次与提交

worker 对边的消失/出生生成局部 `key -> i128 delta` 和新边位置。已经选中的 pair 直接退役；其他旧 pair 只减频，不回升。协调线程当前归并局部 delta、更新标量频次和 heap，然后把出生边分组追加到 arena。低于阈值的出生 pair 仅保留标量，不存 posting；这只是首版简化频次更新的做法，因为它以后不会再被选中，后续可丢弃标量并忽略它的负 delta。新边排序已并行，但 **delta 归并与 arena 追加仍串行**；它们可能成为下一阶段的关键路径。优化它们需要实测证据，不能把这部分时间算作 worker 并行成果。

## 空间与工作量的实边界

设初始相邻边数为 `E0`，实际成功替换数为 `M`。每次替换至多生成两个邻边记录，故总生成记录 `<=2M`，而 `M<=` 初始非零 token 数。这说明历史 posting **长度**上界为约 `E0+2M<=3N` 个 `u32`。这是记录数量上界，**不是峰值 RSS 界**：`Vec` capacity 可能扩张，哈希表、堆、Plan、delta、排序临时记录、Rayon 栈、验证缓冲和把输入转换成 Atomic corpus 时的瞬时旧/新数组均占空间。arena 现在 append-only，不回收已退役 pair 的 span，也没有压缩 stale 位置。自然文本低频 pair 多时，这个策略可能比当前主线更慢、更费内存。

CLI 分别报告 `posting_arena_len/capacity`、仍有 Entry 的 `retained_entry_posting_len`、达到阈值且尚未选中的 `eligible_posting_len`、最终实际活邻边 `final_live_edges`、本轮总生成边记录和 peak Plan/新边/delta。`retained_entry_posting_len` 含 stale 位置，不可称为活边数。初始化两遍、验证、线程池、并行规划、并行应用、串行频次归并、并行出生排序和串行追加各有计时。对照仍必须看包括验证与初始化的完整 `call_seconds`、峰值 RSS、同实现 1→4 核、以及相对最佳直接串行的绝对用时。

## 运行与当前验证

```bash
/root/.cargo/bin/cargo test --manifest-path rust/experiments/radical/Cargo.toml --lib
/root/.cargo/bin/cargo run --release --manifest-path rust/experiments/radical/Cargo.toml -- \
  --input prepared.json --workers 4 --chunk-size 4096 --rules 3000 --min-frequency 2 \
  --trace radical-trace.json
```

独立 crate 的单元测试覆盖 AA 长 run、自配对重叠、相邻 `ABAB`、同频、带权 piece，以及 100 组随机带权输入在 1/2/4 worker 和 1/5/32 chunk 下与父 crate 串行规则及最终 token 的逐项对照。AA parity 模块还穷举短 run、gap、空 chunk、长 token 和接近 `u32::MAX` 的位置。2026-09-30，并行 AA 版本的短编译和单元测试通过；随后增加共同输入验证及细分计时的冻结版本尚待统一执行者重新构建和核对。fixture 性能与峰值 RSS 也尚未完成，当前不预报加速比。

原型当前有明显实验风险：每条 rule 都有一次全局最高候选决策，Rayon 两轮任务派发与 Plan flatten 的常数在窄 posting 上可能淹没收益；初始 two-pass index 在超多 key 输入上受单线程哈希与内存带宽限制；出生排序虽并行，归并/追加是串行；历史 arena 无回收会推高大语料 RSS。性能实验应逐项报告这些阶段，不能只拿更慢的单 worker 原型当作加速基线。

后续两个独立 crate 保留 v1 作可还原对照：[v2](experiments/radical/v2/DESIGN.md) 只存可入选 posting 并取消协调线程的 Plan flatten；[v3](experiments/radical/v3/DESIGN.md) 将初始填充和每轮出生位置构建改为按 `(key,chunk)` 的 count–prefix–scatter，位置扫描与写入并行，协调者仍处理 distinct key 元数据。v2/v3 在完成完整轨迹对照前均是待验证实验，不与 v1 成绩混写。
