# 从有序 posting 选择每批 region cuts

状态：已实现于 `experiments/radical/owned_adaptive_cuts`，28 项 Rust 测试、94 次独立完整轨迹门控通过；同二进制 fixed/adaptive 各 W1/W4 已小测。实际样本类型为 `(u32,usize)`，本机 16 B；下文是原设计及证明，不代表原 8 B 估算已经实现。访问均衡改善不等于全调用加速，详见[本轮报告](ADAPTIVE_REPLAY_REPORT.md)。

本设计只处理**批边界的物理 cut 选择和 selected posting 区间定位**。前提是全部历史 posting 已由 [有序协议](ORDERED_POSTING_REVIEW.md) 保持严格升序；上一批的规划、apply、跨区写和 owner 提交须全部完成，下一批才可选择新的 cuts。cut 可以落在 live token 内部；region 对端点的安全访问、跨区写和 snapshot anchor 仍是各自独立的协议，本设计没有解决它们。首版取目标任务数 `T=min(W, corpus.len())`，与现有 region 上限一致；合法空语料 `corpus.len()==1` 或无选中 batch 时跳过 cut 计算。这样至多约 W 个 region，避免固定 `kW` 把 route header 和 owner 输入从 `O(W²)` 扩到 `O(kW²)`。若后来证据显示 W 个任务不足以平衡有效改写工作，再单独比较 `T=2W`。

设本批选中 `B≤256` 个 pair，历史 posting 长度分别为 `H_j`，总访问量 `H=ΣH_j`，最长表 `Hmax`。H 包含 stale 项；规划仍必须读取并验证它们。每个 cut 的每条 posting 范围可用已有 `partition_point` 定位，矩阵大小 `O(BT)`，查询时间约 `O(BT log Hmax)`，不复制位置索引。两个候选 cut 生成法如下。

1. **最长表分位。**对 `r=1..T−1`，从最长 posting 取 **1-based rank** `⌈rHmax/T⌉` 的位置 x，cut 放在 `x+1`；生成仅需 `O(T)` 次随机索引。相邻非重复 cuts 间最长表访问至多 `⌈Hmax/T⌉`，其他表可以全部落在同一区，故最大总访问仅能保证 `⌈Hmax/T⌉+(H−Hmax)`。只有 `H−Hmax≤H/(4T)` 之类的结构性 guard 成立时，它才给出约 `1.25H/T+1` 的上界。反例是两条各占 H/2、位于语料不相交两半的 posting：用其中一条划所有 cut，另一条的 H/2 次访问可能挤进一个 region。最长表若主要是 stale，按它分位仍能平衡其**扫描次数**，不能保证出生/哈希工作均衡。

2. **所有表的分层 rank 样本。**固定 `ε=1/4`，对第 j 条表取步长 `s_j=max(1,⌈H_j/(4T)⌉)`；在 rank `s_j,2s_j,…` 和末项处取样，每个样本带它代表的实际项数（末块可少于 `s_j`）。每条表至多 `4T` 个样本，总计 `M≤4BT`，只保存 `(position:u32, weight:u32)`，再按位置原地排序并合并相同坐标的权重。令 `F(x)` 为所有历史位置 `≤x` 的真实累计数，`A(x)` 为样本权重累计数。对任意 x，每条表尚未由样本覆盖的尾段少于 `s_j`，所以 `0≤F(x)−A(x)≤E=Σ(s_j−1)<H/(4T)`。对目标 `rH/T`，选使 `A(x)` 首次达到目标的样本坐标 x，并把 cut 放在 `x+1`，再去重。一个坐标在每条严格升序 posting 至多有一个样本，故这次跳跃 `J≤Σs_j=E+B`。于是 cut 处 `rH/T≤F(x)<rH/T+2E+B`；相邻非空区的历史访问至多 `H/T+2E+B <1.5H/T+B`。若连续目标落在同一坐标，去重后的前一 cut 视为它覆盖的**最大目标 rank**；下一不同 cut 的目标只多一档，因此同一单区上界仍成立，损失的是任务数。跨不同 pair 的**stale**历史位置可以同坐标，不能假设总表无重复；`B` 项正是这种不可按物理 cut 拆开的跳跃代价。采样构造 `O(BT)` 次位置读取与 `O(BT log(BT))` 排序，之后仍需 `O(BT log Hmax)` 范围定位。

严格要求“任何输入都不读取全部 H”与有保证的分布互相冲突：若 B 条 posting 各只有一个未知位置，则 H=B，连这些位置也不看便无法判断是否全落在一个 region。分层法只在 H 大于 `4BT` 时保证抽样读数小于 H；H 小时全部读取也被 `O(BT)` 的绝对元数据预算限制，且没有复制完整 posting。实际实现应在入门前给 `4BT` 与样本字节做 checked arithmetic、内存预算 guard；超出预算退回最长表或已有固定 cut，并报告没有相同的全表误差保证。`H=0`、重复候选 cuts、接近尾 sentinel 的 `x+1` 均需显式处理。

**推荐的最小选择。**先按最长表 guard 决定是否用其廉价分位，否则用分层样本。对所得 cuts 和原固定物理 cuts 各用 `partition_point` 算真实历史访问范围，选择最大 region 访问数较小的一组（同分选较少区/较低建表成本）。这样最终历史访问最大值不会比固定 cut 更坏，且任务数始终 `≤T`；上面的采样界只适用于未因预算 guard 回退的分层候选。范围矩阵可直接供 planner 使用，无需再定位一遍。此比较增加第二组 `O(BT)` 二分查询，但不扫 H。记录 cut 构造时间、样本数、range 二分次数、区域历史 visits、valid merges、route keys/birth nodes 和 region wall time，避免把元数据成本藏在 plan 时间内。

**不能由 H 推出 CPU 均衡。**某区的 H 项可几乎全 stale，另一区同样 H 项却几乎全有效；后者有额外 birth 链、hash 更新和写端点工作。selected pair 的加权频率也不是 per-region 计算量，无法在不读位置与当前语料的情况下可靠分解。确定性的亚线性位置采样对任意分布的 valid/stale 标记没有最坏有效工作保证；上界只针对历史 posting 访问。若实测 valid/route 开销主导，应另做采样验证或上一批统计驱动的成本模型，并给错误估计回退，不能把本设计的 `1.5H/T+B` 宣称为训练时间上界。
