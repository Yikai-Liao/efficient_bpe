# 冻结 posting arena 的退役 span 压实（未实现）

v1/v2/v3 的 posting arena 只追加，选中或跌破阈值的 pair 退役时，原 span 变成不可达历史数据。压实可在 rule 已提交、出生 scatter 完毕、没有 worker 持有 arena slice/raw pointer 的屏障处执行。它不改变规则选择或频次，只搬移仍在索引中的有序位置。

令 `A=arena.len()`，`L=Σ Entry.len`，`D=A−L`。`D` 只数**整个 span 已退役**的记录；仍在 Entry span 中但位置已 stale 的记录必须计入 `L`，否则不能按整段搬移。v2/v3 在一个完整 epoch 后只保留达到阈值且尚未选中的 key，每个 Entry 均有非空 span，因此 `K=#Entry≤L`。v1 保留过低阈值的零长度标量 Entry，不满足这一前提，须先清理或过滤，不能直接套用下面的哈希表摊销界。

当 `A>0` 且 `D≥L` 时：

1. 从 Entry 表形成 `(key,old_start,len)` 描述符；对长度做 checked prefix，分配新 arena 中互不重叠的 `[new_start,new_start+len)`。key 顺序任意，**每个 span 内的位置顺序保持不变**，所以 AA posting 的全局位置顺序保持。
2. 申请长度 `L` 的新 arena。worker 按描述符并行把旧 span 拷到独占新区间。读旧 arena 与写新区间全程分离，直到所有 worker 返回才更新 Entry.start、交换 arena 和释放旧分配。
3. 同次重建/收缩 Entry 哈希表，使容量回到 `O(K)`。只重建 arena 而留下历史峰值哈希槽，内存收益可能很小。堆中的过时候选仍可依现有 lazy 校验，但它的 capacity/旧条目也应在需要时重建并计量。

**摊销工作界。** 上次压实后所有逻辑 posting 记录数为 `L₀`；至本次新增 `B`，现有 arena 长度 `A=L₀+B=L+D`。本次复制的 `L≤D`，每个被退役的逻辑记录只退役一次，可把这次 copy 逐个收费给自上次压实以来的退役记录。因此所有压实的 payload copy 总量 `O(初始存储记录+累计出生存储记录)`，后者至多约 `3N` 个 `u32`。由于 v2/v3 有 `K≤L`，描述符/prefix 的本次 `O(K)` 也可收费给本次 `D`。

哈希表容量也需要单独保持不变量：每次 GC 将表重建为 `O(K)` 容量，两个 GC 之间只按正常增长扩容。期间最多有上次 `K₀≤L₀` 个 key 加新增 key 数；每个新增 eligible key 至少有一条出生 posting，故容量 `O(K₀+B)≤O(L₀+B)=O(L+D)=O(D)`，触发时 `D≥L`。这样本次遍历、重哈希旧表容量的工作也可摊销到退役记录。若实现保留空 Entry、强制超额 reserve、或堆条目无限累积，该界就不适用。

**峰值空间不是这个摊销时间界。** GC 同时持有旧 arena 的 **capacity**、新 arena `L`、旧 Entry 表、新 Entry 表、描述符、prefix 缓冲，以及可能存在的 Plan/delta。旧 Vec capacity 可能显著大于 `A`，哈希表重建时两份表共存；因此触发条件 `D≥L` 不能保证峰值 RSS 小于 `1.5×` 或任何固定的用户预算。需实测 `GC次数、copy字节/秒、old/new arena capacity、old/new map capacity、GC VmHWM`，并与不压实版同 fixture 比较。

这项方案只清理退役 span，不能去掉**仍有 Entry 的 span 内 stale 位置**。若 stale 占主导，需另做按有效位置过滤/重建的机制与工作界。把 cold posting 写盘也不等于得到外存训练器：winner 选择、跨轮增量、I/O bytes 和缓存命中仍需完整设计与计量。
