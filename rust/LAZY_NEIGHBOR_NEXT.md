# 首次查询时构建的邻接负证书

这是一个尚未实现、也没有速度或批宽数据的方案。目标是保留 `owned_neighbor_sketch` 的精确空间负证书，同时避免为每个 eligible pair Entry 常驻 8 字节摘要，以及在初始化和每个出生位置读取邻边。`owned_neighbor_sketch` 的 256 KiB 单次筛查把 EN 批次从 69 减到 60、ZH 从 87 减到 82；EN W1/W4 净调用时间变慢，ZH W4 单次变快。证书有实效，但出生即建摘要的全量工作尚不足以证明速度收益。

## 查询时的证书与时间方向

选批仍按精确 `(frequency, pair-key)` 顺序形成连续前缀，AA 仍单独一批。只有下一个候选与已选 pair 的 token 类型可能共享时，才对这些冲突 pair 查询摘要。若两 key 的出生轮不同，总使用**较新 key**的摘要检验较旧 key 的 hash bit；初始 key 之间、同一批出生的 key 之间任取固定一方即可。摘要从该 key 首次被查询时的**稳定当前语料**构建：扫描其唯一 owner 的整条历史 posting，`inspect` 过滤失效位置，再把每个活出现左右邻的 pair key 分别哈希到 64 位 OR mask。左右邻来自 `Plan.left_id/right_id` 与当前 pair `(a,b)`；sentinel 0 不形成邻边。没有活位置的 eligible key 表示索引/频率不变量错误，应返回错误，而非给出空 mask 许可。

同一 key 的位置只在其出生批次形成，此后只会失效。首次查询后的两条**存活旧边**不能在后来的替换中第一次共享 token：触及它们之间边界的合并必毁掉至少一条旧边，并使变化区的新边含新的 fresh ID。因此首次查询时记录的邻接，对未来任一较旧或同出生轮 key 都是保守上界；失效位置和 bit 碰撞只会让批次更短。对于查询时尚未出生的更新 key，必须改用那个更新 key 的摘要，不能反过来使用旧缓存。数值 `max(left_id,right_id)` 只用于区分出生轮；初始 key 的数值顺序和同批 fresh ID 的顺序不代表真实出生先后，但这些同龄 key 在任一首次查询时已处于同一稳定语料，固定选择其中一个仍安全。

缓存是唯一全局 `HashMap<u64,u64>`，key 是 pair，value 是 OR mask。先查缓存；冷 miss 完整扫描该 key 的 posting，然后一次性插入 mask，之后每次用同一 mask 做位测试。若较新 key 是本批已选规则，它的 Entry 虽已从 owner map 移出，仍在 `chosen` 内持有 posting，查询可借用该 posting；若是尚未选中的候选，则借用 owner Entry。任何建集都在选批的只读阶段完成，不与 apply 写端点重叠。已选 key 的缓存项在**整批选择/提交完成后**删除；该 key 的频率归零且其 pair key 不会复生。未选中的失败候选可保留缓存，即使随后变低频也只占有界标量空间。不能为节省空间任意 LRU 逐出仍可能被查询的 key：重建虽不会错，却会失去每 key 至多扫描一次的总工作界。

## 工作与内存界，以及关键路径

设输出规则数为 `R`、非空批次数为 `B`。每批至多有一个首先失败的冲突候选；其余发生摘要查询的 key 已在本批 `chosen`，总数至多 `R`。缓存绝不为任意邻居 key 建项。即使失败候选每批不同，首次构建的 distinct key 数也至多 `R+B≤2R`。一次建集扫描整个历史 posting，包括 stale。由于每个 pair key 的历史 posting 只出生一次，且每 key 最多建集一次，所有冷 miss 的位置访问总和不超过全体**实际保留的** posting 记录数，初始 eligible 边加后续已存出生边为 O(N)。实现可在调用结束核对 `mask_build_visits <= initial_eligible_postings + stored_born_postings`；被扫描 key 的全部记录在右侧只计一次。这个不等式依赖不 LRU 逐出，并把 selected key 的缓存退役放在本批结束。它约束**额外位置扫描**；哈希查询和至多 256 条规则的冲突位测试另计，哈希表期望每次 O(1)，并无对恶意碰撞的最坏常数保证。首个类型冲突若命中一条极长 posting，选择关键路径仍可能被一次 O(N) 扫描占满。

小列表可直接串行 OR。长列表可在现有 Rayon pool 上 `par_chunks` 验证并 OR-reduce，每 worker 只返回一个 `u64`，无需复制语料或 posting，也无逐位置临时数组；这种并行会在选批阶段额外调度任务和同步。阈值只决定调度，不影响证书。缓存裸 key/value 记录为 16 字节，HashMap bucket、控制字节、余量和可能的 rehash 双峰另计；按 distinct 查询 key 数，它的逻辑大小是 O(R)，而非按全部 eligible key 数。若 `R` 随 `N` 增长，此界也可达 O(N)。Rayon 分块调度使用已有 posting 切片，额外显式位置空间 O(1)；若实现另建任务描述符，应报告 O(H/chunk) 的当轮临时峰值。实际总内存仍由进程 VmHWM 验证。

相对出生即建摘要，这种首次查询 mask 只汇总**目前仍活**的出现和邻边，因此常可减少历史死邻接造成的阳性，但 64-bit 哈希碰撞与高邻居多样性仍可能使 mask 饱和。它不保证速度：出生构建的 O(N) 工作被搬到少数类型冲突的选择阶段，额外的 pool dispatch 可能比少掉的批次屏障更贵。若实现，同一二进制须对照 type、birth-neighbor64 与 lazy-query 模式，分别报冷/热命中、cold posting/stale 访问、最大单次扫描、批次数、W1/W4 总调用、CPU 时间和 HWM；先做完整规则/频率/final token 轨迹，再谈性能。

## 需要阻断的错误实现

- `Y X A B → Y X Z` 后，新 `(X,Z)` 与仍活的旧 `(Y,X)` 相邻。查询旧 mask 会漏证；首次查询必须扫描较新 `(X,Z)`，或使用它在出生时已经建立的摘要。
- 两个同批新 key 在端点 apply 未完成时扫描会读到中间邻接。首次查询只能发生在上轮 apply、owner fill、heap 更新都 join 后，且本轮任何写端点尚未开始。
- 一个被选 key 在选择过程中从 owner map 移出后，冷建集必须从 `chosen` 持有的 posting 扫；在本批后续候选检查前删除它的缓存会造成重复扫描。整批结束后再删不会漏证，因为该 pair 不会复生。
- `ABABABCD` 中 `AB` 后的新 `ZZ` 可压过 `CD`。摘要只允许在连续候选序列的当前类型冲突处证明安全；遇到阳性就停，不能跳过 `BA` 去找 `CD`。
- 对历史位置不经 `inspect` 就读邻居会把已失效边当活边。单纯多设 bit 可能只让证书保守，但若错误地借一个 stale 起点推断当前 pair 边界或在并发 apply 中读取，就不再满足稳定快照契约。权重不参与邻接 bit，但原有精确加权频率与 fresh-pair 祖先上界仍是整个批次正确性的必要条件。

`owned_probe` 的现有实现每次遇到类型冲突时只扫描当前候选，遇到冲突/预算可提前停止，候选以后可能被重复扫描；其扫描在调用线程执行。这里的冷 miss 扫完整个较新 key 的两侧邻域并缓存，供以后所有较旧 key 的冲突查询复用，长列表可并行。旧 native `parallel_pair_owned_spatial_extra` 也并行探测追加窗口，但没有此 per-key 邻域缓存；它的结果不构成本方案的速度证据。
