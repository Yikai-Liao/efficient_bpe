# 按首次查询建立邻接证书

本 crate 从冻结的 `owned_neighbor_sketch` 拷贝，仅改变类型冲突时取得空间负证书的时机。CLI `--batch-certificate type|birth-neighbor64|lazy-neighbor64` 在同二进制中运行三种内核；默认 `type`。`type` 与 `lazy-neighbor64` 都实例化无摘要字段的 `Entry<0>`，`birth-neighbor64` 实例化 `Entry<1>`。`--mask-parallel-threshold` 默认 8192 个历史位置；设置为 `usize::MAX` 可强制串行冷建集，W1 总是串行。其余唯一 owner、heap、posting、精确批次规划、AA 左到右 parity、加权频率与 fresh ID 协议保持相同。

首次遇到可能重叠的 pair 类型时，只检查当前候选与已选规则中类型确有冲突的 key。对每对 key 取出生较新的那一个；初始或同批 key 的数值年龄只是稳定选择一方，二者都已经在同一个当前语料中。一个全局 `HashMap<u64,u64>` 以 pair key 缓存其 64 位邻域 bit 集。缓存 miss 扫该 key 的完整历史 posting，用 `inspect` 丢弃 stale 起点，对每个当前活出现把紧邻左边 `(left,a)`、右边 `(b,right)` 的 key 哈希 OR；零 mask 是合法缓存值。若较新 key 已被选中，它的 Entry 在本批 `chosen` 中，仍可借 posting；未选候选从 owner Entry 借用。所有扫描发生在本批端点写入前的稳定快照，绝不复制位置列表。查询命中只做 hash lookup 和位测试。较新 mask 缺少较旧 key 的 bit 才接纳当前候选；任何命中或 AA 均按现有规则停批。

一个 pair key 的历史位置只在初始或出生批产生，之后只会失效；两条都存活的旧 pair 边不会在以后的 merge 中首次共享 token。因此首次查询时的邻域 mask 对其后与更旧或同批出生的 key 的比较始终保守。更新的 key 必须查询自身 mask。selected key 的缓存直到整批提交完才退役；其 key 被删除且永不复生。失败候选的缓存跨批保留，不 LRU 逐出；若它以后被选中才退役。缓存只为已选或本批第一个失败候选建项， distinct build 数 ≤`batch_rules+batch_rounds`。每 key 至多扫描一次完整历史 posting，故 `lazy_mask_visits <= initial_eligible_postings+stored_born_postings`。训练返回前检查这两条界，违反即报错。它们是额外扫描的总量界，并不保证单次冷 miss 的低延迟。

当 posting 长度达到阈值且 W>1，使用现有 Rayon pool 的 `par_chunks(4096)` 逐块 `inspect`、OR-reduce。每块只返回 `(mask, valid_count)`，不分配逐位置临时数组，也不复制语料或 posting；额外调度屏障位于选批关键路径。小列表直接串行 fold。W1 两条配置走同一串行路径。`lazy_mask_seconds` 包含 miss 扫描、Rayon dispatch 和插入缓存，不是纯 CPU 工作。type/birth 两模式不建立懒缓存，birth 模式仍保持原来出生时建 Entry mask 的语义。

指标报告 distinct builds、缓存 hits、扫描位置/失效位置、并行 builds、最大单次列表长度、缓存峰值/最终 key 数及 `16×HashMap::capacity` 代理字节。`capacity` 是可容纳 entry 数，不是实际 bucket 数；代理不含控制字节、allocator 或 rehash 双峰。实际训练内存看 `train_vm_hwm_mib`。候选数上界为 `R+B≤2R`，所以缓存逻辑大小 O(R)，但若目标 merge 数 R 随 N 增长它仍可达 O(N)。初始化和 born fill 在懒模式不做额外邻居读取；相对 birth 模式可能节省全量工作，也可能因冷 miss、额外选择屏障和重复活边验证而变慢。完整性能判定须同时看 type/birth/lazy 的轨迹、批次数、自身 W1/W4 总调用和绝对最佳串行。

定向测试包含被选 Entry 移出 owner 后的零 mask、跨两次查询复用与退役、stale 过滤、串行/并行 OR 一致、新旧与同批 fresh 邻接、`ABABABCD` 的新 `ZZ` 抢占、AA、长 token、非均匀权重，以及随机完整轨迹。`rust/LAZY_NEIGHBOR_NEXT.md` 保留独立反例审查与设计推导；此前 `owned_probe` 的逐候选串行扫描和旧 native extra-only 并行窗口是不同机制，不能视为本原型已有速度证据。
