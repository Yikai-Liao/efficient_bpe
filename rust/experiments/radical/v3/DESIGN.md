# v3: 出生位置的并行 count–prefix–scatter

v3 是独立 crate，承接 v2 的 eligible-only 全局 posting arena 与 chunk 内 Plan。它去掉了每轮 `(key,pos)` 比较排序和协调线程逐个 append 出生位置。所有 Plan 在稳定快照上按原语料位置升序，所选匹配互不重叠；新 ID `Z` 之前不存在，所以一个新 key 只能属于 `(L,Z)`、`(Z,R)`、`(Z,Z)` 三类之一。固定 `(L,Z)` 的 posting anchor 是 `before`，随 Plan 位置递增；另两类的 anchor 都是 Plan 的 `pos`，也递增。AA 的 run parity 只删去一些 Plan，不改顺序。故每个 chunk 对同一个新 key 的出生记录已按位置升序，chunk 又按语料顺序排列。

初始化也采用相同的 count–prefix–scatter。语料按物理位置分块，边由其左端所在块拥有，跨块边照常计入；各 worker 第一遍在自己的块中按 key 聚合 occurrence 数与带权频次。协调者只处理各块的 distinct key 汇总，计算全局频次、eligible key 的 span 和每个 `(key,block)` 的独占子区间。worker 第二遍重扫自己的块，直接将位置写进 arena 子区间。物理块顺序就是 pair span 的位置顺序。`initial_count/aggregate/prefix/fill_seconds` 与 `initial_local_keys/capacity` 显示并行扫描和残留的 O(D) 中央聚合。内存是每块局部 key 表之和，不是每 worker 复制整个 posting 索引；极端高多样性输入仍可能使局部元数据接近边数。

应用阶段保留每个 chunk 的局部 `Changes {delta,born}`，不把 born 列表复制到协调线程的全局数组。worker 并行统计各自 `born` 的 `key -> count`；协调线程仅按 `(key,chunk)` 计数记录确定每个 eligible key 的总长度、全局 arena span 和每个 chunk 的独占子区间。这一步 O(D)，D 为局部 distinct key 数之和，不是逐出生位置扫描。arena 一次 `resize` 后，worker 并行第二遍扫描各自 `born`，使用本 chunk 的 cursor 将位置写入已分配子区间。一个 key 的 chunk 子区间按 chunk ID 顺序排列，因此整个冻结 span 仍升序。

写入用原生 `Vec<u32>` 的 raw pointer。安全条件是所有区间在扫描前由 prefix 唯一分配且两两不交、`resize` 后每个目标元素已初始化、每个 cursor 仅由对应 chunk 的 worker 修改、每次写都检查 cursor 在子区间与 arena 边界内、所有 worker 返回后才读取 posting。缺一项不能声称并行写安全。元数据用 `peak_birth_count_entries`、`peak_birth_cursor_entries/capacity` 报告；这些哈希表和 `Vec<u32>` 扩容都会进入实际 RSS。`birth_count/prefix/scatter_seconds` 单列。

频次归并仍在协调线程，按每 chunk 的局部 distinct delta key 更新标量表；它不是本版的加速成果。与 v2 串行归并一样，它只处理频次净变化。稳定 Plan 直接生成最终邻边，理论上本轮的新 key 只增不减，出生记录加权和等于该 key 的净频次；历史 stale 位置来自旧轮次的冻结 span，不影响本轮新 key 的计数。完整规则和最终 token 仍须同直接串行对照，在发布性能结论前先跑测试/quick trace。
