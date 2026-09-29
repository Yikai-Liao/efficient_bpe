# v4：冻结 posting 上的一次快照精确批次

v4 从 v2 独立分支，保留单份全局出生 posting arena、eligible-only 索引和较低常数的 `(key,pos)` 出生排序后端。它新增与现有 `parallel_pair_owned` 相同的**连续候选前缀证书**，但位置从一个冻结 span 按 `FlatTask {rule_rank,start,end}` 动态分块读取。v1–v3 的源文件不改。

每轮在稳定频次 heap 上按频次降序、pair 字典序升序弹候选；新候选的左端若曾是已选 pair 的右端，或右端若曾是已选 pair 的左端，就在它之前停止并放回。AA 必须单轮走全局有序 posting 的 run parity 路径。上限为剩余规则数与 256。每个候选的重叠频次、rule 顺序、fresh ID 和长度在快照阶段固定；已选旧 key 的 Entry 从索引退役，但其 span 在本批规划完前仍可读。

非 AA 批次按规则 rank、该规则 posting offset 顺序建立 indexed `FlatTask`。所有 worker **先读同一份稳定 corpus**：验证历史位置，生成有效起点，计算旧边负 delta 与本批最终出生边。若左邻 token 本身是另一选中匹配的右 token，该共享边由左匹配的右侧独占，当前计划抑制左边；若右邻 token 是另一选中匹配的左 token，当前计划直接生成 `(Zi,Zj)`。这覆盖同规则 `ABAB` 与不同规则 `ABCD` 上的相邻替换。判断通过稳定端点语料读邻 token ID，并查本批 `selected_key -> new_id` 小表。AA 单轮 parity 仍按左到右选非重叠匹配。

**所有任务规划完毕**后才并行写端点。非 AA 的每个任务只存有效 `u32` 起点，写时由规则长度重算 `right/after`；批内所选出现互不重叠，故写集合互不相交。随后协调者按 key 归并局部 delta、更新标量频次/heap，汇总本批出生记录并行按 `(key,pos)` 排序、一次追加各 key 的冻结 span；每条新边都含至少一个本批 fresh ID。下一批直到本批频次和 posting 发布完成才选择 winner。

这是真批量：一次候选选择、一次稳定规划、一次并行端点写、一次频次归并和一次出生索引生成覆盖多个 rule。CLI 报 `batch_rounds/batch_rules/max_batch_width/singleton_rounds/flat_tasks/planned_positions` 及各阶段时间，和 v2 对照时应看完整 call、RSS、同实现 W1→W4、最佳直接串行绝对时间。批宽、任务数和出生数组容量会决定同步收益；即使批宽大，中央 delta 与排序仍可能限制扩展。AA 权重用重叠 occurrence 选规则，实际替换按跨 chunk run parity 左到右执行。所有长度/位置为 u32，长度加法检查溢出；没有 u8 token 长度约束。

独立 `v4` 测试覆盖空输入、带权 piece、AA 跨 chunk、长度超过 255、同频、相邻不同规则 `AB/CD` 真批次和随机加权小样本；任何性能结论前还须完成同 fixture 全规则轨迹与最终 token 对照。初版没有调入 v3 的 count–prefix–scatter，以便把批次效果与出生构建成本分开。

`combine_seconds` 单列任务局部 delta/born 的中央汇总。非 AA 批次的 `apply_seconds` 只含端点写，AA 路径的 `apply_seconds` 也含其内部 combine；因此这些分段计时不能机械相加当成 call 时间。FlatTask 的下一个分界用 `start + min(chunk_size,end-start)`，包括 `chunk_size=usize::MAX` 也不发生无符号溢出。
