# 将类型证书批次接到全局出生 posting

这份接口审查针对 `experiments/radical/v3`：保留它的单份不可变 posting arena、eligible-only span、动态 Rayon chunk 和出生位置的 count–prefix–scatter；批次先用现有 token 类型证书，AA 继续单独处理。此处只推导接口，尚无合并实现或性能结论。

## 批前选择与一次快照规划

协调端从 v3 的精确堆按 `(frequency 降序, key 升序)` 取连续前缀，沿用 `heads/tails` 类型证书；首次 AA 若在前沿则单独执行，后续 AA 截断当前前缀。因本批每个旧 pair 的全部有效出现互不重叠，选中规则的频率、顺序和 fresh ID 与串行一致。对未选而弹出的第一个候选，必须放回堆；只有前缀里的 `Entry` 从 scalar map 移除，旧 posting span 可留在 arena 等本批读完。批次上限仍受 `remaining_rules` 和原证书 cap 限制。

先为规则分配连续 new ID 并完成长度溢出检查。按**规则 rank、该规则 posting offset** 枚举 `FlatTask { rank, span }`；每个任务读取冻结的语料和一个只读 `selected_key -> new_id` 小表，验证其 span 内全部历史位置。非 AA 的有效位置天然按语料位置递增，无需全局合并 Plan。任务只保存 `Vec<u32>` 起点；规则的 `a/b`、长度、new ID 由 rank 查表。规划时已经从快照算好旧边扣减与最终出生边，放进本任务的 delta 与 born 缓冲。全部规划任务完成后，才有任何 corpus 写入。`FlatTask` 应保存 arena offset 范围，不能跨 `arena.resize` 持有旧 slice；posting 追加须等规划读取结束。

邻接处理可直接用现有 `parallel_sharded::prepare_batch` 的规则：对于有效 `AB`，左邻 `L` 若本身是另一选中匹配的右 token，则左边界由那条左匹配拥有，本计划不再扣旧边或生成新边；右邻 `R` 若是另一选中匹配的左 token，则本计划把最终右边界生成为 `(Zi,Zj)`。判断只需在稳定快照上查看 `L` 的前驱与 `R` 的后继，并查 selected key 表。它同时覆盖同规则的相邻 `ABAB` 与不同规则的相邻匹配；旧边 `(B,R)` 只由左计划扣一次。AA 不套用此逻辑，而沿 v3 的全局 parity 路径单独执行。

所有旧 key 在本批只能下降，所有含本批 fresh ID 的新 key 只能增加；新 key 不可能是批前候选。先归约旧边扣减和新 key 加权出生频率，做阈值、溢出和 span 大小检查。然后可并行写各任务的 Plan 起点：重算 `right=pos+a_len`、`after=right+b_len`，只写 `pos/right/after-1`。证书保证不同有效匹配的 token 区间不交，所以这些端点写入也不交。待写入和 posting scatter 都结束，再发布下一轮 heap 视图。

## 新 key 的唯一生产规则与位置顺序

设本批规则 `i` 生成新 token `Zi`，当前快照里所有 token ID 都小于首个 `Zi`。一个出生 key 若只有右端是新 ID，形如 `(L,Zi)`，只能由规则 `i` 的左边界生成；只有左端是新 ID，形如 `(Zi,R)`，只能由规则 `i` 的右边界生成；两端都是新 ID 的 `(Zi,Zj)` 只能由**左匹配**的规则 `i` 在其右边界生成。右匹配的左边界被抑制。即使 `i=j`，生产规则仍唯一。旧规则不能再生成该 key，也不能在本批消耗它。

固定 key 因而只在一个规则的 posting chunks 中出现。`(L,Zi)` 的 anchor 是该规则匹配的前驱 token 起点 `before`，随匹配 `pos` 严格递增：两个不重叠匹配 `p<q` 之间，`q` 的前驱 token 必在 `p` 的前驱之后。`(Zi,R)` 与 `(Zi,Zj)` 的 anchor 都是左匹配 `pos`，自然递增。过滤 stale、抑制被选中的左边界只会删元素，不改顺序。同一 key 在每个 FlatTask 的 born 缓冲中已有序；FlatTask 又按该规则的 posting offset 排列。因此 v3 的 count–prefix–scatter 按 FlatTask 下标分配该 key 的子区间，就能生成全局有序的冻结 posting span，**无需跨规则按位置排序**。

这里有两个必须保持的前提。FlatTask 不得按完成时间收集结果，须用 Rayon indexed collect 或显式 task 下标恢复 rank 与 offset 顺序；每个共享边只由左匹配产出，否则 `(Zi,Zj)` 会重复。若以后加入实际空间证书，仍须确认它交给规划器的是所有有效出现互不重叠的完整连续前缀；本版先不接探测。

## 最小改动面与测量判据

保持 v3 的 `Entry`、arena、初始 count–prefix–scatter 和 birth scatter。把单规则循环中的候选弹出改成返回 `Vec<BatchRule>` 的类型证书选择器；把 `postings.par_chunks(...)` 改成一次 `FlatTask` indexed Rayon 规划。任务输出 `{ valid_starts: Vec<u32>, old_delta, born }`。其后一次归约/前缀分配，再一次并行端点写入；born count/scatter 继续使用 FlatTask 顺序。AA 仍调现有单规则路径。首版不要在 `FlatTask` 间按语料位置排序，也不要重引入全局 Plan flatten。

对照应量每批宽度分布、Rayon 派发次数、plan/write/birth count/prefix/scatter、旧 key delta 归约、posting visits、峰值 Plan/born/cursor 容量和 RSS。若批宽多数仍为 1，批次接口不会解决 v3 的单步同步成本；若宽度显著大于 1，合并派发才可能抵消额外的候选选择和多规则元数据。完整规则轨迹、频率、fresh ID、替换数和 final token 须先对串行 oracle 一致。
