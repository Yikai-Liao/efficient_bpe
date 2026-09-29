# Pending owner Entry 独立审查

结论：在 owner 独占、完整归约后才发布 heap、所有错误终止本次训练的条件下，
临时 count 放入空 inline `SmallPosting` 可以保持内存安全和 BPE 精确性。
它删除的是**新键汇总用的临时 HashMap**，并不删除新键逐 producer 的哈希查找、
唯一键的末次遍历、birth 链填充或 touched Vec。没有渐近时间收益；永久 owner
表被大量未过阈值的新键扩容，是明确的失败条件。

## 成立所需的状态与次序

本批新键含 `fresh_start` 之后的 ID，不可能已在批前 owner 中；先前批次的
fresh ID 已小于本批 `fresh_start`，因此属于旧键。每个 owner 独占其表，
`entry` 首次插入时才把完整 key 放进 touched Vec。临时 Entry 的 frequency
从零用 `checked_add(u64)` 累计；空 inline posting 的第一个 u32 槽用
`checked_add(u32)` 累计物理 occurrences。归约结束，逐 touched key 读取
最终 count；未达 minimum 的删除，合格的先记下 `(key,count)`，再替换为
`SmallPosting::with_capacity(count)`，之后才允许 birth fill 和 heap 发布。
每条 producer 链仍须校验其局部 count、索引界及非终止 `next` 严格下降；
每个合格 posting 的最终长度须等于 owner 中累计的总 count。
[复用表原型](experiments/radical/owned_reuse_accumulator/src/lib.rs)展示了为什么
总 count 与某个 producer 的局部链头不能混为一个局部 count。

现有 [`SmallPosting`](experiments/radical/owned_reuse_accumulator/src/small_posting.rs)
的默认状态是 `len=0, capacity=0`，两个 inline u32 均已初始化。修改其中
一个槽而仍保持这两个元数据为零，空切片不会读取槽值，Drop 也不会将其当成
heap 指针。若检查或分配失败，整个训练返回错误并 drop owner；尚未转换的
inline 临时项、已转换的空 heap 项和部分填充的 posting 各按本身真实状态
析构。转换应先取出 count，再用新 `SmallPosting` 整体替换旧值，不能在临时
count 上调用 `push`。这种复用**不是类型系统编码的状态**：普通 `push` 无法
区分临时空 inline 与真正空 posting，会覆盖 count。因此私有受检 API 和
严格阶段调用点是语义正确性的必要条件；`as_slice`/Drop 安全本身不能保证
不会错误发布空 posting。

旧键在本批只下降。[已有 direct-old](experiments/radical/owned_direct/DESIGN.md)
可对它们逐 producer 直接扣减，并在 eager heap 模式下每个仍合格的旧键
只发布一次最终 candidate；它仍用临时 map 汇总**新键**。这项提案的新意
仅是新键借永久 owner Entry 汇总。比较应让两个模式都启用同一种旧键路径；
否则会把旧键绕过 combined 的收益错误归给 pending Entry。新键虽然暂驻
owner 表，在完整频率、posting 和 heap 都准备好之前不得参与下一次选择。

## 工作和空间的边界

设 D 为本批所有 producer 的新键 route 记录数，B 为不同新键数，E 为
达到 minimum 的新键数。现有 direct-old+combined 对 D 条记录访问临时
map，再遍历 B 个结果，最终只把 E 个插入 owner。pending 方案对 D 条
记录访问 owner，再用 O(B) touched Vec 完成筛选、分配和 heap 发布。两者
都是期望 O(D+B+births)，哈希碰撞最坏情况仍不受保证。pending 可省去
临时 map 的分配和 E 个新键从临时表移入 owner 的部分操作，但查找目标
可能是更大的长期 owner 表；且 touched 至少占 O(B) 空间，不是零开销。

提交阶段的反例条件是：owner 当前 capacity 较小，本批某个高频规则在
许多不同的右邻 `Cᵢ` 前匹配，生成 B 个各出现一次的 `(Z,Cᵢ)`，
minimum=2。pending 方案向永久表插入这 B 个不合格新键，再全部删除；
HashMap 删除不收缩 capacity，后续即使活跃 key 很少也保留高水位，
而 combined 方案可释放临时表。**这不是已构造出的完整训练反例**：
直接取 M 个词 `[A,B,Cᵢ]` 且各 `Cᵢ` 初始就不同，会让现有
`initial_index` 在阈值过滤前先把 Θ(M) 个原始 pair 插进 owner，
owner capacity 本来就可能达到 Θ(M)。必须用晚期出生的邻居类型或
可控初始索引容量构造可达语料，才可实证新增的长期污染。
`Entry` 还包含 frequency 与
16 字节 `SmallPosting`，比临时 `Delta` 更大，另有 touched `(key,count)`
容量。故 E≪B、长期 K≪B 时，pending 的瞬时与后续内存都可能更差；
E≈B 且 owner 已有足够容量时才更可能同时省时省空间。永久容量高水位
是 `max_epoch(K+B)` 量级，而不是把每批 B 无限相加，但它可远高于
当前活跃 key 数。每轮 shrink/rebuild 会引入新的 O(K+B) 工作与变量，
不能算入这项最小实验的收益。

建议只在同 binary、同 direct-old、同 hash/heap/planner 的控制下测试，
记录本批暂存 B、合格 E、owner 插入前后 capacity、touched Vec capacity、
训练 HWM 与完整调用时间。若高阈值 singleton 语料导致长期 owner 容量
明显超过 combined 控制，即使少了临时表，也应判定此路线不适合通用默认。
