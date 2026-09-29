# 端点融合与 AA 位图组合审查

审查对象：独立 `owned_endpoint_bitmap_combo`；初次冻结 lib SHA 为
`d3b62dfc3dd7d1e283dbf15092b517feea791658781cc28e7d6f55e2c8fe8ba0`。
原子 corpus、唯一 owner posting 和 exact batch certificate 沿用融合原型。
本文件是代码与不变量审查，不替代 executor 的编译、oracle 或性能结果。

组合中最容易破坏下一轮的不是 bitmap 本身，而是 AA 写回。如果 AA 沿用
裸 u32 写者，后续 fused 解码会把新 token 首端点当成尾端点。当前版本将
普通 Plan apply 与 bitmap apply 都导向 `write_merge_at::<TAGGED>`，
tagged 时保持 head→右起点→尾端点的 Release 发布顺序。非 AA 的 Acquire
恢复规则没有因此改变。

AA scatter 和 route 均用 `inspect::<TAGGED>` 遮掉标记位；只有先前 posting
证实的起点能进入 bitmap。scatter join 后检查有效数与 popcount；route
join 后才写 corpus；apply join 后才提交 owner。AA 不与 fused 非 AA
并发执行，因此没有把原先需要稳定快照的 AA 邻居读放进并发改写期间。
bitmap 的 Relaxed OR 由这些阶段 join 提供可见性，不承担跨阶段通知职责。

AA 的 global parity、piece 权重、checked 溢出和重复起点错误保留。
同一长度内 p−L / p+2L 的相邻已选判断依赖完整有效位集和全局奇偶，不能
换成仅查询 selected pair key。ID 高位域检查仍发生在训练入口；超域整次
调用回普通 u32 两阶段，bitmap 仍可用，长度仍为 u32。

新 AA order 分支每个 AA 批次选择一次，TAGGED 为 const generic，没有
在每次 corpus 读取中新增运行时模式匹配。位图只在密度和容量守卫通过时
创建一份；实际 capacity 检查发生在分配之后，依然不构成瞬时 RSS 硬上限。
已有 SmallPosting 的私有 unsafe 所有权没有改动，组合未新加 unsafe。

机制性成功应表现为同一次训练中：非 AA 临时起点计数为零、早期密集 AA
使用 bitmap 而无 Plan Vec，之后稀疏 AA 可正常回退。回退仍产生 Plan，
所以不能说混合算法全程零临时计划。组合的 `peak_plan_len` 表示实际
materialized Plan，和较早位图原型某些逻辑计数口径不同；跨 crate 比较
优先用明确的 Plan capacity 字节及完整进程 HWM。

该审查未发现阻塞组合正确性的差异。是否更快、是否降低完整进程峰值，
只由实际启用路径的同 binary 控制判断，不将两个独立原型的最好数字相乘。
