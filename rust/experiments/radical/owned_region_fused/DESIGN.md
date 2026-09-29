# Region 投影的精确 fused 原型

状态：独立原型，尚待统一 Cargo、完整轨迹 oracle 和计时验证。代码从冻结的 `owned_fused_endpoint` 克隆。`--endpoint-plan tagged-fused --region-mode dynamic|region` 在同一 binary 内对照，默认 `dynamic`；`region` 对其他 endpoint plan 显式拒绝。整数 hash、heap 策略、精确批次证书、key 唯一 owner、`SmallPosting`、tagged Acquire/Release 端点协议保持原样。若 fresh ID 超出 31 位 tagged 域，整次调用回退 `two-pass`，`effective_region_mode=dynamic`，不能把结果计作 region 路径。具体证明见 [总体设计](../../../REGION_ORDERED_FUSION_NEXT.md) 和 [端点审查](../../../FUSED_ENDPOINT_SNAPSHOT_REVIEW.md)。

`region` 把完整物理语料等分成 W 个**逻辑**半开区间，W 是请求的 worker 数；Rayon 可以让任意线程执行任一区间，不提供 CPU 或 NUMA 亲和。W 大于物理位置数时允许重复 cut 和空区间。每个 key 仍只有一个 posting，它仅按 region 编号非降排列，region 内位置可以无序。初始化由固定 region 扫描，再让 owner 按 region 编号接入位置；新 key 仅在出生批次追加，同样按 region 编号遍历输出链。旧 key 以后只失效或退休，因而原 region 投影不会改变。已知 cut 的 `partition_point(pos<cut)` 能切出某 key 的 region 片段，但不能用它查询 region 内任意位置。debug 构建在初始化和每个 eligible 新生 posting 填充后检查完整 region 顺序；`first<=end` 只检查两个二分结果，不是有序性证明。

非 AA 批次每个 region 一项任务，按已选 rule 次序对 posting 做两次边界二分，直接访问自己的历史片段。原 `inspect_fused` 从共享 tagged corpus 还原批前邻居，先形成旧负 delta、最终新 key 和出生位置，再写当前匹配的不相交端点。选中 pair 的任意两个实际匹配不共享批前 token；一个 stale posting 不会重新变活。每个任务持有 W 个 owner 路由桶，但**不**复制全局索引或语料。所有 region join 后才进入 owner 频率归约和出生填充。

右出生的起点就是本匹配起点，留在当前 region。左出生的旧 token 起点可能跨越一个或多个 cut；任务先保存 `{target_region,key,pos,weight}`，join 后将其注入目标 region 输出，再开始 owner commit。一个 cut 在批前连续活 token 序列里最多被一条跨区邻接穿过，故例外数每批不超过 W−1，运行时检查此界。sentinel 会删边，不创造额外跨区边；长 token 可以跨多 cut，但仅产生一个左出生例外。旧负 delta 不带位置顺序要求，仍由原 region 路由。

AA 仍先按物理位置排序历史 posting，过滤 stale，按全局从左到右 run parity 选取实际匹配。原动态 Plan chunks 按起点迁移到 region Plan Vec；这些 Vec 全局按 region 排列、区内按位置排列，所以沿用原 AA 的跨块首尾摘要和相邻选中匹配去重。AA 的右出生按匹配起点归属，跨区左出生走相同的例外修复。AA 依然有旧 `valid_chunks`、原 Plan chunks 和新 region Plan Vec 的临时内存；这条路径尚未采用 bitmap AA。重分组是 O(M) 串行 Plan 搬移，会构成额外关键路径；不能把 AA 也称作完全融合执行。

`region_posting_visits` 与 `region_valid_merges` 是所有 region 批次的总和；`region_sum_max_visits` 和 `region_sum_max_merges` 累加每批最繁忙 region 的工作量，可分别用总量除以它们估计静态划分的负载上界，但历史记录和实际匹配并非同一工作成本。`region_max_visits_per_batch` 和 `region_max_merges_per_batch` 则记录全程单批峰值。非 AA `region_partition_worker_seconds` 是 worker 内二分耗时之和，不是 wall time；AA 的二分在协调端，计入 `plan_seconds`。`region_partition_searches` 同时计非 AA 和 AA 搜索。`region_aa_regroup_capacity_upper_bytes` 是搬移前后两组 Plan Vec capacity 字节数之和的批次最大值，仅为可能共存容量的保守上界，不含 `valid_chunks`、Vec 头、分配器开销或其他暂存；真实峰值看 `train_vm_hwm_mib`。`region_peak_route_delta_capacity` 与 `region_peak_route_born_capacity` 是当批所有 region×owner 路由的 capacity 总和再取最大，不是桶的精确字节数。`region_cross_births` 计注入例外条数。

额外确定性工作是每批约 `2BW` 次二分，最坏 `O(BW log Hmax)`；每个 region 一套 W 个路由头，临时头部 `O(W²)`。静态 region 的热度若偏斜，或二分、AA 搬移与碎片化高于局部性收益，完整调用可能变慢。此实验仅验证 region 归属和弱排序是否带来实际收益，不声称 false sharing 已被实测确认。
