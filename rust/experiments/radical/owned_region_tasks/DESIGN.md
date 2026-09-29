# 固定微区任务与唯一 pair owner

状态：独立原型，尚待统一 Cargo、完整轨迹和性能门控。此 crate 从冻结的 `owned_region_snapshot` 克隆，只改变 region 数量与任务粒度；`--endpoint-plan tagged-fused --region-mode region|snapshot --regions-per-worker k` 设置正整数 `k`，默认 1。`dynamic` 只接受 `k=1`，避免把未生效的设置当成实验参数。`W` 始终是 Rayon 线程数和唯一 pair owner 数，`T=min(kW,N)` 是固定物理 region 数，`N` 是包含末尾 sentinel 的 corpus 长度。乘法用 `checked_mul`；`W>N` 时取 `T=N`，每个区仍非空。若 ID 域超过 tagged 表示上限，整个调用回退 `two-pass/dynamic`，输出 requested/effective 模式与有效 `T=0`。

初始扫描按固定物理 region 产生 `T` 个有序输出。每个 pair 仍只归一个 owner，owner 按 region 序连接出生链，因此每个 posting 的 region 投影非递减；区内位置顺序无需保证。批次准备对每个被选 pair 用两个 `partition_point` 找本区 posting 段，Rayon indexed collect 保留固定 region 序。AA 全局排序和奇偶选择不变，选中的 Plan 按位置归入目标 region 后才路由。左出生锚点落到另一 region 时先暂存例外，任务 join 后注入目标 region；每条 cut 最多有一条活边跨越，例外条数运行时以 `T-1` 检查。快照模式同样以 `T-1` 条 cut 建窗口，非 AA 跨区端点写先入队，全部任务 join 后回放，运行时检查至多 `2(T-1)` 条。AA 仍沿原稳定规划及原子写回。cuts 全程不移动，因为 posting 不具备对新 cut 二分所需的顺序。

增大 T 可降低最重任务的访问量，但也增加每批约 `2BT` 次二分、`T` 个结果、`O(TW)` 空路由头、`O(T)` 快照窗口，以及潜在更多局部 delta 条目。`region_count_effective` 给出实际 T。`region_peak_route_header_capacity_bytes` 是 region 输出路由 Vec 的 capacity 乘 `Route` 静态大小，未计 HashMap/Vec 内部分配和 allocator overhead；`region_peak_route_delta_capacity`、`region_peak_route_born_capacity` 另报内部容量。AA 临时重分组的双缓冲上界、快照窗口和延迟队列仍沿用原指标，真实进程峰值看 `train_vm_hwm_mib`。

`region_posting_visits`、`region_valid_merges` 包含 AA 与非 AA；`region_max_*_per_batch` 和 `region_sum_max_*` 也包含 AA 的物理 region 投影。AA 的排序、验证与奇偶选择先沿原 posting chunk 执行，因此这些 AA 投影值并非完整 AA 阶段的调度界。`region_non_aa_posting_visits`、`region_non_aa_valid_merges` 只记真正独占 region 任务的工作量。`region_sum_visit_makespan_lower_bound`、`region_sum_merge_makespan_lower_bound` **仅在非 AA 批次**累加 `max(ceil(total/W), max_task)`，可与前两项非 AA 总量对照。该式仍把每次记录视为等成本，忽略 route、窃取、同步和内存访问；不是实测线程用时或全调用速度上界。固定 region 是任务划分，不表示绑核或 NUMA 亲和。

定向单测比较 k=1/4 的完整规则、频率及最终 token，包括相邻不同规则、非均匀权重、AA 后非 AA、长度超过 255 的 token 跨多 cut、跨区左出生和延迟右写、W>N、空语料、HEAD 域回退以及无效 k。统一 executor 再以同预算比较本 crate 的 k=1/4、冻结的 `owned_region_snapshot` k=1 和串行参考；在这些门控完成前不从本原型推断性能收益。
