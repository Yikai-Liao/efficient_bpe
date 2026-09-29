# 非 AA 的独占 region 边界快照

状态：独立 Rust 原型，源码尚待统一模型测试、完整规则轨迹、Clippy、release 和计时门控。此 crate 克隆冻结的 `owned_region_fused`；`--endpoint-plan tagged-fused --region-mode dynamic|region|snapshot` 分别执行原动态块、原子 region、边界快照 region，默认 `dynamic`。整数 hash、heap、exact batch 证书、唯一 key posting、AA 全局 parity、owner commit 与最终 token 语义保持相同。完整条件与反例见 [快照审查](../../../REGION_BOUNDARY_SNAPSHOT_REVIEW.md)。

`region` 和 `snapshot` 都取严格非空的 `T=min(W,corpus.len())` 个物理区间；W 仍为线程数和 key-owner 数，`region_count_effective` 记录 T。W>N 时减少逻辑 region，不改变 W 个 owner 桶。只有 tagged ID 满足 31 位域时才走 region；超域整调用回退原 `two-pass/dynamic`，并输出 requested/effective 模式。两种 region 模式使用相同 cuts、初始化和按 region 投影排列的唯一 posting；额外差别仅为非 AA 的语料访问协议。AA 仍先完全稳定规划，再沿旧原子写回，随后刷新 cut anchor。因此本原型不能称为全 trainer 已无原子操作。

初始 Prepared 单位 token 由共同验证契约保证；每条内部 cut 初始 anchor 就是 cut 坐标。批前快照对每条 cut 保留覆盖 token 以及前两个、后三个 live token 的 `(head,id,len)`，遇 piece sentinel 即止。它保存的是自有数值，不借用语料。长 token 即使跨多个 cut，也只占每条 cut 常数个描述符。区内任务对本区 `&mut [AtomicU32]` 调用 `get_mut()` 做普通 u32 读写；跨区旧 head/tail/sentinel 查询只可命中本区 lower/upper 出境 cut 的六 token 数值窗口，失配返回错误，不退化为远程共享语料读。旧邻居仍由 fresh HEAD/裸尾解码和 selected-key 表还原；单个 region 只有一位写者，故本区一次邻居读取间没有竞争，远端则固定批前值。原 zero→Acquire 重读分支在这一路径里不使用。

匹配起点 p 在本区，head 写本地执行；越界右起点与尾端写只入 `{pos,value}` 队列。所有区域任务 join 后，协调者回放这些写，再刷新 anchor，最后 owner commit。每条内部 cut 在批前至多被一个选中匹配跨度跨越，故队列最多 `2(T−1)` 条，运行时检查。跨区左 birth 仍沿原 region 修复路径注入目标 region 输出。任务报错时整个消耗式训练调用退出，不能把部分更新后的私有语料用于下一批。实际可变借用由 `split_at_mut` 产生互不重叠的 region slice；任务阶段没有完整 corpus 的不可变共享借用，回放与刷新都在这些借用结束之后。

批后 anchor 若仍是 HEAD，旧 head 仍覆盖 cut，可能已是新 ID；若变成 0 或裸尾，旧覆盖 token 作为右 constituent 被吞，直接使用批前窗口保存的前驱 head。sentinel anchor 不变。然后从新 head 沿端点和 token 长度各走常数步，重建六 token 描述符；不按 cut 向前扫描任意物理距离。AA 在其全部写入完成后也执行此刷新，供下一批非 AA 使用。在**批前稳定状态**，历史 posting 的 p 若仍是旧 A，则它仍是 live A 起点；批中单靠这一格不够，因为跨区写可能尚未回放。当前选中非 AA 的类型冲突证书另行排除这个 A/B 被其他匹配吞掉，因此延迟写不会令该规则误验。这个证书不适用于 AA 重叠候选，所以 AA 不进入独占融合路径。

`snapshot_build_seconds` 包初始窗口建立，`snapshot_refresh_seconds` 包每批 join 后重建，`snapshot_deferred_apply_seconds` 包协调端回放，不应和 `plan_seconds` 直接当独立阶段相加。计数器 `snapshot_local_reads/writes` 只计非 AA 的普通本地格访问，`snapshot_boundary_queries` 只计远端窗口请求，`snapshot_deferred_stores` 只计非 AA 越界写。AA 的原子读写不在这些计数中。计数、窗口查找、坐标判断和延迟回放也是新路径的真实成本，计时结果只能解释为**整个访问协议的消融**，不能把差额纯归因于 Atomic 指令。`snapshot_peak_descriptor_capacity_bytes` 是窗口 Vec 的实际 capacity 乘静态元素大小，不含分配器开销；`snapshot_peak_deferred_capacity_upper` 将 worker 队列容量与归并队列容量相加，是可能共存的保守上界，`snapshot_peak_deferred_len` 是实际条数。真实训练峰值仍用 `train_vm_hwm_mib`。每批仍支付约 `2BT` 次 posting 二分、`O(TW)` owner 路由头和 `O(T)` cut 窗口；没有 N 大小的额外快照或每 worker 索引副本。

私有模型测试以六个邻居 token 的 1/2 长度做 64 种组合并遍历各物理 cut，核对 head/tail 查询；另覆盖 257/513 长 token 跨多个 cut、sentinel、延迟右写与 anchor 更新。相邻 AB/CD 位于不同 region 时，模型分别先执行 AB 和 CD，调用真实快照 inspect/write 并比较旧邻居、最终邻居、待写端点和批后 anchor。trainer 定向比较原子 region 与快照 region 的完整串行参考轨迹，包括 AA 后的非 AA、W>N、跨区出生、权重及 31 位域回退。通过这些门控之前不解释性能。
