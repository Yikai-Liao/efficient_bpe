# 全局有序的出生 posting

状态：独立原型，尚待 Cargo、完整轨迹和性能门控。此 crate 从冻结的 `owned_region_tasks` 克隆。`--posting-order region|global` 默认 `region`，保留原有仅按固定 region 投影有序的控制路径。`global` 只允许 `--endpoint-plan tagged-fused --region-mode region|snapshot`；`--regions-per-worker k` 仍取固定 `T=min(kW,N)`，W 个 key owner 与原版相同。31 位 tagged ID 域不足时整调用回退 `two-pass/dynamic`，`posting_order_effective=region`，不借全局有序假设。

全局顺序由出生一次性质得到。每个新 pair 含本批唯一 fresh ID `Zi`，因此 `(L,Zi)` 只可能由规则 i 的左邻生成，`(Zi,R)` 和 `(Zi,Zj)` 只可能由规则 i 的右邻生成；旧邻居不可能已经是本批 fresh ID。固定规则的有效匹配起点 p 按位置升序遍历，右出生位置就是 p。左出生位置是 p 的旧前驱 live head：同一 piece 中后一个 p 的前驱不可能退到前一个 p 之前，跨 piece 的 sentinel 不产生边。因此任意固定新 key 在一个 region 内的 `route_birth` 调用位置升序。AA 的有序 posting、全局 LTR parity 与按 region regroup 维持这个顺序；相邻 `Zi,Zi` 只由前一匹配的右出生生成，后一匹配左出生被抑制。

现有 8 字节出生节点用 head 插入，所以每个 `(region,key)` 链遍历顺序恰好相反。`global` 在 owner 填充此链后只反转**刚追加的 segment**，再按 indexed region 输出顺序拼接。不会反转整条 posting，也不做比较排序，不新增按 N 的位置副本。额外工作是一遍被保留出生位置的线性反转；`ordered_birth_reversal_segments/positions` 记录它。初始位置由物理 region 顺序扫描且区内升序；每个 key 只在初始或自身出生批次生成位置，此后记录只会失效，不会追加。因此所有保留 posting 都严格全局位置升序，即使含 stale 历史记录。

跨区左出生注入目标 region 后仍位于该 region 的最后。若某个目标 region 的旧 live head q 的下一 live head p 已在后续 region，q 必是本区最后的 live head；同一目标 region 无法有第二条跨出活边。长 token 可以跨多条 cut，但不会增加这一区的出境活边数量；piece sentinel 会切断相邻边。debug 路径检查每轮跨区目标唯一、q 所在 region 正确、q 的后继已越过 region 上界，以及每个新 Entry posting 严格递增。release 不做这类全量扫描。

全局有序模式选择 AA 时直接用原 posting，跳过 `par_sort_unstable`，记录 `aa_sort_elided_batches/positions`。其余 AA 有效性检查、parity、路由和 apply 不变。`aa_sort_seconds=0` 仅表示该排序已省去，不代表 AA 全阶段零成本。`region` 控制路径仍排序，仍按旧链反序存储。固定 cuts 不移动；尽管全局有序 posting 将允许后续在任意 cut 上二分，动态 cut 需要另行解决选择、快照 anchor 重建与开销，不能视为本次实现。

此原型按 same-binary 两种 posting 顺序比较完整规则 `(pair,frequency,order)` 和 final tokens。单测覆盖 AA 后非 AA、同批相邻 fresh/fresh 边、跨多 cut 长 token、多 piece 与不等权重、k1/k4、snapshot 延迟写和域回退；统一 executor 再做独立 oracle、strict Clippy、release 与轻量计时。只用这次的 full-call 数据判断排序省时是否超过逐段反转成本。
