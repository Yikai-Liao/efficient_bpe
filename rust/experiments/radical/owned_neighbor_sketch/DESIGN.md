# 出生时邻接摘要的精确批次证书

此 crate 从冻结的 `owned_grouped_inline` 复制而来。CLI 的 `--batch-certificate type|birth-neighbor64` 在同一二进制中选择完整训练内核；默认 `type`。两种内核分别实例化 `Entry<0>` 与 `Entry<1>`，所以控制组的 Entry 没有额外摘要字段。训练仍使用唯一 key owner 的频率、heap 和 posting，AA 独占一批，所有规则顺序、加权频率和 final tokens 应与串行贪心一致。

当前类型证书遇到可能重叠的 pair 类型便停批。例如 `(A,B)` 与 `(B,C)` 可能在 `ABC` 上共享 `B`，但若语料只有分开的 `AB` 与 `BC`，它们其实能同批执行。摘要模式为每个 eligible key 存 64 位邻居 bit 集：在该 key 的每个**出生**出现处，把紧邻的左、右 pair key 分别哈希为一个 bit 并 OR。初始索引在原始单位长度语料上建集；此后新 key 都含本批 fresh ID，owner 只在 apply 已结束、按 grouped birth chain 填充 eligible posting 时读取最终语料建集。低频过滤前的初始构建会读取尚未确定是否 eligible 的位置；出生构建只读取保留的新 key 位置。

任一 pair key 的出现只在初始状态或其 fresh ID 批次产生，此后只会失效。若两个当前有效 pair 边共享 token，则它们初次成为邻接时，较晚出生的边必能看到较早的边；合并不会让两条均未改变的旧边突然共享 token。新旧 key 按 token ID 的最大值比较出生先后；两条初始 key 或同批新 key 都在同一个稳定快照中建集，任选其中一方也安全。类型冲突时只要较新 key 的 bit 集缺少较旧 key 的 bit，就证明这两个 key 当前没有重叠出现。候选与每条类型冲突的已选规则都需拿到此负证书，才可接纳。任何 bit 命中、摘要缺失或 AA 均按原证书停批。死邻接及哈希碰撞只能造成保守停批。

位置读取依赖端点表示：`corpus[pos-1]` 保存前一个 token 的 ID，右邻起点是 `pos+length[a]+length[b]`。初始单位长度边只需读 `pos-1` 与 `pos+2`。出生边在 owner 填充时读左右邻；私有 `born_neighbor_bits` 的 unsafe 契约要求 grouped birth 记录确为 apply 完成后当前 pair 的活边，token ID 可索引长度表，右端不越过最终 sentinel，且并发端点写入已全部结束。唯一调用点由稳定快照有效 Plan 经 `route_birth` 产生，apply join 后才执行。debug 构建仍检查 `(a,b)` 端点和边界；release 不为摘要重复解码每个出生 pair。owner 填充完整条 new-key posting 后才返回，下一轮选择自然在这一屏障之后。旧摘要不随旧位置失效而减 bit；它不会漏证。

每个 retained Entry 多 8 字节，`(u64,Entry)` 的结构大小从控制组的 32 字节变成 40 字节；真实 HashMap bucket 与分配器成本需以进程指标另报。额外读取次数至多初始物理边数加被保留的出生位置数，即 O(N)；位哈希常数不免费。选择时对每条实际类型冲突候选最多检查已选规则数，批宽上限 256，最坏 O(256R) 次位测试。指标分别记录初始/出生读取、冲突数、负证书扩批、保守停批、选中/最终摘要 popcount 和最终摘要 payload。`sketch_peak_capacity_scaled_bytes=8×ΣHashMap::capacity` 只是按可容纳 entry 数缩放的代理值，不等于实际 bucket 字节，可能低估 bucket 载荷与 rehash 重叠；`train_vm_hwm_mib` 给整个训练调用的进程高水位。

同一位置的哈希 bit 可能很快填满 64 位，尤其高频 pair 的邻居种类多时。若大多数类型冲突仍为 bit 阳性，构建读取与 Entry 容量只增加成本。本实验因此先要求完整 oracle 轨迹、定向空间分离/新旧方向/碰撞/AA/长 token/权重测试，再以同一二进制比较批次数、总调用、W1 与 W4。批次变宽而调用变慢也应保留为负结果。

`rust/src/ablation/parallel_spatial.rs` 的 `train_extra` 已实现有界并行探测追加候选，旧结果见 `rust/batch_results/pair-owned-spatial-extra-v1/README.md`。它是另一条已测算法，不属于本 crate 的新增摘要机制。
