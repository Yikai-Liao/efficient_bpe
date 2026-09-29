# 融合端点快照原型

本独立 crate 从冻结的 `owned_integer_hash` 复制，保留其 greedy 连续精确批次、
全局唯一 owner posting、动态 FlatTask、owner 路由、heap、AA parity、`std|ahash`
选择及完整 JSON/trace 协议。唯一实验变量由 `--endpoint-plan` 指定：

- `two-pass`（默认）：原 u32 端点、plan 后全批 apply，作为同 binary 控制。
- `tagged-two-pass`：标记端点并使用下述 Release 写顺序，但仍保留原 plan→apply
  屏障与有效起点数组，隔离表示和内存序成本。
- `tagged-fused`：仅非 AA 把一个有效 occurrence 的规划、route、端点发布放进
  同一个 worker；AA 仍使用排序、全局 run parity 和 tagged 两阶段路径。

`train` 按 hasher×端点模式一次分派到 `<H,const TAGGED,const FUSED>` 核，
热路径没有每位置 enum 判断。原 `SmallPosting` 源文件保持逐字相同。所有模式
仍用原有 `AtomicU32` corpus，不增加 N 大小的 side array，也不复制永久索引。

## 表示与域

`HEAD = 1<<31`，低 31 位保存 token ID。初始非零单 token 加 HEAD；合并
token 的首位置写 `new_id|HEAD`，末位置写裸 `new_id`，被吞的右 token 起点在
长度大于一时清零。稳定读取先 mask 掉 HEAD。`initial_lengths.len() +
max_merges <= 2^31` 由 checked 算术在入口判断，保证最大可能新 ID 小于 HEAD；
不满足时整次调用走原 u32 `two-pass`，输出 requested/effective 模式与
`endpoint_domain_fallback`。原输入校验仍在 timed call 内，长度继续使用 u32。
若 max_merges=0，初始 ID 最大可为 `2^31-1`，这个边界也满足公式。

标记模式按顺序对非 AA occurrence 执行 Release store：先 head p，再右起点
q（长度一时裸新 ID，否则零），最后仅在右 token 长度大于一时写末端
t-1 的裸新 ID。AA 也使用同样的 tagged writer，但所有读取/route 在写入
前已经结束。两阶段和融合模式每批都 join 所有 producer，才交给 owner 更新、
最终解码或下一批选择。完整语料里的每个当前 token ID 被 stable reader mask；
融合专用邻居读保留 raw HEAD 位以恢复旧邻居。

## 融合证书

非 AA 的批次仍是已证明精确的前缀：选中 pair 的任何两个实际匹配不共享
批前 token。验证 p 和 q=p+len[a] 以 Acquire 读取去标记 ID，必须仍是旧 a/b。
任何本批写只产生零或 fresh ID，均不能写回旧 a/b；稳定历史 posting 一旦因
合并失效，其 p 或 q 改变且不可复生。因此通过验证的 occurrence 在自己
写入前不会被其他合法匹配失效。

邻居可能已被相邻 occurrence 部分改写。`old_end_id` 仅在**已知批前 token
末端**使用：读到旧 ID 直接返回；读到本批 fresh HEAD 返回对应规则的旧左
constituent（其长度须为一）；fresh 裸尾返回旧右 constituent。自己的
`p-1` 给出左邻旧 L，据此算 L 起点，再在 `before-1` 得旧 K，
`selected[(K,L)]` 判断左侧 occurrence 是否已被本批选中并抑制重复 birth。

自己的 after=t 若为 fresh HEAD，直接由规则表取得旧右邻 C 和最终新 ID。
若仍是旧 C，读取 u=t+len[C]：非零旧 D 或 fresh HEAD/裸尾可由专用
`old_next_id` 恢复旧 D，再用 selected[(C,D)] 决定最终右邻；若 u 为零，
Acquire 重读 t。零可为 C 后的原始 piece 边界（最终右邻仍是 C），也可为
选中 `(C,D)` 清掉长度>1 的 D 起点，此时重读必得 fresh HEAD。
该推论要求 head Release store 先于 clear Release store，读 clear 的 Acquire
与其同步，再读 t 的 Acquire 受 happens-before 与同一原子位置的 coherence
约束。t 只有一个批内写者。详细反例审查及顺序一致交错模型见
[根级审查](../../../FUSED_ENDPOINT_SNAPSHOT_REVIEW.md)。

worker 先为该 occurrence 完成所有频率扣减、出生 key 和本地链记录，
才发布自己的端点。`selected` 和按 `fresh_begin` 索引的 batch 规则元数据
在启动 worker 前已冻结。fused worker 不分配有效起点 Vec，不向
`WorkerOutput.results` 添加元素，也不建立按 FlatTask 排序的 apply Vec。
选中 posting 必须保留到所有 worker join；错误时整个消耗式训练调用
返回 Err，不再使用已经部分改写的私有 corpus。owner commit 仍只在 join
之后进行，debug birth key 校验看到完整最终 corpus。

## 指标与局限

`fused_non_aa_batches/merges` 计真正走融合路径的工作；
`decoder_zero_rereads` 计右邻 u 为零后的 t 重读，包含正常 piece 边界；
`non_aa_start_positions_peak` 和 `non_aa_start_bytes_peak_proxy` 估算旧
非 AA plan 留存的有效起点载荷，后者只按 4 字节/位置，不声称等于 Vec
容量或 RSS；fused 对此为零。现有 `plan_seconds` 在融合模式包含该阶段的
端点写入，非 AA `apply_seconds` 为零；不可直接把两者相加与其他模式的
单阶段比较。没有每 occurrence 时钟。

更多 Acquire/Release、mask 与邻居解码会增加读取成本；Release 写和
逐 occurrence store 也可能降低批量 apply 的吞吐。因此此原型只验证
是否能减少完整调用时间及 HWM，不预设收益。需先通过完整规则、频率、
最终 token oracle，再在同 fixture/affinity/build 的 `two-pass`、
`tagged-two-pass`、`tagged-fused` 之间比较。
