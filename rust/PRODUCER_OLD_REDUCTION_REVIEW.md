# 在 producer 完成本地聚合后直接提交旧频率

已实现于 [owned_atomic_old](experiments/radical/owned_atomic_old/DESIGN.md)，
通过 17 项 Rust 库测试及 [85 次完整轨迹对照](batch_results/radical-atomic-old-gate-v1/README.md)。
当前精确批次已经固定选中的连续规则前缀，随后规划只依赖稳定
语料、规则和 selected 表，不再依赖其他旧 key 的实时频率。旧 key 的出现
本批只会消失，新 key 必含本批 fresh ID。这允许把旧键减频从 owner 提交
阶段移到各 producer 规划结束时，而不改变 greedy 选择语义。

## 可变状态的边界

同一 binary 保留 `old-reduce owner|producer-atomic`。两种模式使用同样的
`Entry { frequency: AtomicU64, positions: SmallPosting }`；owner 控制在独占
阶段通过 get_mut 普通减频。第一版只让 lazy heap 启用 producer-atomic，
eager 请求整调用回 owner 并报告 effective 模式，避免混入另一套堆发布协议。

先移走 selected Entry 并固定本批规则，然后把所有 owner 字典仅共享借给
规划任务。此阶段不得插入、删除、rehash、移动 Entry 或释放 posting；只有
frequency 原子标量可更新。每个 producer 保留原来的局部 delta 聚合，对同一
key 最多在本任务末尾提交一次，而不是对每次物理替换访问共享计数。producer
结束后扫描其 route map：fresh 记录不变，old 记录按以下协议消费。全部规划
任务 join 后，才重新取得 owner 的独占可变访问、删除退休项、归约和填充新
key，并允许下一轮选择。AA 同样必须经过完整路由、语料写回和 owner 提交后
才选择下一条规则；可以提前减旧频率，不能提前进入下一 epoch。

## 单调减频与唯一退休记录

令旧键批前频率为 f≥minimum，各 producer 的局部非负减量为 d_i。已有规划
协议必须保证每条被移除的旧邻边恰好计数一次，因此 Σd_i≤f。非 AA 相邻匹配
间的共享邻边由既有 left-selected 抑制重复；AA 沿既有全局奇偶规划。这里不
改变这些规则，不能用原子操作补救错误的重复减频。

每次查到 Entry 后执行 `old = fetch_sub(d_i, Relaxed)`。原子的修改顺序给出
一条单调序列。在 `old≥minimum && old−d_i<minimum` 的唯一那次保留本地旧
delta 记录，作为退休 marker；其他旧记录从 route map 删除。新记录全部保留。
owner 提交阶段在 producer 模式看到旧 key 时只删除其 Entry 和 posting，
不得再扣减一次。marker 的 key 本来就在该 owner 的 route 中，不必再建退休
队列、每 worker 的索引副本或新增 Entry 字段。所有 producer 已 join，故
删除时可以断言最终频率低于 minimum。每个 key 最多一个 marker，不会两个
owner 争抢同一 posting。

selected key 已移出字典；低于 minimum 的旧 key 也可能在批前就被删除。
这两种查询未命中都可跳过，与原 owner 协议一致。一个键在本批中跨过阈值后
仍要保留 Entry 到 join，其他 producer 的合法减量可以继续扣到最终非负值。
频率小于 minimum 不会再恢复，因此提前作出退休决定安全；实际释放必须等
共享借用结束。lazy heap 的旧候选可以过期，下一次 peek 用完整提交后的频率
刷新；本批期间没有消费者可观察未完成的计数。

若取回 old<d_i，说明减量不合法或实现错误。fetch_sub 已经改变私有计数，
所以只能让本训练调用返回错误，等待全部任务结束后丢弃状态，不能继续训练
或发布下一批。第一版若改用 checked CAS，则需单独记录重试成本，不能将其
计时当作一条无竞争 fetch_sub。Relaxed 用于各计数的单变量归约；结构冻结
以及任务 join 才是相位边界，不能把 Relaxed 当作字典或 posting 的发布机制。

## 预期工作和可能失败的原因

设 D_old 是 producer×旧 key 的局部记录数，R 是本批新退休 key 数。
producer 路径对现有 Entry 至多做 D_old 次字典查询和原子减法，owner 只处理
至多 R 个退休 marker；fresh 归约仍完全相同。它减少的是后一个阶段的旧键
工作，并可能将这些工作与其他 producer 剩余规划重叠，不是删除所有查找或
屏障。每 key 的原子修改次数最多 W，热门 key 可在一条 cache line 上争用；
不同 key 也可能共享 cache line。控制路径由唯一 owner 普通写，可能更有
局部性，尤其在多 NUMA 节点上。因此没有预先可保证的速度提升。

局部 route map 在规划时已分配。删除 old 条目并不保证释放底层分配，所以
不能仅凭提交记录更少宣称峰值内存下降。Entry 在本机应保持 24 字节、posting
16 字节，并用布局断言确认；这些是本机布局，不是所有架构的 AtomicU64 对齐
保证。原 W² 路由头、出生链和 owner 永久索引都还在。需分别记录 flush 前
D_old、实际 atomic 命中/减法数、R、flush 后记录数、route capacity、完整
调用 CPU/wall 与 HWM；phase timer 是否重叠也须明确。

第一轮用实际并发的小模型验证同 key 分散在多个 producer、恰好跨阈值、权重
接近合法 u64 上限、selected/missing 旧 key、非法下溢终止，以及 marker 不
重复减频。再比较完整 trainer 的 AA、相邻匹配、长 token、非均匀权重和域
边界，并对两模式与 eager fallback 做完整串行轨迹核对。计时只做必要小窗；
不因原子次数比 occurrence 数少，就声称优于原 owner 分片归约。
