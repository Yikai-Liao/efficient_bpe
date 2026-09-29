# 密集 AA 位图的精确奇偶选择

独立消融位于 [owned_aa_bitmap](experiments/radical/owned_aa_bitmap/DESIGN.md)。只替换选中 `AA` 的处理；非 AA 批次和 owner 更新仍沿用 `owned_integer_hash`。同一二进制用 `--aa-order sort|bitmap-adaptive` 对照，默认 `sort`。候选先通过逐规则 `(pair,frequency)` 和 final token oracle，再讨论速度。

令 `N=corpus.len()`、`H=selected AA` 的历史 posting 长度。仅在 `H>=ceil(N/16)` 且一份位图的实际容量不超过该 posting 的堆容量字节时使用位图；否则按原路径排序。实际 capacity 守卫在分配/清零后执行，若 allocator 超配，回退前仍会有短暂额外峰值；不能把它称为严格的分配前 RSS 上界。`AtomicU64[ceil(N/64)]` 中的第 `p` 位仅由历史 posting 中在稳定语料上通过 `inspect(p,A,A)` 的**真实起点**设置。不能盲扫语料的 ID 标签：长 token 两端都写 ID。例如三个长度 2 的 A 会形成 `[0,A,A,A,A,A,A,0]`，物理位置 2 是末端标签，却可能通过局部 ID 邻接检查。散射所有任务 join 后，统计 bit popcount 与有效 posting 数相等，避免重复真实起点被位图静默去重，并释放已选 posting。

按连续的 64 个 64-bit word 分块，各块并行流式枚举 set bit，计算首尾位置、尾 run 长度奇偶、是否全块单 run。相邻有效起点差恰好 `len(A)` 才属于同一 run；无论长 token、空物理块或 piece sentinel，都按此规则处理。协调线程只对 `G=ceil(ceil(N/64)/64)` 个摘要计算入块奇偶。随后第一次并行位图遍历按全局左到右奇偶选择不重叠匹配，仍在稳定语料上取邻居和 piece 权重，并直接路由 owner delta 与 birth 链。对已选 p，若 `bit[p-L]` 存在，它处于本 run 偶数偏移且偏移至少为 2，因此 `p-2L` 已选，左边出生由前一合并负责；若不存在则无紧邻的前一已选匹配。若 `bit[p+2L]` 存在，`p+L` 的 A 同时由 p 的有效边和后者的有效边保证，所以 `p+2L` 是下一个偶偏移已选匹配，右边最终出生为 `Z/Z`。越界查询返回 false。所有路由任务 join 后，第二次并行位图遍历只凭位置、token 长度、fresh ID 写不相交的端点。没有 `valid_chunks`、selected 起点 Vec 或 Plan Vec，也不会在端点写入期间读取邻居快照。

加权候选频率仍来自所有**重叠** AA 边；奇偶只决定本轮实际替换的起点。每个替换位置的增减量重新按 pivot 读取权重。一个 key 只会被选一次，全部密集 AA 的历史记录总量不超过初始 eligible posting 与存储的出生 posting 之和。每轮散射 `O(H)`、摘要和两次选中遍历各 `O(N/64+V)`，其中 `V<=H`；密度条件给出 `N<=16H`，故整个调用的新增扫描为 `O(ΣH)`，但 stale 很多或原子 OR 竞争激烈时仍可能慢。位图逻辑载荷约 `N/8<=2H+8` 字节；散射峰值同时含选中 posting，路由阶段同时含位图、`O(G)` 摘要和原有 owner route 输出。没有 `O(WN)` 副本。原 `peak_plan_len` 仍是逻辑选择数，真实临时 Plan 内存由新 Plan Vec capacity 指标表示，且不包括 sort 路径有效起点 Vec 和 Vec 头；总峰值仍以进程 HWM 检查。分别记录位图实际容量、bitset 块数/每块 64 words、sort 模式 Plan Vec 实际容量、散射/摘要/前缀/路由/写入时间与密集轮次/回退轮次，以免把峰值或阶段代价隐藏在总时间里。

后续独立实验是 scatter 原子操作的 word 缓冲，已通过 18 项 Rust 测试及
162 次完整轨迹 oracle，见[8 次轻量小测](batch_results/radical-aa-bitmap-cache-quick-v1/README.md)：
每个任务可用局部 `(word_index, pending_bits)` 合并连续落在同一 64 位 word 的
起点，切换 word 或任务结束时才 `fetch_or`。它不要求 posting 全局有序，最坏
仍每有效位置一次原子操作；连续起点则可能降为每 word 一次。局部状态仅 O(W)，
没有额外 N 数组。全局有效数与 popcount 的核对仍能发现被局部合并的重复位。
unary/AB 的原子 OR 次数约减少 30/19 倍，四格 scatter 子阶段均下降，
完整调用方向仍不一致；不能把更少的原子指令直接换算为训练加速比。
