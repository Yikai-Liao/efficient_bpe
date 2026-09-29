# 保持精确串行优先级的并行贪心：可迁移结构与失效边界

## 问题抽象

把当前 token corpus 看成动态路径：每条活邻接边带 label `q=(left_id,right_id)`，权重是该 label 的加权出现次数。每轮挑全局 `(-frequency, label)` 最小者，将其互不重叠的出现收缩成 fresh ID；局部边删除、最多在收缩边界产生新 label。旧 label 的计数只会下降；新 label 的频次不高于本轮 winner 的频次。后一个结论来自：一类新 label 每个被替换的 occurrence 至多产生一个对应方向的边，replacement 总权重不超过 winner 的 pair-occurrence 权重。它不会变成高于 winner 的新高峰，但可能与 winner 同频，且可能高于随后剩余的旧候选。

目标不是只返回一个合法 grammar，而是保留严格相同的全局 winner 序列、同频字典序、每条规则的出现计数和最终语料。并行调度允许交换的前提是能证明交换后的状态与严格序列在所有可观察量上等价。

## 相关精确并行范式

### 静态优先级依赖 DAG：MIS / maximal matching

Blelloch、Fineman、Shun 的 [greedy MIS/matching 论文](https://arxiv.org/abs/1202.3205)给每个顶点固定随机顺序。顶点只依赖冲突邻居中更高优先级的决策；建立 priority DAG 后，同一时刻可以处理所有根。这样返回的 MIS/matching 与指定顺序的串行 greedy 完全一致。对任意输入图、随机顶点顺序，依赖深度为 `O(log² n)` w.h.p.；文中给出线性 work 实现，并实验了 work/span 的折衷。论文不是 compression，也没有动态重打分：图与优先级在一次 greedy 运行中固定。即使有 `O(log² n)` 理论 span，worst-case priority DAG 仍可为线性深；更关键的是 BPE/Re-Pair 的候选分数会被已提交 rewrite 改变，还会创建新候选。不能只把 pair label 当顶点、将初始频率排序后套 DAG。

可迁移的是“先建立精确依赖，再只运行 ready tasks”，而不是 MIS 的随机深度结论。若把一个规则提交视作事件，事件依赖包括：所有可能改变当前 `argmax(-freq, pair)` 的既有 pair 更新、可能产生更靠前新 pair 的边界事件，以及占用相同路径位置的规则应用。没有频率界证书时，全局 maximum 是每条规则的潜在依赖，规则 DAG 会退化成链；按固定随机序排列 pair 会改变算法语义。

### Relaxed priority queue：可控违序不等于保留 greedy

Alistarh 等人的 [Relaxed Schedulers Can Efficiently Parallelize Iterative Algorithms](https://doi.org/10.1145/3212734.3212756)研究允许有限 priority inversion 的并发队列，并在 MIS/matching 上证明可通过问题结构得到确定性结果和额外迭代界；exact scheduler 被描述为不易扩展，relaxation 用额外工作换取并发。可借鉴的是：明确度量违序、额外工作和完成轮数，再设计恢复/修正协议。不可直接移植的是“队列近似 top-K”：对 BPE，只要把一个低频 pair 提前替换，语料就可能立刻生成不同的新 IDs，之后 grammar 轨迹不可逆地分叉。若做 relaxed mode，必须将输出明确标为近似/不同规则序列，并单独评估压缩质量或 downstream 效果。

### Borůvka / graph contraction：安全边集依赖问题特有定理

并行 Borůvka 每个连通分量选择最小出边，然后并行收缩；其正确性来自 cut property：选中的边都能扩展到最小生成树，轮数按分量数几何下降。这个过程通常不按 Kruskal 的全局边序逐条执行，但可得同一 MST；若边权全异，MST 唯一时最终边集也与 Kruskal 相同。它保留的是目标最优解，不是串行 greedy 的执行 trace。Re-Pair 没有对应 cut/cycle property：高频 pair 不一定“安全”，同时替换一组各自当前频繁的 pair 可以改变彼此频次与之后的可见边界。比如路径 `a b a b a b ...` 中 `ab` 与 `ba` 的 match 重叠；选择其一会改变另一类的 occurrence 和产生的 fresh-ID edges。仅证明 batch members 当前不共享 occurrence，仍不足以证明新 pair 的下一轮 priority 不变。

因此，适合的迁移不是借用 Borůvka 的算法本身，而是寻找语法替换问题是否存在类似 exchange/cut 的安全性定理。若没有，Borůvka 式“每个分区各取赢家再收缩”只能作为另一种 grammar heuristic，不能声称与串行全局 winner 等价。

### Huffman：特殊有序权重结构可让多轮贪心并行

Berman、Karpinski、Nekrich 的 [Approximating Huffman Codes in Parallel](https://doi.org/10.1016/j.jda.2006.10.007)表明，已排序权重下可并行合并序列构造 almost-optimal Huffman tree；论文也给出以 Huffman 树高度 `H` 为界的 exact construction（`O(H)` time、`n` processors）。关键是 Huffman 每步取两个最小树，合并权重是两者之和，批量合并可利用排序序列/权重分层和深度界；这不是任意动态优先队列都能批量化。exact tree 的高度在最坏情况下仍可能线性，且不同 tie 约定可能得到不同的同最优树。Re-Pair 的 merge 会局部改变许多不同 label 的频率，不产生单调的“合并权重有序数组”，故不能照搬 Huffman 批次。可迁移思路是寻找可证明的频率区间/批次，并将树高对应为“winner frequency plateau / 同优先级依赖层”，而非对所有候选做 speculative replacement。

### 低空间并行 Re-Pair：全局精确但 work 较高

[Re-Pair in Small Space (2021)](https://doi.org/10.3390/a14010005)给出 CRCW PRAM Re-Pair 变体：并行扫描、对高频候选并行排序/归并，复杂度 `O(n²/p)` time、`O(n²)` work，并在额外工作空间中加入 `O(p log n)` bits；它还给外存算法。论文中的 bigram frequency 是非重叠 occurrence，最大频次平局可任意选择，所以它不是对固定重叠频次、字典序 tie-break 实现的逐轨迹证明。论文没有给出现代共享内存多核的 scale-up benchmark，并报告先前版本处理 1 MiB 已需约一小时。它证明全局 Re-Pair 语义有并行 PRAM 算法，但当前可核验的路线并非 work-efficient。工程目标应是把 serial work 降到接近实际受影响 occurrence 数，同时把同步数压到少数证书安全的 epoch。

## 频率单调性带来的证书空间

设当前 winner 的加权频次为 `F`。所有既有 pair 的频次在替换后只会下降；每类新 pair 的频次 `f_new ≤ F`。因此每轮最大频次不增。这比“新频次可能任意上涨”更强，但仍不允许跳过全局选择：一个新 pair 可以恰好等于 `F`，也可能大于已被当前 rewrite 削弱的下一旧候选。Tie-break 是语义的一部分。

可以研究一个 **certified epoch**，而不是按线程数固定一次批多条：

1. 用当前 heap/top candidates 产生候选 winner 顺序 `r₁, r₂, …`，为每个 rule occurrence 记录其左右边界上下文和 piece 权重。
2. 对每个候选 rule 及可能由前序规则创建的 label，维护 frequency interval `[L(q), U(q)]`。旧 label 的上界可取当前频率并只扣除已知删除；新 label 的上界来自产生它的 rewrite occurrence 的 weighted mass，并可按 `(left_context, right_context, piece)` 细分，避免全都用粗上界 `F`。
3. 只有当目标 `rᵢ` 的下界高于所有未完成候选的上界，或等于上界且 pair key 按精确 tie-break 更小，才能证实它是下一条串行规则。还须证明已选应用集在物理 occurrence 上无冲突，且所有边界创建/删除 delta 能在独立 shard 计算后无损归并。
4. 证书若失败，提交已证实的最大前缀，或退化为单规则 barrier；精确模式不可因超时/粒度不足静默放宽证书。

这个方向仍有未知点：`L` 通常难算，因为不同批次规则可能删除同一候选 occurrence；`U` 过松会使 batch 长度常为 1；建立 per-pair context 上界自身可能复制 occurrence 索引、消耗内存。需要先离线记录真实 trace 中证书命中率、每次证书检查 work、epoch 数、batch 长度与热点分布。只有同时匹配 serial trace 且减少端到端 barrier/CPU，才是有效改进。

另一个更保守方向是让 worker 并行计算 winner 证书所需数据、coordinator 串行提交规则：减少扫描或归并，而不尝试多规则一轮。按 label 分片的局部 histogram 只有在完整归并后才能确定精确 global max；减少共享原子写和稀疏长尾有价值，但不是减少 greedy span。静态优先级工作里常见的 randomized ordering 对输入规则优先级做了重排，也不能用于 exact BPE。

## 并行 grammar 替代目标与规模外推

[Stable local consistency (SEA 2025)](https://doi.org/10.4230/LIPIcs.SEA.2025.14)配合 [作者实现 LCG](https://github.com/ddiazdom/lcg)，在 16 threads、3.46 TB 人类基因组、7.9 TB 细菌/古菌集合等场景演示并行 grammar construction；7.9 TB 实验约九小时，单机配置为 192-core Xeon、3 TiB RAM。其 parser 按稳定拓扑跨块独立工作、之后 merge，目标就是高吞吐 grammar compression，不是同一 Re-Pair/BPE greedy trace。数据以基因组为主，不能把其 bytes/s 或 compression ratio 当普通自然语言的预期。

这类局部一致 grammar 是项目追求吞吐时的真实替代研究线：应保留独立 algorithm/grammar 标签，比较训练耗时、RSS、语法大小、解码/随机访问能力及下游任务；不要把“exact BPE 加速”作为结果标题。基因组中长重复和极小 alphabet 对 parsing-core 重用有利，普通语言的空格、标点、词频长尾和跨段结构不同，必须各自benchmark。

## 设计路线建议

1. **先落依赖度量，不改规则顺序。** 用现有精确训练轨迹为每条 rule 记录 visited historical positions、有效 occurrences、实际变化的 pair labels、worker shard、计数 delta、heap winner margin、产生的新 label 频次。区分热规则（广域更新）和长尾规则。作为当前 coordinator/barrier 的成本基线。
2. **做动态 rule-event DAG 离线重放。** 若规则 `r_j` 对 `r_i` 的 winner 判定没有任何可能影响、且 corpus applications 对易位可交换，则不连依赖边；以最保守的 pair frequency 上界先构图。记录 DAG depth、frontier width、可并行 work 与 certificate false positives。不能将静态随机 MIS 的 O(log² n) 直接套到这张 trace 上。
3. **尝试每 pair 的上下文频次上界。** Fresh pair 必含本轮 ID，且 weighted count ≤ 其生成规则的 replacement mass。将生成量按 boundary context 与 piece/shard 归类，寻找 tight upper bounds。先用 replay 证明所有已跳过候选在每一步都不可能胜出，再考虑运行时证书。
4. **只实现可证安全 epoch。** Epoch 最大化满足 serial-prefix winner 的连续规则数；winner 仍按 `(-freq, lex pair)`。批次 apply 阶段需要对 overlap、piece 权重、共享边界字节/word、fresh ID 分配和 delta 归并给出无数据竞争证明。碰到相同优先级/上界不紧时立刻缩批或单步。
5. **单独评估不同 grammar 并行路线。** 如果 exact 证书命中率很低，保留 coordinator exact 作为语义基准，另外实现 stable local consistency/grammar merge 或其他局部一致法，明确承认 grammar 不同。比较常规文本、源码、DNA、多尺度长度与重复度，不只报 genome 多核数字。

每条“exact 并行训练”结论至少须通过逐条 rule key/count、ID 分配、实际替换位置和 final corpus 与固定 tie-break serial oracle 完全相同；对随机权重、重叠 runs、极端同频、新 label 恰好达到旧 winner、空/单个 piece、超长 piece 和不同线程/分片配置重复验证。性能同时报告 init + 全训练的 CPU 和 wall、barrier/证书成本、总 RSS/PSS、每线程负载及单巨 piece 表现。work、span、并发核数和 NUMA 通信是不同指标：共享 global heap 热点可能限制 wall speed，复制索引提升局部性却提高内存，跨 NUMA 的每轮全局 histogram 可让 nominal parallel work 被通信主导。外存结果则须给 I/O 次数与扫描/排序字节，不能仅用 RAM 算法复杂度代替。

## 主要来源

- Blelloch, Fineman, Shun, [Greedy Sequential Maximal Independent Set and Matching are Parallel on Average](https://arxiv.org/abs/1202.3205), 2012.
- Alistarh, Brown, Kopinsky, Nadiradze, [Relaxed Schedulers Can Efficiently Parallelize Iterative Algorithms](https://doi.org/10.1145/3212734.3212756), PODC 2018.
- Berman, Karpinski, Nekrich, [Approximating Huffman Codes in Parallel](https://doi.org/10.1016/j.jda.2006.10.007), 2007; [ECCC technical report](https://eccc.weizmann.ac.il/eccc-reports/2002/TR02-018/index.html).
- Köppl et al., [Re-Pair in Small Space](https://doi.org/10.3390/a14010005), 2021.
- Díaz-Domínguez, [Efficient Terabyte-Scale Text Compression via Stable Local Consistency and Parallel Grammar Processing](https://doi.org/10.4230/LIPIcs.SEA.2025.14), 2025.
- Matsushita & Inoguchi, [Parallel Processing of Grammar Compression](https://doi.org/10.1109/DCC50243.2021.00068), DCC 2021. 目前只核到会议官方议程与 DOI；在取得全文前，不将其等价性或实验细节作为结论。
