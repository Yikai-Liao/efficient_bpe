# 并行语法压缩文献核验

本页只归纳与 Re-Pair/BPE 训练或并行语法构造直接相关的论文。关键区分是：输出能还原原文、输出一个压缩语法、输出与串行 Re-Pair 同一 grammar，以及逐条遵守同一个全局 greedy pair 顺序，是四种不同的保证。

## Re-Pair/BPE 训练和同一 grammar

**Köppl et al., “Re-Pair in Small Space,” Algorithms 14(1), 2021.** [论文](https://doi.org/10.3390/a14010005) · [作者版 PDF](https://koeppl.github.io/bin/paper/algorithms21repair.pdf)。论文给出低工作空间算法，另给 CRCW PRAM 与外存变体。并行版本将文本分块扫描、以并行排序维护高频 bigram 候选，再归并分块候选；定理为 `O(n²/p)` 时间、`O(n²)` work，工作空间在主算法空间上另加 `O(p log n)` bits。其 Re-Pair 频次按非重叠 bigram occurrence 定义，tie 可任取；因此不能直接声称逐条匹配本项目的重叠 occurrence 计数与固定字典序 tie-break。它不是现代 shared-memory 下的实用线性工作加速：作者指出早期版本处理 1 MiB 约需一小时，论文也未给出多核机器的 scaling 实验。外存算法权衡反复扫描与排序 I/O，目标是容量而非线程加速。价值在于明确展示：精确 Re-Pair 语义可并行化扫描/候选归并，但已知低空间路线可能付出高 work；不是用“并行替换一批规则”绕开串行 winner。

**Kim et al., “Re²Pair: Increasing the Scalability of RePair by Decreasing Memory Usage,” ESA 2024.** [Dagstuhl 论文与摘要](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2024.78) · [作者代码](https://github.com/jkim210/Recursive-RePair)。递归 prefix-free parsing 避免物化大 parse，同时恢复 RePair grammar；论文在 SARS-CoV-2 与 1000 Genomes haplotypes 上报告最大输入的峰值内存下降超过 40%，运行时间比 BigRePair 快 12%–79%。这是单线程/容量可扩展路线，不是线程化全局 greedy；“grammar 与 RePair 一致”针对其递归构造证明，不等于训练规则并行。

**Varki, Gagie, Boucher, “Efficient Grammar Compression via RLZ-Based RePair,” CPM 2026.** [Dagstuhl 论文](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CPM.2026.5) · [作者实现](https://github.com/rvarki/RLZ-RePair)。用参考序列上的 RLZ parse 保存目标文本结构，再对 parse 与 reference 协调执行 RePair；在参考选得合适时可得到 standard RePair grammar。作者报告可节省超过 80% 内存，代价是 modest runtime increase 和较少的替换次数。实验是 SARS-CoV-2、拟南芥、人 19 号染色体等高度相似基因组集合，不能直接外推到普通语言；该算法解决内存，并未把全局规则训练并行起来。

**Matsushita & Inoguchi, “Parallel Processing of Grammar Compression,” DCC 2021.** [IEEE DOI](https://doi.org/10.1109/DCC50243.2021.00068) · [DCC 官方议程](https://datacompressionconference.org/Programs/DCC2021Program.pdf)。这是最直接命名 Parallel Re-pair 的线程化工作。可访问的会议元数据摘要称各 CPU core 使用共享 dictionary 同步变量分配，报告最高 16/32 cores 的压缩加速；但此轮无法从 IEEE 获取论文全文或作者代码，因此不能核实它是否每一轮仍选全语料全局最高 pair、是否仅独立压块/编码、输入规模、硬件、speedup 数值及初始化是否计时。应作为必须追原文复核的线索，不能据摘要把它列作“已证实精确并行 greedy 训练”。

## 并行训练但构造不同 grammar

**Díaz-Domínguez, “Efficient Terabyte-Scale Text Compression via Stable Local Consistency and Parallel Grammar Processing,” SEA 2025.** [Dagstuhl 论文](https://doi.org/10.4230/LIPIcs.SEA.2025.14) · [作者代码](https://github.com/ddiazdom/lcg)。Stable local consistency 让相同 pattern 的独立分块产生拓扑一致的 parse core，线程独立构造 grammar 后再合并。作者用 16 threads 在 7.9 TB 细菌/古菌集合上约 9 小时完成，工作内存 0.43 bits/symbol；HUM 3.46 TB、COVID 267.4 GB、Linux kernel 54.4 GB 也被测试，机器为 192-core Xeon E7-8890 v4、3 TiB RAM。论文将 LCG 的并行效果与其他工具对比，并画 HUM 线程扩展曲线；摘要中的 7.9 TB/9 小时包含实际端到端压缩 pipeline。其算法是 locally consistent grammar 加后处理，不遵守 Re-Pair 每轮全局最高 pair 顺序。这是可扩展语法压缩的强替代目标，不是 BPE/Re-Pair exact baseline。

## 边界：编码并行不等于训练并行

固定 vocabulary 下对 token stream 做并行编码、按块套用已学规则、并行解码，均不会并行化规则学习本身。它们可以作为 pipeline 的其他阶段，但不能支持“Re-Pair/BPE training 是并行的”这一主张。类似地，分块各训一个 grammar 再合并，除非给出可证明的全局 greedy 等价，不等价于训练同一 rule sequence。

## 本项目可用结论

目前能找到的资料里，最扎实的“精确全局贪心 + 并行”证据是小空间 Re-Pair 的理论扫描/排序方案，work 达 `O(n²)`；“多核实用且扩到 TB”证据属于 stable-local-consistency 等不同 grammar 目标。RLZ-RePair / Re²Pair 是内存压缩路线。DCC 2021 Parallel Re-pair 值得拿到全文后单独审计。不能把 fixed-vocabulary encode、局部独立压块或同一压缩结果当作同一条全局 greedy 规则轨迹。
