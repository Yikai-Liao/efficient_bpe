# BPE / Re-Pair 复杂度对照：计数口径与适用边界

这份笔记只比较训练核心的工作量与表示成本，不把不同算法的实测时间推成渐近界。除特别注明外，哈希表操作按期望摊还 `O(1)`、机器整数和数组随机访问按 `O(1)` 计算；Python 的对象分配、哈希冲突、缓存局部性仍决定实际常数。

## 统一记号

| 记号 | 含义 |
|---|---|
| `N` | 原始输入中实际出现的 Unicode 字符总数；重复词的每次出现都计入。若程序处理 UTF-8 字节或 ASCII，应另标输入单位。 |
| `U` | 去重及预切分后，物理语料中的初始 token 位置数，包含固定分隔符时需一致计入。重复词只占一份物理位置，其出现次数成为权重。通常 `U ≪ N`，但预处理本身仍须读完 `N`。 |
| `R` | 所有轮次中实际执行的**物理相邻配对替换**总次数；一次规则可替换多个位置。每次替换使活 token 数少一，因此在不可跨分隔符合并时 `R ≤ U`。这不是词频权重之和。 |
| `P` | 任一时刻候选堆的最大规模；`P0` 是初始化候选数。另用 `E` 表示累计堆事件数，不能把某一时刻的规模与整个训练的事件总量混为一谈。 |
| `K` | 选中的 pair 规则数，即训练轮数；正常每轮至少一次有效替换，所以 `K ≤ R`。 |
| `G` | 去重词按频率排序后的不同频率组数；`G ≤` 去重词数。 |
| `L` | 一个 token 对应的原始字符长度；`L_max` 为所训练 token 的最大值。`ΣL` 指明是哪组操作/输出上的长度和。 |
| `D` | 实际物化或输出的 token 字符串总字符数；只保存整数规则 `(a,b)→z` 时，不必在每轮支付 `D`。 |

`N` 与 `U` 不能互换：去重前的读取/计数至少是 `Ω(N)`，去重后逐位置处理才可能按 `U` 计。词频是权重，不能把权重总量误作物理替换数 `R`。

## 逐实现对照

| 实现 | 核心数据与可支持的工作量界 | 不能省略的成本/条件 |
|---|---|---|
| 用户 v1 | 字符串语料、`array('B')` 边界长度、`pair→array('I')` 位置倒排、懒堆；不逐轮全串重数。[源码](/root/code/efficient_bpe/ebpe.py:224) | 每次实际替换会创建左右 token 的 Python 字符串切片，并在字典查找/插入中对新字符串计哈希；`word_comb`、`get_pre_word/get_nxt_word` 同样按字符长度工作。[源码](/root/code/efficient_bpe/ebpe.py:299) 这些操作的总量至少包含对应的 `ΣL`，最坏可写成 `O(R·L_max + K·L_max)`，再加倒排访问与堆事件的 `O(E log(2+P))`。自配对还会反转/复制位置数组。[源码](/root/code/efficient_bpe/ebpe.py:33) `array('B')` 对大于 255 的边界长度另有表示限制，不能据此声称任意长度的统一界。 |
| 原 v2 | 整数 ID、加权去重词语料、`array('Q')` 位置倒排、懒堆；建立物理语料约 `O(U)`（词计数与排序另算）。[源码](/root/code/efficient_bpe/ebpe_v2.py:178) | 每次替换的 `for i in range(pos_x+1,pos_end-1): corpus[i]=0` 会重写整个合并跨度，增量为 `Θ(Σ_replacements (L_pair-2)_+)`，可达 `O(R·L_max)`，不是 `O(R)`。[源码](/root/code/efficient_bpe/ebpe_v2.py:352) 频率组跳跃的 `freq_pivot[freq_i+1:]` 先复制剩余切片，再二分，单次最坏 `O(G)`，不能只记 `O(log G)`。[源码](/root/code/efficient_bpe/ebpe_v2.py:343) `assign_token`/模型输出仍会连接、构造字符串，另计 `D`。[源码](/root/code/efficient_bpe/ebpe_v2.py:298) |
| 修正 v2 双端 | 保留整数 ID 和两端写入：被吸收的旧右起点与旧左终点清零，新 token 只写起止两端；对应原型见 [FastEbpeEndpoints](/root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite/backends_compact.py:195)。 | **条件式训练核心界** `O(U + (P0+R) log(2+P) + R log(2+G) + D)`：先完成去重/排序；每个有效替换只改常数个边界及邻接计数；权重查询对原 `freq_pivot` 用不复制切片的二分；倒排只对新增 pair 追加常数个位置且失效项不被重复重建；懒堆事件数须按初始候选及邻接变化次数摊还约束；哈希期望 `O(1)`。若已知位置按组有序，频率查询还能改为顺序游标的摊还界。它是**满足这些修正后的目标界**，不是当前 `ebpe_v2.py` 已证得的界。预处理另有至少 `Ω(N)` 与去重词排序成本。 |
| `linked + common` | 无 RLE 的数组内链，四数组 `16U` 逻辑字节或省 runlength 的三数组 `12U`；邻居查询/单次合并为 `O(1)`。[后端](/root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite/backend_linked.py:18) | [common.py](/root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite/common.py:38) 仍建 Python `pair→array('I')` 倒排、频率哈希表、懒 `heapq`，并每次有效替换二分 `pivots`。在同一 `P/P0` 口径和假设下，训练核心同样是 `O(U+(P0+R)log(2+P)+Rlog(2+G)+D)` 的保守上界；只换邻接布局主要改变常数与数组字节数。 |
| `Prezza halfword + common` | 原型 [FastPrezzaHalfword](/root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite/backends_compact.py:249) 用 `u16` 语料、64 位活位图和每块 `u32` skip，逻辑数组约 `2U + U/8 + 4⌈U/64⌉` 字节；机器字假设下邻居/替换仍 `O(1)`。 | 它和 `linked` 使用相同 [common.py](/root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite/common.py:45) 倒排、哈希、懒堆、权重查询，故也只有上述**共同驱动**的条件式界。它没有移植完整 Re-Pair 的全局 `TP` arena、在旧位置区间中惰性发现新 pair 的同步过程、以及专用高/低频队列。[原实现](/tmp/efficient-bpe-audit-20260929/prezza-repair/rp.cpp:167) 因而不能借论文 `O(N)` 证明给这个 Python port；2.19 字节/位置也仅是此后端数组，不是进程总 RSS。 |
| 2017 完整 Re-Pair | 论文为**可重写输入**的长度 `n` 个机器字给出期望 `O(n/ε)` 时间、输入之外 `(1+ε)n+√n` 个字的工作空间（固定常数 `ε∈(0,1]` 时为期望线性时间）；另一个方案为期望 `O(n log n)` 时间、`n+√n` 额外字。[作者论文](https://www2.imm.dtu.dk/~phbi/files/publications/2017serpcC.pdf) | 证明依赖完整两阶段队列、全局 `TP` 位置区间与同步/重分组、可重写文本及其摊还分析。作者实际 C++ `rp` 面向 ASCII 文件，README 报告约 `6n` 字节 RAM 与线性运行时间；这是该程序自己的输入口径和经验实现声明，[README](/tmp/efficient-bpe-audit-20260929/prezza-repair/README.md:5) 并非 Python 半字后端的 RSS 保证。它也没有本比较的预切分、去重词权重与相同 tie 规则。 |
| YTTM 高/低队列 | [bpe.cpp](/tmp/efficient-bpe-audit-20260929/yttm/youtokentome/cpp/bpe.cpp:149) 的低频端是频率整数桶，高频端每次取顶扫描并刷新数组；阈值 `B=floor(sqrt(text_len[0]))`，[源码](/tmp/efficient-bpe-audit-20260929/yttm/youtokentome/cpp/bpe.cpp:1049) 其中 `text_len[0]` 已汇总原输入各线程 Unicode 字符数，[源码](/tmp/efficient-bpe-audit-20260929/yttm/youtokentome/cpp/bpe.cpp:1013) 不是去重物理位数 `U`。 | 我们的 Python 高低队列若以原始**加权邻接量** `M` 为 `initial_mass`，其阈值才是 `floor(√M)`；YTTM 直接用原输入字符数，二者遇到空白等预处理差异时也未必相等。`M` 与 `U` 在重复词语料上可相差很大。一次高频取顶扫描 `H` 个候选并逐个汇总线程局部计数，约 `O(H·T)`（`T` 为线程数），低频桶指针下行另有成本。打开确定性 tie 宏时，高频数组每次还排序，低频桶变脏时排序，不能把固定 tie 的代价当作免费。[源码](/tmp/efficient-bpe-audit-20260929/yttm/youtokentome/cpp/bpe.cpp:165) YTTM 按 run 压缩同符号片段，自配对有效数为 `⌊run_len/2⌋×词频`，[源码](/tmp/efficient-bpe-audit-20260929/yttm/youtokentome/cpp/bpe.cpp:140) 与 `common.py` 先统计重叠邻接再从左到右替换的计分不同；队列阈值和节点数均不能直接口称 `O(U)`。 |

## 何处需要实测，何处已有证明

四个 Python 邻接版本可以在同一 `common.py` 规则、倒排、权重和堆下比较**实际常数**；这个对照隔离了物理语料表示，却共享 Python 字典与堆的主开销。完整 Prezza C++ 程序与 YTTM C++ 程序适合作工程基线，但输入单位、预切分、词权重、自配对分数、停止阈值和同频排序不同，时间/RSS 结果应与这些差异并列报告。低频桶的确定性排序、Python `array` 容量余量、字典/元组头和输出字符串 `D` 都必须进入实测解释。
