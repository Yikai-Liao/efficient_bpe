# Rust 精确 BPE 基线：契约与演进顺序

本 crate 是可复现的加权贪心 BPE 核心基线，输入已经是数值化的 `Prepared` 语料。它用于固定语义、边界布局和测量口径；当前没有原生多线程训练，也不应把它描述成突破性压缩算法。

## 输入与输出契约

调用方提供四个对齐部分：`corpus: Vec<u32>`、`initial_lengths: Vec<u32>`、`pivots: Vec<u32>` 和 `weights: Vec<u64>`。文本解码、规范化、预分词、piece 权重汇总及 token ID 映射都在 crate 之外完成；Rust 基线不接收原始文本，也不实现 HF/regex tokenizer。

`corpus` 是展平的 piece 序列，首尾必须是 ID 0；每个 piece 非空，piece 之间也用 0 隔开。`pivots` 按严格递增顺序给出各权重段的起点；每个起点必须位于一个 piece 的首 token，首 pivot 必须为 1；它与正权重数组一一对应。初始长度表是稠密 ID 表，ID 0 也占一项，且每项初始长度均为 1；每个非零初始 ID 都必须出现。语料位置数必须小于 `2^32`，初始 ID 与最多新建的 ID 必须能放进 u32。空语料的唯一合法形式是 `corpus=[0]`、只含 ID 0 的长度表以及空 pivot/weight 数组。

频率按 piece 权重累加，并包含重叠相邻对；选择频率最高且达到 `min_frequency` 的 pair，同频时选择 `(left_id, right_id)` 字典序最小者。一次规则应用在每个 piece 内从左向右作不重叠替换，piece 边界不可跨越。新 token ID 严格递增且不复用。频次和合并长度用 checked arithmetic；权重、总加权邻接数、pair 频次或 token 长度溢出会返回错误。输出包含完整规则轨迹、最终展平 token 序列以及时间、访问次数、后端逻辑字节数和指纹所需数据。

## 基线表示、内存口径与安全边界

当前端点后端是每个原始语料位置一个 u32 的 `Vec<u32>`：逻辑后端大小是 `4N` 字节，其中 `N = corpus.len()`，不是总进程内存。初始 pair 起点索引的每个位置是 u32；令 `E` 为初始非分隔符相邻 pair 的出现次数，则其位置 payload 的基础计数是 `4E` 字节。合计 `4N + 4E` 只是两个数组 payload 的口径，未计 pair-position 的动态新记录、`Vec` 容量余量及头部、`HashMap`/`HashSet`、频次表、堆、长度/权重/pivot 表、规则结果、分配器、输入 JSON 和进程运行时。因此 `backend_buffer_bytes` 和 `initial_occurrence_bytes` 不是 RSS，也不是完整内存上界。

端点记录当前 live token 的起点和末端 ID，零值用于分隔符或已失效的右起点；合并保留的旧左端点可能成为内部陈旧值。`inspect_pair` 只对历史索引里的 pair 起点作可信检查：fresh ID 使被消费起点失配，接着按 token 长度计算右起点并验证 pair。它不是任意位置的通用 `alive` 查询。每次合并依赖通过检查取得的上下文；轻量更新需要 2 或 3 次 u32 写入，工作量不随 token 长度增长。

`Bounds::Checked` 使用常规索引。`Bounds::Unchecked` 仅让私有端点 `read/write` 在经过 `train` 的完整输入验证后使用 `get_unchecked`；它并没有让无效输入变合法，也没有取消相邻关系、算术溢出和终点边界检查。安全依赖是：后端不逃逸给调用方；初始扫描位置来自已验证的 corpus 范围；`inspect_pair` 先验证 `pos`、ID、长度计算、`right < last` 和 `after <= last`；`merge_known` 只接收该检查返回的上下文，所以 `pos`、`right`、`after-1` 均在分配范围内；最终遍历仍断言下一个边界递增且不越过 EOF。新增代码若改变这些调用关系，必须重新证明每个 unchecked 访问的界限。unchecked 只是一个待测构建选项，不能先验称为更快。

初始长度表和合并长度均为 u32，长于 255 的 token 不再受原 u8 长度限制；测试已覆盖 ID 与 token 长度越过 65535。这个改动没有把语料后端降到每位置 1 或 2 字节：当前仍是 `4N`。Python 实验中的 H3/H2.5 将 ID 与边界 tag 组合，`ByteSpans` 则只实现拓扑边界而不存 token ID；它们尚未移植到 Rust。相关的 memory 口径、取舍和实验结果见 [Python evolution 报告](../benchmarks/bpe_core_comparison/evolution/REPORT.md) 及其中的 [ByteSpans 说明](../benchmarks/bpe_core_comparison/evolution/byte_spans_notes.md)。

## 复杂度读法

对一个已知 live token 起点，邻居位置可由长度表作加减；`inspect_pair` 和 `merge_known` 访问固定数量数组单元，是 O(1)，不因 token 变长而扫描其内部位置。这只描述边界操作，不等于整个训练器 O(1) 或线性时间。

令 `N` 为 corpus 位置数，`E` 为初始 pair occurrence 数，`R` 为实际替换数，`V` 为被访问的历史 occurrence 数（包含 stale visit），`G` 为权重段（pivot）数，`K` 为输出规则数，`P` 为堆最大规模，`P0` 为初始候选数，`H` 为堆 pop 次数。初始扫描执行 O(N) 次预分词位置处理和期望 O(1) 的哈希更新；堆候选用 `BinaryHeap::from` 一次建堆，与 Python 的 heapify 一致，初始化堆成本为 O(P0)。每条有效替换最多建立两个新 pair 起点，因此历史位置记录访问量由初始 E 和至多 2R 条新增记录决定，stale occurrence 检查计入 V，而不应忽略。每次有效替换通过 `partition_point` 查所属 piece 权重，额外 O(log G)。频次/位置索引和新 pair 去重使用哈希容器，O(1) 是固定宽 key 下的期望成本。惰性堆事件的总成本按实际 pop 数计，为 O(H log P)，候选入堆也需计入；不能仅以实际规则数代替所有 stale heap work。

以这些参数表示，训练核心可读作期望 O(N + P0 + V + R log(G+1) + (R + H) log(P+1))，其中容器哈希假设期望常数；这是按本实现中的操作计数组织的界，不是对恶意哈希、任意精度权重或最终字符串词表构造的无条件界。空间需容纳语料、位置 occurrence、规则/长度和容器，逻辑项约为 O(N + E + R + K + P)，另有 Rust 容器容量与结构开销。这里的 `O(1)` 邻居查询绝不能被单独拿来推导总 trainer 的复杂度。

## 后续工作顺序

1. **冻结语义基线。** 保留 checked/unchecked 两种模式和完整规则轨迹；先用全频次重算的朴素 oracle 锁定重叠计数、字典序同频、piece 内从左到右替换、fresh ID、权重和阈值行为。当前 differential 记录为 205 个用例、410 次 Rust 运行，规则、最终序列和指纹均与 Python oracle 相同；基线变化前更新该证据。
2. **固定可复现计量。** 对真实英文/中文、regex/paragraph 及长 token / 重复 run 夹具分别记录初始化、合并、总训练 CPU 与墙钟、峰值 RSS/PSS、N/E/V/stale/heap-pop 和输入 hash。每配置至少 3 个新进程取中位数并保留范围；不要只报 merge 阶段或挑最好一次。内存数字须区分 `4N`、初始 `4E`、容器容量和进程峰值。
3. **独立评估计数索引。** 单独试验按初始最低频率筛掉不可能达阈值的 pair occurrence。旧 pair 频率只会下降；本轮新 pair 必须等全轮 occurrence 加完后再决定是否入队。逐条规则轨迹与 baseline 完全一致后，比较初始化 CPU、总训练 CPU、occurrence 数和 RSS。不要同时改变语料布局，以便归因。
4. **独立评估紧凑边界。** 先在独立 Rust 组件中验证 tag/payload、长短切换、u32 长度、previous/next 与 stale boundary，再考虑和 ID 编码组合。评估时报告整个后端字节和 RSS，并覆盖 ID >65535；单独一个 1N 边界组件不能被写成完整 1N tokenizer。H3、半字节布局或其它方向每次只改一个因素。
5. **建立原生线程分片基线，可与紧凑布局独立推进。** 以 piece 为不可拆分所有权单元，分片独占语料片段、pair-position 和局部频次。全局 coordinator 在每一轮完整选择唯一的最高频 lex 最小规则；随后发布 fresh ID/长度，worker 在各自 piece 内按左到右替换并返回精确频次 delta 与新 pair，barrier 等待所有 worker 后归并，再允许选择下一条规则。该 barrier 才能保证全局 greedy 顺序；不要并行发布规则或只靠两个 pair 不重叠来推断等价。
6. **显式处理线程边界与内存竞争。** piece 分片保留了词内邻接，但单个超长 piece 只能由一个 worker 处理，可能让四线程实际只有一个忙碌分片。初版保持巨 piece 不拆分并报告最重分片；若之后切 piece，必须为跨界 token/pair 与规则应用设计协议，并以完整 trace differential 证明同一全局语义。共享只读 token lengths 可在 barrier 版本化后发布；可写 corpus/occurrence 归 shard 所有。H2.5 一类半字节元数据可能让相邻 shard 写同一物理 byte，bitmap 可能共享 u64 word；逻辑位置不重叠也不代表无 data race。先用分片私有元数据或按物理字节/word 对齐所有权，再考虑共享压缩结构。Rust safe slice 分割优先；任何跨片 `unsafe` 都须有书面别名/同步证明与并发检查。

## 每个阶段的验收标准

语义改动必须与朴素 oracle 逐条规则和最终 token 序列一致，且输入、配置和规则指纹固定；测试覆盖 `aaa` 重叠、多个 piece、长短 token、旧 occurrence 失效、非均匀权重、阈值边界、u32 长度/ID 溢出与拒绝行为。并行版还须在 1/2/4 worker、不同分片顺序和重复调度下逐条匹配串行规则轨迹；对单巨 piece 明确报告 worker 利用率。`unchecked` 的用例必须与 checked 完全一致，并覆盖每个 unsafe 边界前提；改动 unsafe 周边后重跑测试，并在可用环境使用 Miri 或等价工具检查。

性能结论看完整训练 CPU、墙钟与内存的联合变化，至少三次新进程的中位数与范围；差值落在运行范围内时称为持平/未分辨，而非胜出。线程版须同时报告 call/train wall、总进程 CPU、每轮 barrier/归并开销和进程内存；更低 wall 若以显著更高 CPU/PSS 换来，应明确写成延迟与资源的折中。unsafe、紧凑编码、线程数或新索引都没有预设收益，只有语义门槛先通过、再出现可复现的目标指标改善，才进入默认路径。
