# Rust 消融实验覆盖清单

目标是找到跨输入规模、重复度、piece 结构、token 长度和 worker 数都能解释的时间—内存折中。结果按 workload 与变体报告 Pareto 前沿及明显退化边界；不按某一语种、固定 RAM 预算、某个线程数或单个最快结果调参。当前执行环境只暴露 6 个 CPU、单 NUMA 节点；它只是本轮可测上限，不是用户要求的双路 Xeon 验收机。本文和结果不得外推成双路硬件表现。

## 已有 Python 试验到 Rust variant 的映射

表中“旧证据”指已有只读 Python 实现或微基准。Rust 变体必须与同语义 oracle 比完整规则轨迹和最终 token；`metrics` 中的操作计数可以因数据结构而异，不能因此判错或要求相同。

| Rust variant | 旧试验 / 来源 | 要隔离的维度与必要说明 |
|---|---|---|
| `full_clear` | `python_rewrite/chain_bench.py:FullClear` | 每次合并按 span 清内部区间；只在合法长链微基准有旧实现。检验逐 token 清零的长度相关成本。 |
| `endpoints` | `backends_fused.py:FusedEbpeEndpoints` | 四写端点：清旧左端/右起点，写新起点/终点；pair 使用 tuple。 |
| `lean` | `evolution/lean_backend.py` | tuple pair + 2/3 写；依赖 fresh ID 和历史 pair-start 查询契约。 |
| `packed` | `evolution/packed_driver.py` | lean 后端不变，将 pair key 打包为 u64；单独测 key/hash/比较收益。 |
| `unfused_endpoints` | Rust 新对照 | 保持 endpoint corpus，但恢复拆开的 pair 检查、neighbor 查询和 merge 查询；隔离 context fusion 的收益。 |
| `unfused_halfword` | Rust 新对照 | halfword corpus 上运行拆开的查询接口，受 u16 字母表限制；和 fused `halfword` 比较接口融合成本。 |
| `unfused_h3` | Rust 新对照 | H3 corpus 上运行拆开的查询接口，受 u16 字母表限制；和 fused `h3` 比较接口融合成本。 |
| `separate_counted` | Rust 新对照 | Separate pair index + counted/filter 初始化；与 `combined_filtered` 共用计数逻辑，隔离组合 hash 表的影响。 |
| `linked12` | `backend_linked.py:FastCompactLinkedBackend` | val/prev/next 三个 u32 逻辑槽；Rust prev/next links 当前为 signed i32，长度上限约 2^31 positions。不是 YTTM 的 run-length encoding。 |
| `linked16` | `backend_linked.py:FastLinkedBackend` | 与 linked12 相同链表，另有每位置 run-length 槽；Rust link 上限仍为约 2^31 positions。RLE 槽在此实验始终为 1/0。 |
| `bitmap_u32` | `backends_compact.py:FastPrezzaBitmap` | u32 ID corpus + live bitmap + 64 位块间 skip；邻居查找与端点不同。 |
| `halfword` | `FastPrezzaHalfword` | 初始 ID 使用 u16，fresh u32 ID 占两个 u16；bitmap/skip 仍在。初始字母表须小于 65536。 |
| `h3` | `hybrid_backends.py:HybridByteTags` / `FusedHybridByteTags` | u16 文本 + 每位置一个方向 tag；H3 与 fused inspect 路径要能分开识别。初始字母表须小于 65536。 |
| `h25` | `HybridNibbleTags` | u16 文本 + 每字节两个方向 tag；低字节量不代表低总 RSS，必须完整测量。 |
| `filtered` | `evolution/filtered_driver.py` | 初始计频后只为能达到最低频率的旧 pair 建 occurrence 数组；旧 pair 频率只降，新 pair 必须在本轮累加完成后筛选。 |
| `arena` | `evolution/arena_driver.py` | pair state + 可复用 `(position,next)` occurrence 池，先为初始 pair 建 state，再丢弃稀有 state。 |
| `arena_counted` | `evolution/arena_counted_driver.py` | 与 arena 相同 merge 路径，先计频再为合格 pair 建 state；需要分开看初始化峰值与常驻内存。 |
| `filtered_h3` | `evolution/filtered_driver.py` + `FusedHybridByteTags` | occurrence 预筛与 H3 组合；只能和两个单因素版本及 baseline 对照，不能把组合差异归于单一技术。 |
| `combined` | Rust 新探索 | occurrence/frequency 状态放进一个 hash 表，Python 没有直接等价实现；当作新的原生布局，不写成 Python 结果的复刻。 |
| `combined_filtered` | Rust 新探索 | combined 状态加低频预筛；同时报告临时计频表和最终表的峰值代价。 |
| `certified_prefix_probe` | 后续原生语义诊断 | 使用 combined_filtered 存储，仅预取经证书允许的连续规则前缀，仍逐条串行应用；独立记录批宽，不加入第一轮性能 Pareto。 |
| `combined_filtered_h3` | Rust 新探索 | 先筛稀有 pair，再组合 H3 压缩 corpus；受 u16 初始字母表限制。单独量化小 occurrence index 是否能使更紧凑 corpus 获得净收益。 |
| `combined_filtered_halfword` | Rust 新探索 | 先筛稀有 pair，再使用 halfword corpus；受 u16 初始字母表限制。与 `combined_filtered_h3` 分开，避免把 bitmap/tag 与索引筛选混为单因素。 |
| `bucket` | `python_rewrite/queues.py:HighLowQueue` | `floor(sqrt(weighted initial mass))` 高扫描区/低频桶队列；不等同 YouTokenToMe 的完整 run/队列/worker 算法。Rust 桶数上限为 `max(1024, min(2N, 1,048,576))`；超过上限必须显式 fallback heap 并在 metrics 标出。 |
| `bucket_normalized` | Rust 新探索 | 用 piece-weight GCD 归一化队列分数以避免统一放大权重使平方根扫描退化；规则输出仍保留原始精确频次。验证 gcd=1/大权重时 fallback 与完全相同的 trace。 |
| `parallel_broadcast` | `evolution/parallel_driver.py` | piece 所有权 + 每规则广播 + 全局队列/barrier；消息数与同步开销必须报告。 |
| `parallel_owner` | `evolution/parallel_sparse_driver.py` | owner mask 减少不相关分片通知；owner 目录成本和分片不均衡可能抵消少发的消息。 |
| `parallel_occurrence` | 新 Rust 路线 | 第一版使用串行稀疏 overlay 规划，不是永久 occurrence owner 路由；比较时计入 overlay hash 查询/更新成本。 |
| `parallel_occurrence_snapshot` | 新 Rust 路线 | 无永久边界表；每轮稳定快照并行验证 occurrence，再按确定性顺序应用规则。需与永久 occurrence 路线分开比较 snapshot、run 匹配修正和临时状态成本。 |
| `parallel_occurrence_adaptive` | 新 Rust 路线 | 按每 worker 的历史 position 数自适应：小轮由 coordinator 精确更新，大轮使用 snapshot 与两个 barrier；阈值 1024。worker=1 走 direct-local。 |
| `parallel_occurrence_adaptive_256` | 新 Rust 路线 | 与 adaptive 路径相同，阈值改为 256 historical positions/worker；只用于阈值敏感性分析。 |
| `parallel_occurrence_adaptive_4096` | 新 Rust 路线 | 与 adaptive 路径相同，阈值改为 4096 historical positions/worker；只用于阈值敏感性分析。 |
| `parallel_serial` | Python `serial1` 调度对照 | 同一分片/聚合流程但 worker 顺序执行，用来测调度框架成本；另有直接 `packed` 单进程对照。 |
| `bytespans` / u8 topology | `evolution/byte_spans.py` | 一字节边界组件，只有 topology，不存 token ID，不等于 BPE 后端；由独立 CLI micro 校验和计时，不放入完整 trainer Pareto 表。 |

Rust CLI 可为 BPE variants 选择 checked/unchecked；具体变体是否因此走不同代码路径要按实现和指标确认，不能假定所有变体都有不同的 unchecked 路径。差分仍分别传入两种模式核对语义；报告保留 bounds，但不得笼统称“所有变体都验证了 unsafe 路径”。Compact/u16 布局遇到超过 65535 的初始字母表时，可以只对该明确限制跳过；同一个变体仍须在受支持输入上完成完整 trace 比较。

## Fixture 维度

`ablation_fixtures.py` 复用 `rust/results/fixtures.json` 的 11 个旧 prepared fixture，并把原文件逐字节 hash 校验后引用，不改旧数据。新增 fixture 放在 ignored 的 `rust/fixtures/ablation/`，清单放在 `rust/ablation_results/fixtures.json`；脚本重跑时验证现存 fixture 内容，不覆盖变化的快照。

| 维度 | 覆盖输入 |
|---|---|
| 语言、规模、字母表与真实重复分布 | en/zh/de/ja 约 1 MiB regex，en paragraph，en/zh 约 4 MiB regex；旧 11 个 snapshot 全保留。 |
| 重复度与输入结构 | random-131072、runs-131072、单一 `a` run、单 piece `ab` 交替、混合长/短/稀有 pair、512 个 singleton bigram 加重复高频 pair。 |
| merge 长链及边界 | chain-2k/4k 与旧 chain-8k/16k；run 产生超过 255、65535 的 token 长度。 |
| 权重与队列退化 | en-1m 固定 prepared 权重×64、最低频率也×64；另有 differential 的 `2^40` 权重，检查精确 u64 和 bucket fallback / normalized。 |
| 分片与无预分词输入 | en/zh 全文各作为一个完整 piece，保留空白和标点作为普通符号；multi-piece 4 MiB 与 paragraph 作对照。它仍是 Python 预构造的 numeric `Prepared` 输入，不计 tokenizer/preprocessing 时间。 |
| piece 数与极端长 piece | 单 piece run/alternating 及多 piece 真实语料；检查 worker 数请求值、实际 shard 数和最大 shard 工作量。 |

这些 fixture 是覆盖轴，不是“sweetspot”参数的单一目标。应在多个规模和结构上呈现每个版本的时间/RSS Pareto 及差异落入五次重复范围时的不可区分项。另可探索流式预处理以降低原文/JSON 的峰值，但 mmap 本身不代表 bounded-memory：prepared corpus、pair counts、occurrence index、heap 仍可能随输入增长。streaming 必须作为独立 pipeline 口径报告。

## 运行口径与步骤

准备 fixture、跑小规模 differential 和开始测量的脚本：

```sh
python3 rust/tools/ablation_fixtures.py
python3 rust/tools/ablation_differential.py --binary rust/target/release/ablation
python3 rust/tools/ablation_benchmark.py --profile sweetspot --repeats 5 \
  --output rust/ablation_results/sweetspot-<run-id>.jsonl
python3 rust/tools/ablation_benchmark.py --profile multicore --repeats 5 \
  --output rust/ablation_results/multicore-<run-id>.jsonl
python3 rust/tools/ablation_micro.py --output rust/ablation_results/topology-<run-id>.jsonl
python3 rust/tools/ablation_summarize.py \
  --inputs rust/ablation_results/run-a.jsonl rust/ablation_results/run-b.jsonl \
  --output-prefix rust/ablation_results/summary-<run-id>
```

benchmark 每个 job 用新进程、固定种子随机交错、默认五次重复；默认 scalar 固定在 CPU 5，parallel 使用当次可见完整 affinity（本 VM 为 0–5）。输出与 companion environment 使用 exclusive create，已有归档绝不覆盖。环境摘要记录 binary、Rust source/Cargo、依赖的 Python 参考源码 hash、rustc/cargo、CPU 型号、逻辑 CPU 数、affinity、NUMA 节点、profiling 状态及 fixture hashes。无 profiling 的 release binary 是唯一可比较输入。固定条件不代表在当前 VM 完成双路 Xeon 或多 NUMA 验收。

`sweetspot` 跨语言、规模、paragraph、权重、链长和稀有 pair，默认跑单进程算法；`multicore` 对 en/zh 多 piece 与连续单 piece 试验 `packed`、broadcast、owner、occurrence、snapshot、adaptive 三阈值和 serial，worker 数为 1/2/4。`legacy` 只选旧 11 fixture，`stress` 选权重/长链/稀有及长 run，`all` 选所有 fixture 和 scalar + parallel 变体。`--variants`、`--workers`、`--bounds`、`--rules`、`--min-frequency` 可缩小或扩展矩阵；`--cases` 可按 manifest 的 case ID 选择小型 pilot，不改或重写 manifest。缩窄矩阵时必须把配置和结果一同归档，不把它当完整覆盖结论。

`ablation_micro.py` 驱动独立 `boundary_micro` CLI，覆盖 `u8_only`、`byte_spans` 与各现存 topology 的随机、均衡及长链模式，并扫描长度 63/64、255/256、65535/65536。`u8_only` 遇到长度大于 255 时写入带原因的 skip row。CLI 测量字段名是 `seconds`，runner 同时保留它并复制到规范列 `train_seconds`。该 micro trace 在 CLI 内要求相同的 expected checksum；runner 也核对同一 `(length, positions, pattern)` 的所有可运行布局和 bounds checksum。它只测边界元数据组件，`buffer_bytes` 是实现报告值；单独记录 HWM、缓冲 capacity、操作数和 checksum，不将其换算成完整 trainer 的每 token 内存或完整 BPE 速度结论。可用 `--lengths`、`--positions`、`--patterns`、`--variants`、`--bounds both` 和 `--repeats` 缩小/扩展矩阵。

`ablation_summarize.py` 只读 raw JSONL 和相邻 environment metadata，按 `(case, variant, bounds, requested workers)` 汇总每字段 min/median/max，并对所有数值 metrics 都做同样汇总。它先拒绝同一 case 混入不同 `(rules, min_frequency)` 请求，再按训练请求 `(case, rules, min_frequency)` 检查跨文件 fingerprint；随后按文件 metadata 验证声明重复次数及 binary/fixture provenance。它不会拿当前源码 hash 去拒绝旧 pilot。正式数据要求至少两次重复；单次试跑要显式写 `--allow-pilot n1`。输出 `.json` 和 `.md` 新归档，均拒绝覆盖。Pareto 在每个 case / bounds / worker 数内分别列 scalar、parallel 和合并 fronts；不会跨 case 求均值。Speedup 使用 median，packed 尽量同 bounds，不存在时标明 checked fallback；并行强扩展和相对直接 packed 的加速分开给出。运行统计不足时输出 `null`，不补造 worker 指标。

验收顺序：

1. **清单完整性：** 每个 fixture 的文件 hash 与 manifest 一致；输入 raw hash、source 和分词/piece 口径清楚。旧 11 fixture 不重写。
2. **单机语义：** 小随机、`aaa` 重叠、多个 piece、空白标点单 piece、非均匀权重、低频阈值及大权重的 full-recount oracle，对每个受支持变体核对完整 `(left,right,frequency)` trace 和最终 IDs。高字母表只在明确 compact 限制下跳过，记录错误类型和受影响组合。
3. **并行语义：** parallel 1/2/4 workers 与 serial/packed 的所有规则和最终序列逐字节一致；改变 piece shard 顺序和 worker 调度仍需保持确定性。报告请求/实际 worker、空闲轮、每轮消息、最大 shard、barrier/归并 CPU 与全进程 CPU。
4. **边界组件：** Bytespans 独立 topology oracle 检查 63/64、255/256、65535/65536、payload 覆盖和邻居边界；不将其结果与完整 tokenizer 的每位置字节直接比较。
5. **速度/内存：** 至少五次新进程用于正式候选，记录 train CPU、墙钟、RSS/PSS 与 N/E/occurrence/heap 指标；汇总中位数、min/max、Pareto 和极端退化。比值改善小于运行范围则描述为持平或未分辨。不同结构之间不得只挑一项最快值或一个固定预算的唯一赢家。

## 当前环境范围与待扩展

当前执行容器是 6 vCPU、一个可见 NUMA 域；CPU 5 仅作为 scalar pin，parallel job 允许 0–5。双路 Xeon 的物理核心数、NUMA 拓扑、内存通道与频率会改变并行甜蜜点，后续应在目标服务器独立保存环境 hash，并扫描 1/2/4/6/更多 worker，而不是将本 VM 的 4-worker 结果写成硬件结论。

33 个训练版本的第一轮正式矩阵已完成，具体覆盖与结果见 `ABLATION_REPORT.md`。ByteSpans 仍是独立边界组件，尚未集成完整 trainer；并行语义通过不代表扩展性成功。随后增加的 `certified_prefix_probe` 是单独验证的串行批宽探针，不属于这轮固定二进制的性能矩阵。
