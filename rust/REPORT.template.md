# Rust 基线实测（2026-09-29）

本轮固定一个算法，比较 Python rewrite、Rust checked、Rust unchecked。它回答“迁移后的原生起点有多快，以及局部 unsafe 是否值得”，不据此判断相对于其他原生 BPE 库的新颖性或速度排名。与 Re-Pair、YTTM 等工作的关系，见[前期调研](../benchmarks/research_archive/research.md)和[共同 Python 比较](../benchmarks/bpe_core_comparison/REPORT.md)。

七组真实文本上，Rust checked 的核心中位耗时约为对应 Python rewrite 的 **1/12–1/20**。unchecked 的测量范围与 checked 重叠，没有显示稳定收益，默认入口仍采用 checked。这个数量级的提升建立了可用的原生实验起点，本身主要反映语言及表示差异，不是新的算法突破证据。

## 比较口径

Rust 与 Python 接收同一份数字化、去重加权后的输入。预切分由已有 Python 程序统一完成；下载、解码、regex、JSON 读取均不计入训练核心。相同全新 ID、最高加权频率、字典序同频选择、重叠计数及从左到右替换。参考 Python 是已优化的 `packed_driver + LeanEndpoints`，并非直接计时原项目或 Hugging Face。

机器为 Xeon Gold 6140；使用可用 CPU 集合中的 CPU 5，顺序启动进程，每组五次重复，变体与输入随机交错。CPython 3.12.13，Rust 1.98.1，release + thin LTO + 单 codegen unit；未启用 hotpath。标准库 HashMap 使用默认随机种子，不改为专门针对整数的 hash 算法。原始环境、二进制和源码 SHA-256 见 [baseline.environment.json](results/baseline.environment.json)。

Python 参考实现的父类及准备流程补充哈希见 [reference-source-hashes.json](results/reference-source-hashes.json)；这些文件均与此前已推送的研究提交 `ac7fd6a` 逐字一致。运行器本身在整个矩阵期间保持不变。

核心耗时 `train_seconds` 只覆盖初始化和训练循环，两边都不含最终 token 提取及指纹计算。Rust 公开 API 的输入验证也在核心计时之外，原始结果另有 `call_seconds` / `call_cpu_seconds` 包含验证和提取；Python 的 call 还包含指纹计算，因此不把两个 call 值直接当作同口径算法速度。Rust 接管输入 Vec，Python 后端复制输入 array；此所有权差异包含在初始化成本内。

## 核心耗时

单位为秒，中位数。最后一列大于 1 表示 unchecked 更快。`1m/4m` 是输入 UTF-8 文件的大致字节规模，实际语料位置数因去重及切分不同，见 [fixtures.json](results/fixtures.json)。

| 输入 / 切分 | Python | Rust checked | Rust unchecked | Python / checked | checked / unchecked |
|---|---:|---:|---:|---:|---:|
{{TIMING}}

初始化与合并阶段的 checked 中位数，以及两个 Rust 版本五次运行的最小值–最大值：

| 输入 / 切分 | 初始化秒 | 合并秒 | checked 核心范围秒 | unchecked 核心范围秒 |
|---|---:|---:|---:|---:|
{{PHASES}}

短运行、随机 hash seed 和机器调度都会影响常数；两个版本的差距若落在重复测量波动内，就不视为稳定收益。局部 unchecked 只改变端点数组访问，其余 hash、堆、分配及权重查找成本完全保留。后续优化优先依据本地剖析和多组输入，不能假定 unsafe 必然更快。

## 内存

下表为 Linux `/proc/self/status` 的 `VmHWM`，MiB，中位数，包含该进程的输入解析、训练、结果提取和指纹构造。它不是训练阶段的活跃堆大小，尤其 Python JSON 解析产生的临时整数/列表也会抬高峰值。原始结果保留 `getrusage` 的 `peak_rss_mib` 作为诊断字段；不使用它推断小进程的精确占用，以避免启动链历史峰值的影响。

| 输入 / 切分 | Python VmHWM | checked VmHWM | unchecked VmHWM |
|---|---:|---:|---:|
{{MEMORY}}

明确可计算的逻辑存储为端点 **4N** 字节、初始 occurrence 元素 **4E** 字节；另有 Vec 预留容量、每种 pair 的 Vec 头部、hash 表、频率、堆、长度表、权重段及结果。Rust 输入 Vec 移入后端，Python 保留输入 array 并建立副本。因此该表同时体现语言表示、分配策略、解析流程和所有权差异，不能把全部内存下降归功于新的 BPE 算法。`backend_buffer_bytes` / `initial_occurrence_bytes` 统计元素长度乘宽度，不包含预留容量，也不是总 RSS。

## 正确性及边界

Rust 单元测试覆盖独立朴素训练器、120 组随机输入、自重叠与历史位置、65,536 初始 ID、65,536 长 token，以及非法输入和 `u64::MAX` 权重边界。两种 bounds 模式共用公开验证。跨语言测试另做 205 组输入、410 个 Rust 进程，与 Python 全量重新计数 oracle 比较每条规则、最终 token 和指纹；其中包含重复/不去重输入、不同阈值及放大到 `2^40` 级别的权重。

正式性能矩阵共 **165 个训练进程、11 组输入、每变体五次重复**。所有规则及最终 token 指纹一致，规则数、实际替换数、位置访问/失效数、堆弹出数也一致。校验文件见 [differential.json](results/differential.json)、[verification.json](results/verification.json)；逐次结果见 [baseline.jsonl](results/baseline.jsonl)，统计范围见 [summary.json](results/summary.json)。

独立的 hotpath 构建还完成了三个真实输入的剖析，语义指纹与正式矩阵相同。中文 4m 的初始化明显值得优化，英文 paragraph 主要花在规则应用；计时范围和原始报告见 [HOTPATH.md](HOTPATH.md)。该粒度尚不足以断定 hash、分配或权重查找哪项最贵。

unchecked 经独立代码审查，其安全性依赖私有后端、输入验证、每轮全新 ID，以及历史 occurrence 只来源于当时的活跃起点。随机差分和审查不是形式化内存安全证明；本轮未运行 Miri 或 sanitizer。当前 CLI 仅在 Linux x86_64 实测，32 位长度比较已修正但未做跨平台构建测试。

本轮已去掉 u8 长度上限，使用 u32 长度表。更低逐位置开销的 1N 边界编码、2N 左右压缩布局，以及原生线程分片仍是后续实验，详见 [DESIGN.md](DESIGN.md)。局部 O(1) 端点更新也不等于整个训练器严格线性：权重段二分、hash 操作、历史位置、堆维护和最终词表字符串物化需要分别计入。
