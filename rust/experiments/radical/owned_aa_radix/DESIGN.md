# 原地 AA 位置排序实验

本 crate 复制冻结的 `owned_integer_hash` 作为同 binary 控制，保留
`--integer-hash std|ahash`、owner/route、选批证书、heap policy、SmallPosting、
AA 奇偶规划及完整 JSON/trace 协议。唯一训练核变化是选中 AA 规则后、读取其
posting 位置以前的排序：`--aa-sort std` 调用原来的 Rayon
`par_sort_unstable`；`--aa-sort radix` 使用安全 `slice.swap` 实现原地 MSD
8-bit 分桶。默认 `std`。这不是并行 radix；W4 的其他规划与更新阶段仍照常
并行。

位置是 `u32`，每个大切片先扫描最小值和最大值，找最高变化字节。
若全相等直接返回；若高字节全相同，就跳过它们。对选定字节统计 256 个桶，
构造各桶 `[start,end)`，用游标和交换环将元素放入自己的区间，再逐桶处理
剩余低字节。对于长度不超过固定 64 的切片，调用 `sort_unstable`。这个
阈值不依赖语料或语言。

分桶循环的不变量是每个桶 `[start,cursor)` 已放对，`[cursor,end)` 尚待处理。
当前元素属于本桶就推进本桶游标；否则与目标桶 `cursor` 位置交换，推进目标
游标并重试当前槽。一个已满桶不可能仍有属于它的待处理元素：桶计数正好等于
其物理区间长度，而已处理前缀只含该桶元素。实现只在一个可变切片内使用安全
索引和 `swap`，没有共享原始指针或额外的 H 长度缓冲。

每层至少固定一个不同字节；`u32` 最多四层。计数、边界、游标各有
256 个 `usize`，同时在栈上的这些数组最多
`4 × 3 × 256 × size_of::<usize>()` 字节（64 位环境为 24 KiB），另有
常量栈帧开销。短切片排序的长度受 64 限制。每层对元素做常数次扫描和交换，
因此排序部分最坏 O(H + 256·节点数)，节点数不超过常数倍 H；固定 32 位域下
可写为 O(H)，而原比较排序最坏 O(H log H)。全调用仍含 heap、哈希、
权重二分等其他成本，不能因此宣称训练整体线性。

`aa_sort_positions` 计入被选 AA 的全部历史位置（含 stale）；
`aa_radix_distribution_passes` 计真正分桶的子切片数，`aa_radix_swaps`
计交换数，`aa_radix_fallback_calls` 计短切片调用，
`aa_radix_max_stack_payload_bytes` 是上述三个数组的最大同时载荷估计，
不是进程栈高水位。只在每次 AA 排序外计时，绝不逐元素调用时钟。

此方案可能输给标准排序：标准实现对有序或小数组有优化，原控制路径可用多核，
radix 会多次扫描每层子切片并初始化 256 桶数组。若 AA 子阶段只占完整调用的
小部分，局部改进也不一定有净收益。比较需使用同一可执行文件中的两种模式，
固定 worker 数、hasher、fixture、CPU 亲和、规则参数和完整轨迹指纹。
