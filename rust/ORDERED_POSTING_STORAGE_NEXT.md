# 有序 posting 的压缩存储方向（未实现）

前提是 [全局有序 posting 协议](ORDERED_POSTING_REVIEW.md) 通过完整轨迹验证：初始 region 扫描按位置升序，region 结果按物理顺序连接；每个出生 key 的有效位置只由一条选中规则的一个 birth 方向产生；跨区 left birth 注入目标 region 的末尾；owner 把每个 `(region,key)` 反向 birth 链刚追加的片段局部反转。AA 的相邻 `(Z,Z)` 由左匹配的 right birth 唯一生成，右匹配的 left birth 被抑制。这样一个 pair 的所有历史位置严格升序，旧 key 此后只失去出现，不再增加 posting。若实际执行器改回按动态 worker 完成顺序拼接，不能沿用该结论。

一种完整 `u32` 位置域的表示是首项绝对位置 `u32`，随后将正 gap `1..=65535` 记为 `u16`，更大 gap 记为 `u16` 零哨兵加一个 `u32` gap。严格升序保证零不是合法 gap。若共有 `H` 项、物理语料跨度小于 `N<2^32`、大 gap 数为 `E`，则 `E≤⌊(N−1)/65536⌋`，编码流约为 `4+2(H−1)+4E` 字节。`H≥N/16384` 时，流本身至多约 `3H+O(1)` 字节，对比原 `4H` 位置 payload 有至少约四分之一的理论节省。这个界**不包括**容器头、checkpoint、容量余量、allocator、birth routes 和转换时的同时存活量；应同时用密度门槛与实际分配字节 guard 决定是否压缩，不符合则沿用 `u32`。`H≤2` 保留现有 `SmallPosting` 的两项内联；较长但小或稀疏的列表也保持原布局。

随机 cut 定位不能对压缩字节流做普通数组二分。每 `K` 项保存其绝对位置与编码流 byte offset（项号可由 checkpoint 序号推得），先二分 checkpoint，再顺序解码最多 `K` 个 gap；单 cut 成本 `O(log(H/K)+K)`。按 posting offset 分 flat task 时，尽量在 checkpoint 边界切块，避免每任务额外重复解码 `K` 项。checkpoint 的实际 byte offset、条目大小和任何对齐空间都须计入 guard。`N<2^32` 使绝对位置与 gap 可用 `u32`，但编码流 byte offset 的范围应单独检查，不可无证明缩成 `u32`。

现有 `SmallPosting` 在 64-bit 平台为 16 B：`capacity=0` 表示内联，`capacity≥4` 表示独占 `Vec<u32>` raw parts；值 `1/2/3` 尚未用于 heap，可作为私有压缩 tag 候选。这要求修改全部 `Drop`、`as_slice`/迭代、`take`、`push` 和 AA 访问接口。压缩指针必须指向能恢复真实 allocator `Layout` 的 owned header；例如指向含实际编码缓冲/检查点所有权的盒装对象，而不是把一个 `Box<[u8]>` 胖指针假装成单个 8 B 指针。也不能把 `Vec<u8>` 的地址当作 `Vec<u32>` raw parts 释放：长度、容量和对齐布局均不同。若新增 tag 使 `Entry` 从 24 B 膨胀，须把每 key 的固定增量与 payload 节省相抵，而不是只报告编码流长度。

出生阶段是内存结论的难点。先构造完整 `Vec<u32>` posting，再压成另一个 blob，会在转换时同时持有两份，并不能证明训练峰值下降。可先从已按 region 排序的 birth 链计数、计算 gap/escape/checkpoint 长度，再直接分配与填充压缩对象；这多一次读取链，仍需局部链校验、最终 count 校验和错误时安全析构。另一条较简单的实验路径是先只对初始有序 posting 使用压缩，单独量初始化峰值，再扩展出生。无论哪条路径，不能因编码流界忽略当前 `BirthNode` 与 route map 的在途分配。

有序历史表还能让 AA 去掉最前排序，按位置流式验证 run/parity；若并行分区，仍需每块 parity 摘要与跨块前缀，不能各块独立从偶数开始。磁盘上的有序 extent 未来可支持 cut 定位和顺序读取，但频率/heap、owner 字典、route 以及出生写入仍需单独定预算。这份文档只定义后续可测的存储接口，不把有序位置表等同于已完成的外存 BPE。
