# 内联出生区域计数

状态：从已冻结 `owned_replay_birth` 克隆的独立原型。30 项 Rust lib 测试、fmt、strict Clippy、release、80 次标准和 16 次定向独立完整轨迹对照已通过。性能只评价随附的 26 调用小测；控制与实验同一二进制。

`--birth-fill chain|replay|replay-inline` 默认 chain。chain 保留原出生链；replay 使用原 `Vec<(usize,u32)>` 计数；replay-inline 仅更换每个 CombinedDelta/StagedBirth 的区域计数容器。两个 replay 模式仍要求 global posting、tagged-fused、region；tagged 域回退时均整次回到 chain。所有合并、频次、边界例外、最终 posting 的安全切片填充协议与重放原型相同。

`InlineCounts` 对 0、1、2 段不分配堆数组，第三段才分配并搬迁前两段。每段改为两个完整 u32：region 与 count。验证保证 T≤N≤u32::MAX，容器自身仍检查 region 的 usize→u32 转换；count 不窄化。保留插入 region 的升序，消费迭代器无需排序或新 Vec。无新增 unsafe。泛型实例分别保持原 Vec 控制的对象布局和 inline 布局，没有为控制加 enum 包装。

此机原 Vec 头 24 B，InlineCounts enum 头 32 B。Vec 首次通常分配四个 16 B 段，inline 的三段及以后分配四个 8 B 段。一到两段的分配省去，旧 key 的空 CombinedDelta 与新 key 的 staging 头却都增加 8 B。这是实际折衷，不能把计数 heap 字节减少直接等同于 RSS 下降。描述符哈希表、第二次 selected posting 扫描、最终 posting 预零、延长 selected 存活期仍存在。

`replay_count_single_segments/double_segments/many_segments` 累计统计每批达阈值的新 pair 的区域段数。每个 fresh pair 只在其创建批次出生；这些 pair 数不是最终词表数。`replay_count_peak_heap_bytes` 是某批达阈值 staging 的计数 heap 容量峰值，不含 CombinedDelta 的所有过滤前计数。`replay_count_header_bytes` 是容器类型大小。原 replay 的 staging 和同时存活容量代理也按实际泛型类型重算，不含 allocator/HashMap 桶等，仍不是 RSS。

消融必须同时看 chain、原 replay 与 replay-inline。若 inline 只改善 replay 而仍比 chain 慢，就不应把 replay 作为通用默认。单核与四核扩展性仅对同窗口存在的 inline 测量计算；不用旧测量拼出其他模式的加速比。输入没有预分词空格假设，长度表保持 u32，W>N 与切点穿过长 token 保留精确处理。
