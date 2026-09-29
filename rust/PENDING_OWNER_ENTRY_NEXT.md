# 用尚未发布的 owner Entry 汇总新键

状态：已实现于 [owned_pending_entry](experiments/radical/owned_pending_entry/DESIGN.md)，
14 项 Rust 测试和 264 次完整轨迹对照通过。以下保留原隔离实验设计。
前一个最大 producer 表复用原型减少了汇总入口访问，却没有
一致的完整调用收益。另一个值得独立验证的问题是：新键的最终 owner 表是否
可以直接承担计数，完全免去临时 combined 哈希表？这不是已获优化，也不是
对现有原型的默认改动。

已有 `owned_direct` 能直接对旧键扣减；由于旧键只减不增，低于阈值后可立即
退休。它仍将新键汇总到临时表，再写进 owner 表。新方案只改变这后一部分。
出生键必含本批 fresh ID，不可能与批前 owner 中的键混淆；一个 owner 独占
自己的表，而且在全部 owner 提交完成前不会再次选候选。

首次遇到新键时，直接在 owner 表插入频率为零、posting 为空的 Entry，
并把 key 加入本批 touched Vec。随后在这个 Entry 累加 weighted frequency
与物理 occurrence count。所有 producer 归约结束后，遍历 touched 新键，
删除低于 minimum 的项；为其余项按总 count 分配 posting，然后按原 birth
链填充。所有合格项的长度必须与总 count 相同，完成后才允许下批使用。
heap 中也只发布已完成归约的合格新键。

不必给永久 Entry 新增 count 字段，也不必引入含未初始化指针的新 union
标签：`SmallPosting` 的空 inline 状态已经有两个初始化的 u32 槽、len=0、
capacity=0。可以在私有受检接口中暂用 inline[0] 保存 count；空切片仍为空，
析构也仍按 inline 处理。这个状态只准出现在本次 owner 提交中的新键，分配
前必须恢复为正常 posting。`push`、排序和跨批发布都不得看到临时计数。
异常退出时必须仍能安全析构。用 touched `(key,count)` Vec 保存完成计数后
的校验值；在 64 位平台一般为 16 字节/项，不可当成没有临时空间。

该方案的主要风险是**把不合格键短暂插进长期 owner 表**。它可能触发 owner
表扩容；删除不保证归还底层分配，低频出生键多时，长期内存可能恶化。
公开 `HashMap::capacity()` 在删除时仍可能因 tombstone 变化而下降，不能将它
当作分配字节数，或断言删除前后相等；见[首轮验证纠正](PENDING_OWNER_ENTRY_REVIEW.md)。
原算法临时表虽然多一次哈希，却能在永久索引入表前过滤这些键。这是必须
测量的结构取舍，不能用少了一张 map 宣布内存更低。报告暂存新键数、最终
合格数、owner capacity 增量、touched Vec 容量及完整调用/HWM；尤其需要
大量 singleton 新键和较高 minimum 的反例。若容量污染明显，应停止这条
路线，而不是静默加入每轮 shrink/rebuild 再声称省了哈希。

最小控制应在相同 owner 连续提交内核中比较 combined 与 pending-entry，
保持同一 checked arithmetic、birth 校验、hash、heap 和语料规划器。已有
direct-old 路径应同时启用或同时关闭；不要把旧键绕过 combined 的收益重复
算成新发现。它暂不进入当前 region/endpoint-bitmap 的验证窗口。
