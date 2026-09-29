# Region 边界快照与普通 `u32` 语料审查

结论：这是一个**附条件可证明的算法协议**，不是把现有
[`inspect_fused`](experiments/radical/owned_fused_endpoint/src/lib.rs) 的
`AtomicU32` 改成 `u32` 即可。若每个物理 region 在一批中只有一个执行者，
远端读完全由批前边界描述符回答，越界写到所有任务 join 后才回放，
则各任务只借用不相交的 `&mut [u32]`，语料不需要共享原子格。
非 AA 的精确 token-disjoint 证书和旧 ID 不复生性质是语义证明的一部分；
AA 必须保留先全局 parity/规划、后写入的阶段。当前尚无实现或计时证据。

## 每个 cut 的常数状态

对每条内部 cut `c`，保留覆盖物理位置 `c` 的**批前活 token** 起点 `h(c)`；
若 `c` 为 piece sentinel，则记为不可合并的边界。初始 token 长度为 1，
所以 anchor 可直接建立。每批开始、尚未拆分可变语料时，由每个已知
anchor 的端点 ID 与 `lengths[id]`，按 token 而非物理位置跳转，复制
anchor 前至多两个和后至多三个活 token 的 `(head,id,length)` 描述，
遇 sentinel 即止。长 token 可覆盖许多 cut；对每条 cut 仍只需常数个
描述符，不扫描到任意远处的 head。描述符是自有数值，不得持有指向
即将可变借用的 corpus 引用。

这个半径来自实际查询位置，而不是经验缓存大小。一个本区候选起点 `p`
先检查本地格；若它仍是批前旧 A，则检查 `q=p+len[A]`。`q` 若在右区，
紧邻本区的上 cut 必在 A 内或恰在 B 起点。对有效旧 `(A,B)`，再向左
至多需要 L、K，向右至多需要 C、D。若只有更远的 C 或 D 端点越界，
上 cut 也可在 B/C 内或恰在 C/D 起点；覆盖 token 因而可能是
A、B、C、D 中任一个。从它向前 2、向后 3（总计最多六个 token，
不是三个物理格或总共三个 token）足以得到**越界的** `q`、`after`
或 C 的后继 `u`；其余 K,L,A,B,C,D 邻居由本地切片或下 cut 窗口给出。左区查询
由紧邻下 cut 的前 2 个 token
覆盖。远端位置虽可能跨越多个空活起点 region，仍是这几个 token 的
已知 head、tail 或 next 坐标。窗口应取**本 region 出境的** lower
或 upper cut，不可按远端目标 region 任取一个 cut。查询接口应断言
每个远端地址与描述符中的某个**允许的旧端点**相符；不能在失配时
退回跨区直接读。

历史 posting 的 stale 起点不引入任意远端探测：其 `p` 属本区，先读
本地 `p`。在批前稳定端点表示中，若 posting 的旧 A 仍在 `p`，这个
曾为 token 起点的位置尚是活 A 起点；吞掉它的合并只写 0 或更大的
fresh ID，不会写回旧 A。批中要再加一层论证：越界写可能尚未回放，
物理旧 A 不一定仍是**当前逻辑活 A**。但当前选中非 AA `(A,B)` 的
这个 A 若被另一匹配吞为右 token，那条规则必为 `(K,A)`，与 `(A,B)`
触犯类型冲突证书；同一非 AA `(A,B)` 的不同有效出现也不重叠。
因此这个规则的候选在自身发布前不会因延迟写而误收。`q` 的 B
也不能被不同选中规则 `(B,C)` 吞为左 token；若它是另一匹配的右
token，其唯一批前前驱就是当前 A，便是当前这次 `(A,B)`。AA 的
重叠出现不满足此论证，不能走非 AA 单步融合检查。

## 本地现值与远端旧值的解码

本区任务会在规划另一个出现前写过自己的端点，故本区端点读取要保留
`old_end_id`、`old_next_id`、fresh HEAD/裸尾规则。远端只返回批前
旧 token 描述，并借助 selected-key→new-ID 表判断相邻 K,L 或 C,D
是否在本批选中。对于右侧 C,D：若 `t`、`u` 在本区，可沿用本地
HEAD/清零重读；若 `u` 越界，旧描述符给出 D，直接按 selected map
计算最终右 ID。若 `t` 已越界，`u` 更不可能回到本区，两者均读批前
快照。左侧同理。同区唯一写者使本区 `t/u` 不会在一次邻居读取之间
被另一任务改写；`u` 越界时它永远来自旧快照。因此原子版的
zero→Acquire 重读不能照搬成跨区发布协议。若提前回放远端 clear，
这项分类就失效。任务须在
所有选中规则已固定后才能解码；selected map 只证明非 AA 的有效
批前相邻出现，不证明 AA 的某个重叠候选被 parity 选中。

## 延迟写、anchor 更新与空间界

匹配起点归属本 region，head 写始终本地。若右 token 起点或合并尾端
在别区，只排队 `{absolute_pos,value}`；在所有 region 任务 join 后，
按本匹配原本的右起点、尾端次序回放，再做 owner commit 和下一 epoch。
选中匹配的两个旧 token 跨度互不交叠。每个含越界端点的匹配至少穿过
一条 cut，而固定 cut 至多落在一个选中跨度内；映到最左穿过的 cut
为单射，所以跨区匹配数至多 `T−1`，越界 endpoint store 至多
`2(T−1)`。即使一个 token 跨很多 cut，仍仅对应一个匹配和至多两次
store。队列载荷 O(T)，协调者可按绝对位置直接回放 O(T) 次写；若选择
先按目标 region 分桶，二分定位另需 O(T log T)。左 birth 的跨区
例外按[region 提案](REGION_ORDERED_FUSION_NEXT.md)另计至多 O(T)；
每 region 到 key-owner 的空 route 头仍是 O(TW)。join 后由协调者顺序
回放这 O(T) 条写即可，不需要另建 producer×owner 的跨写队列。

回放完成后才能更新 anchor，也才能选下一批或建下一批边界快照。
具体反例是上批邻区匹配 `(B,C)` 已逻辑吞掉本区 B，但清零 B 起点的
跨区写仍排队；若此时先拍快照，下一批历史 posting `p=A,q=B` 可把
已消失的 B 当成活邻居。同批 token 类型证书不保护跨批错误。
若旧 `h(c)` 格仍带 HEAD，它是批后覆盖
`c` 的 token 起点（本批可能作为左 token 改成 fresh head）；若它变成
0 或 fresh 裸尾，它在唯一匹配中作为右 token 被吞，批后覆盖 `c`
的 head 是**批前存下的直接前驱 head**。该前驱不能同时被另一匹配
吞掉，因为选中跨度不重叠。piece sentinel 固定不变。这里必须用旧
前驱，不能从被改写后的 `h-1` 倒找。AA 在全局 parity 确定所选非重叠
跨度、所有端点写和队列回放完成后，遵循同一 anchor 更新；不能按
所有有效 AA 候选更新。若 tagged ID 域检查失败，HEAD 判据不可用；
这版协议应整调用回退既有 full-width/两阶段实现，或另设计显式
selected-span anchor 更新，不能暗中改用裸 ID。

## Rust 所有权与适用边界

先从完整 corpus 建立批前数值快照，再用 `split_at_mut` 按严格递增的
cuts 拆出互不相交的 region `&mut [u32]`。每个 slice 只交一个逻辑
region 任务；Rayon 可动态分派这些任务，但同一 region 内不可再并发
写端点。任务只共享 immutable posting、规则和边界数值，不保留完整
corpus 的 `&[u32]` 或跨 slice 原始指针。全部任务 join 并释放 slice
借用后，单独回放跨区写；回放再 join 后才建立下一批快照或读最终
语料。任何规划、route 或队列错误都须终止并丢弃私有训练状态，不能
在半提交语料上续跑。AA 可先借整个不可变 corpus 完成 inspect、
全局 parity 和 Plan，再拆 slice 写并延迟越界端点；这保留 AA 的
两阶段内存成本，但不需要 AA 的原子语料。严格 cuts 允许 region
没有活 token 起点；anchor 仍由覆盖该 cut 的长 token 或 sentinel
给出。严格 cuts 可取 `T=min(W,N)`，避免零长度物理 slice；允许重复
cuts 的现有 region 原型属于不同协议，不能直接套用此 anchor 证明。

去除原子类型的价值是明确的独占所有权及后续存储布局可能性，**尚不能
推断 x86 有净速度收益**：边界快照、位置判断和延迟写可能抵消收益。
[Rustonomicon 的原子章节](https://doc.rust-lang.org/nomicon/atomics.html#hardware-reordering)
指出强序硬件上的额外顺序要求可能很便宜；其
[Acquire/Release 说明](https://doc.rust-lang.org/nomicon/atomics.html#acquire-release)
也称这类访问在强序平台常可免费。这些硬件事实不替代上述无数据竞争
的 Rust 借用证明。

即使边界协议成立，它只建立语料切片的独占访问，并未实现外存 BPE。
唯一 posting、owner HashMap、heap 和 birth routes 仍可能按语料规模
O(N) 留在内存。增大 T 还继续支付 O(TW) 路由头，以及每批约
`2BT` 次二分的 `O(BT log Hmax)` 定位工作。tagged 域不满足时可整调用
保留既有 32 位原子/两阶段实现，无须为了这一路径缩小可训练 ID 域。

首个验证模型应逐步枚举 `A,B,C,D` 与 `K,L,A,B` 跨 cut 的位置，允许
长 A/B/L/C 跨多条 cut，并分别让相邻匹配先后由本区或别区处理；检查
旧邻居、最终新邻居、每个写位置和 anchor。再覆盖 cut 恰在 token
起点、只含长 token 内部的 region、piece sentinel、stale posting
起点、AA 连续 run 的奇偶与跳过候选、31 位域回退。任何合法出现请求
描述符窗口外远端地址，或一条 cut 对应两次选中跨区匹配，都直接推翻
这里的常数窗口或 O(T) 队列证明，不应以全语料远端读取掩盖。
