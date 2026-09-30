# 用稳定语料重放出生位置，替代每位置 BirthNode

状态：已实现安全切片控制于 `owned_replay_birth`，27 项 Rust 测试及 96 次独立轨迹门控通过；追加 `owned_replay_counts` 内联计数版，30 项测试及 96 次门控通过。两窗口共 66 次计时含其他切区候选，详见[本轮报告](ADAPTIVE_REPLAY_REPORT.md)。本版只支持 atomic region，不支持 snapshot replay；未实现未初始化 posting 接口。BirthNode 去除成立，总训练 RSS 尚无一致改善。

下文保留反例与原所有权设计。候选基于固定 region、全局有序 posting 和 tagged 端点协议。目标是删除每个新边的 8 字节 `{pos,next}` BirthNode，而不是减少精确频率统计。每批现有规划仍遍历被选 pair 的历史 posting，算旧键减量与新键的 `(weight,occurrences)`；新键增量按**出生位置所属的目标 region**保存。此遍不存新边位置。选中匹配的端点全部写完后，owner 合并所有 region 的频率与出现数、按完整净频率应用阈值，给保留的新键一次分配最终 posting，并按 region/key 计数分配互斥写入段。第二遍重放仍保留的被选历史 posting，将新边位置写到这些段；全部任务 join 且逐段计数吻合后才发布可读长度和开始下一批选择。

## 为什么重放能认出本批实际匹配

规则 i 的 fresh ID `Zi` 只在该规则的实际匹配起点 p 写入带 HEAD 的单元。第二遍对它的每个历史位置要求 **raw `corpus[p] == HEAD|Zi`**；只比较掩码后的 ID 会把长 token 尾端的裸 `Zi` 误认成起点。一个 stale 历史位置不可能由别的规则写成 `Zi`；同一 pair 的物理位置在其唯一出生时只记录一次，旧 pair 以后不会复生，所以也不会因重复记录误认。AA 左到右 parity 未选中的重叠起点只会保持旧 head 或变成零/裸尾，仍不通过。长度 1 的右 constituent 会写裸 `Zi`，同样不通过。整个判断只在 apply、延迟回放与 AA 写入全部 join 后读取稳定语料；31 位 ID 域失败时回退旧路径。

对于通过检查的 p，新的 token 长度是 `lengths[Zi]`。若 `p-1` 是 sentinel，没有左出生；否则从该物理末格低位 ID `L` 与 `lengths[L]` 还原旧/新左邻 head `q=p-lengths[L]`，可在 debug 验证 `q` 带 HEAD 且 `q+lengths[L]=p`。只有 `L<fresh_start` 时生成 `(L,Zi)` 于 q；若 L 是本批 fresh，左匹配已从自己的右出生唯一生成这条 fresh/fresh 边。右邻位置 `r=p+lengths[Zi]` 若为 sentinel 不出生，否则由稳定 head 得到 R，生成 `(Zi,R)` 于 p；R 可以是另一个 fresh ID。AA 相邻匹配的 `(Zi,Zi)` 也只由左匹配右出生写一次。每条边用当前 piece 的 weight，第一遍已决定精确 weighted frequency；第二遍只填位置，并检查每段实际数量与第一遍 `occurrences` 一致。新键低于 min_frequency 时没有写入段，重放跳过它，不能在每个 producer 的局部频率上提前过滤。

跨区左出生的位置 q 在较早的**目标** region；不能按匹配 p 的 producer region 给它分段，否则破坏有序拼接。第一遍已有至多 `T-1` 条跨区例外。可保留这些小例外，在目标 region 的同键段预留最后一格；第二遍各 region 任务只写本区的局部前缀，join 后由协调端把例外 q 写入末格。跨区 q 是目标 region 最后一个批前 live head，且不会同时是本批被合并的左 token，因此仍是最后一个最终新边起点。长 token 跨多 cut 不增加同目标 region 的例外数，piece sentinel 切断跨 piece 边。debug 应检查目标唯一、q 后继已越过目标上界、所有写段恰好填满及最终 posting 严格升序。若第二遍重新计算例外，也必须与第一遍 `(key,q,target)` 对照，不能默默补错计数。

## Rust 所有权和内存界

每个 owner 先完成频率归并和低频判断，给保留新键按 region 顺序做 checked count prefix；各 `(region,key)` 的目标范围互不重叠。per-region 描述符需 key、count、写游标与目标切片，大小 `O(D)`，D 是非空 region/key 组合数；空路由头仍 `O(TW)`。本批被选历史 posting 保留到重放完成，所以与新 Entry 及描述符同时驻留。内存比较应逐批记录旧链 `sum(route.born.capacity())×8` 与实际 `sum(len)×8`、新方案 selected posting 的驻留容量、预填最终 posting 和描述符容量，再比较进程 HWM；`generated_birth_records` 是全调用**累计工作量**，不能乘 8 称为峰值。无需保留 AA Plan 供第二遍使用：AA 写入 join 后也通过自己的历史 posting 与 fresh-head 检查重放，但第一遍 parity/Plan 成本仍存在。

**可先实现的无新增 unsafe 控制**：owner 为每个保留新键以 counted total 建立 SmallPosting，用安全的 `push(0)` 将全部最终槽初始化。然后在独占借用中遍历这些 Entry，取 `as_mut_slice()`，按 region 的精确计数反复 `split_at_mut`，把互不重叠的 `&mut [u32]` 收入每个 region 自己的 key→切片/游标描述符。描述符可作为 `Send` 的可变切片所有权随 Rayon region 任务移动；同一 `(region,key)` 中给跨区例外单独切出末格，由协调端在 join 前后持有并填入，不能和局部任务共享该格。全部任务 join 且每段实际写数吻合后，丢弃描述符的借用，才重新访问 owner 和进入下一批。若在已有 `owners.entries.iter_mut()` 上直接构造，可能每批扫描全部 K 个旧 Entry，产生 `O(RK)` 成本；更合适的实现是把**本批新 Entry**先放在独立 staging Vec/Map，只遍历新键，填完且所有借用结束后再移入正式 owner HashMap 和 heap。移动 SmallPosting 前必须释放指向其 inline 槽的切片，heap 槽也不再被写。这个办法不复制每个出生位置，不需要原始指针或新 unsafe；代价是每槽预零一次、len 在私有重放阶段提前增长，以及 `O(D)` 描述符。若任务返回错误，Rayon 先 join，训练直接 Err；所有槽都是已初始化 u32，SmallPosting 可安全析构，未填的零不会作为成功结果发布。

仅当预零写或提前增长确实成为瓶颈，才考虑更窄的未初始化接口。那时 SmallPosting 的 inline 两槽已初始化，heap 分配来自唯一 Vec，容量由 counted total 检查，空 heap 合法；私有 `NonNull<MaybeUninit<u32>>` 句柄按互斥范围写，owner/staging Entry 与底层分配都不得移动或增长。所有范围写完并核对 count 后才发布 len；错误/展开时 len 保持旧值，部分写入的 u32 无析构，Drop 用旧 len 释放 heap。该方案的 `Send`、provenance 和未初始化读取界都要单独证明，不应因安全控制也能并行填充而预先引入 unsafe。

每个 pair 只被选一次，其历史 posting 在这额外一遍最多再访问一次；额外累计扫描至多所有曾保留初始及出生位置的总量，`O(N)`，而不是每轮全 corpus 扫描。每个实际新边还需一次邻居解码和一次描述符查找，owner prefix 处理 `O(D)` 元数据；两次阶段 join、预零写和较晚释放的 selected posting 是实际代价。目标是删除本批 `route.born` 的 8 字节节点载荷及链索引、逐节点 owner 读取；它的真实节省应与本批链 capacity、selected 驻留、最终 posting 与描述符的**同时峰值**比较。是否改善速度/RSS，必须以同 binary 完整轨迹与 HWM 实测，不能只用累计出生数或载荷上界推断。
