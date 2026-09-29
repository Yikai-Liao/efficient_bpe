# 按物理 region 投影排列唯一 posting，执行融合端点规划

状态：已在独立 [owned_region_fused](experiments/radical/owned_region_fused/DESIGN.md) 实现，16 项 Rust 测试、168 次完整 oracle 通过；[8 次轻量小测](batch_results/radical-region-fused-quick-v1/README.md)未见自然语料净速度收益，暂不默认采用。目标是在已验证的 key-owner、唯一 `SmallPosting` 与非 AA tagged-fused 规划器上，使同一段物理语料的邻居读写尽量由同一个逻辑任务处理，同时不复制语料或把每个 key 的索引拆成 T 份。减少跨核通信只是待测假说；现有数据没有证明 false sharing 或远端邻居读是瓶颈。下文保留原设计推导，实际实现允许 W>N 时重复 cut，并报告 AA 重分组额外空间，见[代码审查](REGION_ORDERED_FUSION_REVIEW.md)。

这里复用 [BATCH_OWNERSHIP_DESIGN.md](BATCH_OWNERSHIP_DESIGN.md) 的物理起点分区和「每条 cut 至多一条跨区活边」观察，也借鉴 [ORDERED_BIRTH_SCATTER.md](ORDERED_BIRTH_SCATTER.md) 的按物理顺序建立 posting。**新点较窄**：持久表仍按 pair key 唯一归属，不保留每 key 的 W 份片段或分区目录；posting 只按固定 region 编号排列，region 内任意顺序。这样在已知 region cut 上可直接二分到该 key 的本区间，并把 tagged-fused 执行限制到物理 region。它不是全局有序 posting，也不重复宣称跨 cut 的 O(T) 界是新发现。

## 表示不变量与取任务

在语料物理坐标上设严格递增的边界 `0=cut[0]<...<cut[T]=N`，令 `region(p)` 为起点 p 所在的半开区间。每个 key 的唯一 `SmallPosting` 满足：属于 region 0 的位置都排在 region 1 前，以此类推；同一 region 内可以是倒序、链表逆序或任意 producer 顺序。出生后的旧 key posting 永不追加，后续只失效或整条退休，因此 stale 位置也保持原 region 编号。

对一个**已知 region 边界** c，谓词 `pos<c` 在该 posting 上先全真后全假，即使内部并未按 pos 排序。因此 `posting.partition_point(|pos| pos<c)` 给出边界偏移；对 region r 用 cut[r] 和 cut[r+1] 得到自己的连续切片。不能对 region 内任意物理 cut 使用同样二分，也不能按任务完成时间连接 region 输出。初版每个 region 任务对每个选中 key 做两次二分、直接借用所获 immutable slice；不建持久 per-key×T 目录，也不必先复制位置到 region 任务。任务的身份由**逻辑 region 编号**决定，可由 Rayon 任意 worker 领取，不能用物理线程 ID 代替。一次 region 任务处理该区的全部选中规则；T 取 W 或固定倍数 W 是待测选择，不把细粒度 region 视为免费动态偷取。

初始化按物理 region 扫描。每个 region 输出按 key-owner 路由物理起点和加权计数；owner 汇总全部 region 权重后才执行阈值过滤，并按 region 编号依次把合格 key 的位置追加到唯一 posting。region 内顺序不重要，跨 region 的连接顺序必须固定。保留原始 `validate_prepared` 完整计时契约，不能因为 region 化偷偷省掉验证。

## 一批非 AA tagged-fused 的执行与修复

协调端仍取精确 greedy 连续前缀，按原次序分配 fresh ID，并把选中 posting 移出 owner。每个 region 任务二分所选 posting 的本区切片，用已有 `inspect_fused` 在共享原子语料中恢复批前邻居，计算旧负 delta 与最终新边，然后写自己的不相交匹配端点。端点跨度可越过 cut；region 只决定**匹配起点的工作归属**，不是语义边界。tagged-fused 的 HEAD/裸尾解码、Release/Acquire 清零重读证明和 token-disjoint 证书仍必须完整保留。任务结果放在以 region 编号索引的固定槽；owner 只在所有 region 任务 join 后开始最终归约和出生填充。下一轮选择还要等所有 owner 完成。

右出生 `(Z,R*)` 的 posting 起点就是本匹配 p，必属于当前 region。左出生 `(L,Z)` 的 posting 起点 `before` 可能早于当前 region，因为一个长 L 可以跨多个 cut。若 `L` 未被相邻左侧匹配选中、`left_id!=0` 且 `region(before)!=region(p)`，当前任务只记录例外 `{key=(L,Z), pos=before, weight, target_region}`，**不得**同时在自己的 route 中调用 `route_birth`。旧边 `(L,A)` 的负 delta 可继续留在当前 producer 的 owner 桶，因为它没有位置排序要求。所有 producer join 后，协调者逐例外调用 `route_birth` 注入目标 region 的 route（包含出生权重、物理出现数和链节点），再让 owner 归约。目标 region 即使没有本轮匹配也须有输出槽。必须检查 route_birth 的加权/次数溢出；若此时失败，训练调用终止，不能在已部分改写的语料上继续下一轮。

对固定一条 cut c，批前连续 token 序列里至多有一条旧邻接 `(L,A)` 满足 `start(L)<c<=start(A)`：若 c 落在 token 内，只有该 token 能跨过 c；若 c 恰是 token 起点，只有其紧邻前驱能在 c 左侧。piece sentinel 只会删除可能的边，不会生成第二条。每条例外对应一条这样的**批前**邻接，并至少穿过一条 region cut；将例外映到它穿过的最左 cut 是单射，故每批例外数至多 `T−1`。一个长度很大的 L 可穿过许多 cut，但仍只有一个后继 A 与一条例外；不能误设目标一定是相邻 region。该证明依赖同一批的 token-disjoint 完整选中语义与 `left_selected` 抑制；若改为仅执行某 key 的部分 occurrence，需要重做 birth 归属证明。

owner 先对所有 region 的 delta 完整加总并按全批权重判断新 key 是否合格；随后对每个合格 key，**按 region 编号**遍历各输出的该 key 出生链并追加位置。链内顺序可逆，下一批只用 region 边界二分，非 AA 不再要求全局位置顺序。不能按 owner 消息抵达次序、Rayon 任务完成次序或 HashMap 遍历跨 region 拼接。新 key 含本批 fresh ID，仅在本轮出生；既有 key 只减、不再收到位置，因此 region 投影不变量可归纳到整个训练。

## AA 与空间边界

AA 仍单独一轮，候选频率包括全部重叠边，实际替换由全局从左到右 run parity 决定。对稀疏 AA，可在选中时**各 region 内**排序并过滤有效起点，再按 region 顺序做摘要前缀；跨 region 的空块和长 token 延续按现有 `aa_parity` 处理。对密集 AA，唯一物理 bitmap 已天然按位置枚举，可按 region 边界划分枚举区间；cut 不一定按 64 位 word 对齐，首尾 word 必须掩掉区间外的 bit。AA 路由也必须写入以 region 编号固定的输出槽，左出生同样走跨区例外修复，随后才 owner commit。不能把 AA 的 `selected_key` 成员判断当作某个重叠 occurrence 被选中的证明，也不能在 fused 非 AA 任务仍写语料时启动 AA bitmap 的稳定快照读取。首版可先保留 AA 完整排序路径，并清楚记录它尚未享受 region locality；这时持久 posting 不变量仍须由 AA 出生填充维护。

## 工作、空间与判定失败的条件

设 B 为批宽、H 为所选历史记录总数、T 为 region 数、W 为 key-owner 数。非 AA 规划仍访问 O(H) 个 posting 记录；任务定位另需约 `2BT` 次二分，最坏 `O(BT log Hmax)`，每批重复支付。跨区修复最多 O(T) 条出生，查目标 region 可二分边界 O(log T)；本任务自己的下界可先做常数比较。每个 region 一套按 owner 路由桶，头部和 HashMap 实例数为 `O(TW)`，不是 O(W²) 除非 T=W；payload、HashMap capacity、owner 表/heap、被移出的 selected posting 和异常记录可能同时存活。异常记录 O(T)，不是 O(N) 或每 key 一张 T 位目录。唯一持久 posting 的 `SmallPosting` capacity、HashMap bucket、AA 排序/bitmap、原子 corpus 与训练调用 HWM 仍要分别报告。

若某 region 含大部分热匹配，静态 region 任务会限制并行；增大 T 又放大 `BT log H`、路由头和 owner 填充中的空 region 扫描。长 token 和大量 stale posting 仍可能造成跨 region/NUMA 读取；cut 不提供预分词，也不承诺缓存行隔离。比较时固定 exact certificate、hash、heap、fixture、CPU 配额和 W1/W4，量出每 region 有效/stale 访问、最大/中位任务耗时、跨区例外、二分时间、route capacity、owner fill 和完整调用/HWM。若假设的 locality 收益小于二分与碎片化成本，应保留原动态 chunk 路线，不把较少远端读当作已测事实。

原定最小实验现已完成：独立克隆 tagged-fused crate，改初始化和出生填充以建立 region 投影，给每个 region 固定 output 槽和跨区左出生修复，再让非 AA 规划按二分的切片执行。完整 `(pair,frequency,fresh ID)` 轨迹与 final tokens 对照通过，定向用例覆盖长 token 跨多 cut、空 region、多 piece、weighted AA 和域回退。实现完成不代表已经提速，首轮数据见开头链接。
