# 由唯一出生来源维持全局有序的位置表

已实现于 [owned_ordered_posting](experiments/radical/owned_ordered_posting/DESIGN.md)，
通过 25 项 Rust 库测试、90 次完整轨迹对照和 dynamic/global 拒绝检查；
性能判断见本轮小测归档。这项设计针对 region 执行器，目标是在不比较排序、不增加
N 大小缓冲的条件下，让每个 pair 的历史 posting 始终按物理位置严格升序。
它比现有“按 region 投影有序”更强，可直接取消 AA 的历史位置排序，未来也
允许在任意新物理 cut 上二分。当前实验先固定 cuts，不同时实现动态分区。

## 不变量和归纳

初始扫描在各 region 内按物理位置递增；owner 按 indexed collect 的 region
次序连接同 key 的出现，初始 posting 全局有序。每次合并分配新的 token ID，
所有出生 pair 至少含一个本批 fresh ID，因此旧 pair 不再新增位置，只留下
历史 stale 项。保留或删除这些项均不破坏已有次序，关键只在本批新 pair。

对一个固定出生 key，分三种情况：

- `(L,Zi)`，L 是旧 ID、Zi 是本批新 ID，只能由规则 i 的左 birth 产生。
- `(Zi,R)`，R 是旧 ID，只能由规则 i 的右 birth 产生。
- `(Zi,Zj)`，两者都是本批新 ID，只能由左边规则 i 的右 birth 产生。右边
  规则 j 的左 birth 被既有 left-selected 判定抑制，不能重复生成。同一
  规则的相邻匹配产生 `(Zi,Zi)`，也由左匹配右 birth 唯一产生。

因此同一 key 不会因多个规则按 rank 交错处理而打乱出生顺序。region 任务
扫描规则 i 的 posting 子段时，有效匹配起点 p 是原严格升序列的子序列。
右 birth 在 p；左 birth 在 p 的批前直接前驱 head before。两个有效匹配
p1<p2 若都产生左 birth，其前驱满足 before(p2)≥p1>before(p1)；跨 piece
的 sentinel 使没有真实左邻的 birth 被跳过，不导致倒序。故每个 region 的
同 key 局部出生 append 顺序递增。AA 使用全局左到右 parity 后的 Plan，
按 region 分组仍递增，三种来源和相邻匹配抑制规则相同，故归纳同样成立。

## 跨区左 birth 为什么可以最后注入

设本区匹配 p 的左前驱 L 起点 before 落在更左的目标 region。L 覆盖
`[before,p)`，而 p 已越过目标 region 的上 cut。因此 before 是目标 region
最后一个批前 live head；该 region 不可能还有第二个 head 也作为另一条
跨出上 cut 的活邻边起点。即使 L 跨多条 cut，仍只有这一个 before 和后继 p。
当前协议在 join 后向目标 region 注入这条 birth，正好把它放在其他局部出生
之后，保持同 key 的 append 顺序。多个 piece 不改变唯一性：若中途有
sentinel，L 与 p 就不是合法邻居，不能产生这条 birth。

这依赖目标 region 的输出顺序，不能把例外附加到原 producer 输出，或按
任务完成顺序收集 region。出生位置必须是批前 live head；历史 stale 位置
通过验证前不能被用来论证“最后一个 head”。

## 如何消除链表逆序

grouped birth 使用 head 指向最新节点，沿链读取天然逆序。owner 每次处理
一个 `(region,key)` 时，先记住该 posting 的原长度，按现有链校验与填充方式
追加这一段，再只反转刚追加的 slice。随后处理下一 region，整条 posting
便全局升序。不能在末尾反转整条 posting，否则连 region 顺序也会颠倒。
debug 下检查新 posting 的严格单调，模型还应检查同目标 region 的跨区例外
唯一性。若任一合法 fresh key 来自不同规则的非有序混合，就推翻这里的证明，
不能用隐蔽 sort 修补。

每个出生位置只进入一份 retained posting，初始加所有出生位置数为 O(N)，
所以全部新增反转工作累计 O(N)，单次额外空间 O(1)。原 posting 分配、route
map、出生链仍存在；这不是总内存降到 O(1)。第一次实验采用受检 slice.reverse，
先隔离不变量；直接反向初始化预留空间可以另作有安全证明的常数优化，不混入
当前比较。动态 flat-task 执行器的输出按 worker 而非位置收集，单个 worker
的升序子序列不能直接拼成全局有序，因此该模式不能默认套用此证明。

## AA 与后续用途的边界

有序历史 posting 经有效性过滤后仍有序。AA 保留既有 run/parity 规划和
替换协议，仅跳过最前的 posting sort；历史扫描变为 O(H)，不再依赖排序器
最坏界。这不是承诺 AA 完整训练时间线性：哈希、堆和其他阶段仍需计入。
已有排序器对有序数据也可能很快，新增反转又作用于所有新 posting，故要测
完整调用，而不是仅计“少了一次 sort”。

HEAD 域回退到 dynamic/full-u32 时，若没有维持同一有序发布协议，应整调用
关闭该优化并报告 effective 模式。固定域/长度和所有加权频率语义保持不变。

未来移动 cut 时，全局有序 posting 足以支持新 cut 的二分，却不能自动重建
snapshot anchor；atomic region 可单独探索，snapshot 还需新的 anchor 定位
协议。磁盘封存后按有序 extent 读取也可减少 AA 外部排序需求，但初始/出生
构建峰值、在途读取块、频率/heap 常驻和磁盘 I/O 放大仍未解决。本次实现不
应被称作动态分区或严格预算的外存训练器。
