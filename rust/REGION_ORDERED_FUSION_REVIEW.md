# Region 融合原型的代码审查

对象为独立 `owned_region_fused`，主线只在 tagged-fused 下启用 region；
超出 tagged ID 域时回原 two-pass/dynamic。此处记录静态审查；编译、
完整轨迹和性能以 executor 的版本归档为准。

初始化按固定物理区间产生 W 个输出，Rayon indexed collect 保持 region
编号顺序。owner 按此顺序追加，建立每 key 唯一 posting 的 region 投影。
以后 key 只在出生批次追加，填充仍按 region 输出顺序进行；区内 birth
链逆序无妨。debug 在初始化及每个合格新 key 填充后检查整个投影非降。
`partition_point` 的条件只在这些已知 cut 上成立；两个返回值 first≤end
不能代替该不变量。

W 大于物理位置数时允许重复 cut 和空区间。`RegionCuts::of` 使用最后一条
不大于位置的 cut，因此一个非空位置仍只属于一个区间；空区间不产生扫描
或重复出生。入口先验证 W>0。初始化不扫描首尾哨兵之外的邻接，不引入
全 posting 排序来获得投影。

非 AA 只处理本区的起点片段，继续使用原 tagged decoder 与不相交匹配
证书。左出生若落在本区之前，先进入异常表；所有 region join 后才在目标
区调用一次 route_birth。负 delta 留在原 producer，出生 weight/count
只注入一次。每个真实例外对应批前一条跨 cut 的旧邻接；每条 cut 只容一条
这样的邻接，长 token 跨多个 cut 也不增加同一条边的例外数。实现检查
每批异常数≤W−1。sentinel 删除邻接，重复 cut 只让这个上界更宽松。

AA 先沿原路径全局排序、过滤、传递 run parity；按起点将已选 Plan 移入
固定 region Vec。区内顺序保持，因此前后区的已选边摘要可用于相邻匹配
去重；AA 的左跨区出生也走异常注入。不能省掉这一步而直接把动态 producer
当成 region。代价是额外串行 O(M) Plan 搬移和新 Vec capacity，原型显式
报告两组 Plan capacity 共存的保守上界。AA 尚未组合 bitmap。

永久索引仍唯一，任务头与 route 实例为 O(W²)，非 AA 每批新增 O(BW log H)
边界定位。单区热度偏斜时无动态细块偷取；`total_visits/sum_max_visits`
只表示每条访问等成本时的理想静态分区并行度，merges 的对应比同理，不能
替代实际 1→4 调用加速比。worker 内二分计时之和不是 wall time，也不能
直接从 call 中扣除。没有测得 cache miss 或 NUMA 数据前，不声称已证实
false sharing 瓶颈或物理线程亲和。

本次审查未发现阻塞正确性的变化。局部性是否抵得过二分、静态倾斜及 AA
重分组，只以同 binary dynamic/region 控制判断；region 不预先成为默认。
