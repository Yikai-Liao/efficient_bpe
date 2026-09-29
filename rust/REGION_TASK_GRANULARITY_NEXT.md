# 将物理区域数与线程数分开

已实现于 [owned_region_tasks](experiments/radical/owned_region_tasks/DESIGN.md)：
24 项 Rust 库测试、92 次独立 oracle 完整轨迹及 8 组 k1 新旧二进制对照通过。
[正确性归档](batch_results/radical-region-tasks-gate-v1/README.md)不代表速度收益。
以下保留设计动机；实现前的 region/snapshot 把 T 取为 min(W,N)，每批非 AA 只产生 T 个区域
任务。它建立了明确的独占写入协议，但固定四个等长物理区间可能分配不均。
[已测 region 小样本](batch_results/radical-region-fused-quick-v1/README.md)中，
英文总历史访问量除以逐批最重区域访问量之和为 3.69，中文只有 2.59。旧指标
包含 AA 位置投影，而 AA 扫描/排序并不按同一 region 调度，不能将其直接当作
实际阶段的扩展性上界。新原型另记非 AA 的总访问和逐批
`max(ceil(total/W), max_task)` 之和，只对非 AA 等成本访问模型给出任务调度下界；
它也不是完整训练的实测加速比。
增加 worker 唤醒策略无法修正这些固定任务自身的粗粒度。

实现固定 T=min(kW,N)，小测仅用 k=1、4，保留 W 个 pair owner。每个区域
仍独占切片；Rayon 将 T 个独立任务调度到 W 个线程。初始扫描和出生 posting
始终按这 T 个固定区域排列输出，owner 仍只持有一份 posting。快照数量、越界
写上界从 W 换为 T，分别为 O(T) 与 2(T−1)。当前代码已经区分 region 与 owner
数量，但这不意味着增加 T 的所有路径已验证，特别要检查 AA 重新分组、跨区
左 birth 的目标、输出收集顺序以及 W>N 的回退。

代价不能省略：每批约 2BT 次位置二分、T 个工作结果，以及 O(TW) 空路由头。
同 key 跨更多区域后，局部聚合条目数也可能增加。若每个任务太短，调度和快照
构建可能比负载改善更贵。因此先记录总访问量、逐任务访问量、实际 wall/CPU、
路由容量和窗口容量，不用任务数或理论比值宣布收益。全调用 W1→W4 仍需比较
相同 k，并保留最快直接串行与 k=1 的绝对耗时，不能靠把 W1 切得更碎制造比例。

这版不动态移动 cuts。其控制 posting 只保证按已知区域投影有序，区域内部可以
任意排列。任意新 cut 上的 `partition_point` 不成立；每批按工作量重新切区
会要求重新排列索引或更细的永久区段目录，是另一项算法及空间代价。固定更多
小区是先隔离负载分配的控制，不声称已经实现自适应分区或 NUMA 局部性。
另一独立原型现已[维持全局有序 posting](ORDERED_POSTING_REVIEW.md)，
为未来任意 cut 二分提供前提；本微区实验未引入该改动。

验证先用同一份连续输入的所有规则/频率/最终 token，覆盖长 token 跨多区、
sentinel、AA→非 AA、stale posting 与延迟写条数界。轻量性能筛选只比较 k=1/4，
确有足够任务和可测负载变化才补中间粒度；没有理由预先扫描一大串参数。
