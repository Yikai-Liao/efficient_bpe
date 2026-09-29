# AA dense bitmap 独立审查

结论：在现有 `owned_aa_bitmap` 的前提下，未找到 AA 奇偶选择、相邻 birth
抑制或阶段可见性的正确性反例。此审查是源码证明检查，未运行 Cargo 或性能程序；
完整规则/频率/最终 token oracle 仍是原型放行条件。

## 位的来源与奇偶边界

bitmap 不是按物理语料扫描 ID 得到的。[AA 主路径](experiments/radical/owned_aa_bitmap/src/lib.rs)
只遍历当前选中 key 的历史 posting，每个候选先在稳定语料调用 `inspect`，
通过后才对其**起点**置位。长 token 的尾端也保存 ID，若直接扫 ID 可能将
尾端误认为起点；当前来源排除了这类误判。[Bitmap 实现](experiments/radical/owned_aa_bitmap/src/aa_bitmap.rs)
在 scatter join 后按物理顺序遍历 set bits，并用 popcount 总数与有效 posting
访问数比较；重复有效位置会报错，不能被位去重后悄悄丢失。

对同一个旧 token A，令其长度为 L。一个 AA run 的有效 pair 起点必依次相差
L，左到右非重叠选择第 0、2、4… 个。已按[共享 parity 摘要](experiments/aa_parity.rs)
选中的起点 p 若 `bit[p-L]` 为真，p-L 正是该 run 中被跳过的前一候选，
再前一候选 p-2L 必存在且被选中；应抑制 p 的左边重复扣减和出生。
反之若没有 p-L 候选，p 与任何上一选中 match 不相邻。若
`bit[p+2L]` 为真，则 p+L 必也是有效候选（两端的 A token 连续），
p+2L 是本 run 下一选中 match；p 的右 birth 应使用新 ID 作右端点。
这里的等价依赖**完整有效出现集合**和先完成全局 parity；对任意候选位或
未验证的 stale posting 不成立。piece sentinel、长度大于 255、空 chunk
和跨 chunk run 由同一物理差 L 的 summary/incoming 规则处理。

## 阶段与内存序

bitmap scatter 完成后才计算 chunk summary 和 incoming parity；所有 route
任务在稳定 corpus 上重新 `inspect` 并生成加权 delta/birth，全部 join 后才开始
第二遍选中位遍历和端点写入。apply 任务只读不可变 bitmap/parity 元数据，
选中的 AA span 两两不交叠。apply 全部 join 后才做 owner commit，下一 epoch
的非 AA 读取因此能看到写入。bitmap 使用 Relaxed 原子位操作足够依赖这些
阶段 join；它没有在 route 和 apply 并发时承担跨格发布协议。

## 密度成本与 guard 限制

设 N 为物理 corpus 位置数，H 为本次选中 AA key 的历史 posting 长度。
触发条件 `H >= ceil(N/16)` 给出 bitmap 的 `ceil(N/64)` 个 word 为 O(H)。
scatter 扫 H 条历史记录；初始化及 summary、route、apply 三次位遍历各为
`O(N/64+V)`，V 为有效位数且 V≤H，所以该 AA key 的附加工作 O(H)。
同一旧 pair key 选中后退休，不能在以后 epoch 再次出生；这个界可随选中
posting 历史量摊销。它不保证更快：若 H 大多 stale，仍会触发整张位图扫描；
长 run 的 atomic OR 也可能争用同一 word。

代码先检查逻辑 bitmap 字节数不超过选中 posting 的 heap capacity 字节数，
成功分配后又以实际 `Vec` capacity 复查。因此被采用的 bitmap 常驻载荷
不大于该 posting 的已分配槽位载荷，且只有一份，不按 worker 复制。
但实际 capacity 是**分配之后**才知道：若分配器超配而复查失败，短暂尝试
期间仍可能出现超过该预算的峰值，然后才 drop/fallback。该 guard 是采用
条件，不是严格的瞬时 RSS 上限。它也未覆盖 Vec header、owner route 输出、
allocator overhead/碎片；`aa_bitmap_peak_bytes` 报 bitmap 的 actual capacity，
`peak_aa_plan_capacity_bytes` 只报 sort path 的 Plan Vec capacity，二者都
不能替代训练时 VmHWM。

## 与 tagged-fused 的最小组合契约

当前实验是原始 u32 端点表示。若将来与 tagged-fused 组合，AA bitmap 的
scatter/route `inspect`、owner birth 校验及最终 token 解码须走同一个
`const TAGGED` masked stable reader；AA apply 须调用 tagged 的
head→右起点→尾端 Release writer，不能保留直接写裸 `new_id` 的分支。
AA 自己仍保持 scatter→route→apply 两阶段隔离；非 AA fused 与 AA 间必须
逐 epoch 全部 join，融合专用旧邻居 decoder 不应被 AA bitmap 调用。
ID 高位域的整调用回退也必须由共享入口决定。bitmap 的 Relaxed 位操作
不需要因为 endpoint tag 而升级，只要上述阶段屏障仍在。
