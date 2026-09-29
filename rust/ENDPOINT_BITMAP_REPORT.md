# 融合端点读取与密集 AA 位图

两条独立 Rust 路线都已通过完整 greedy 轨迹验证。融合端点版去掉非 AA 的
逐匹配起点数组和 plan→apply 屏障；密集 AA 版用一份共享位图替代排序、有效
起点数组和逐匹配 Plan。它们削减了确定的临时状态，但首轮单次小测波动明显，
不能据此宣称达到四核 3× 或切换通用默认。

## 两项改动分别解决什么

[owned_fused_endpoint](experiments/radical/owned_fused_endpoint/DESIGN.md)
让新 token ID 同时指出它合并前的两个 constituent，用一个 head 标记区分新
首端点与末端点。线程即使看到邻居只写到一半，也能恢复此次批次之前的邻居。
右起点被清零时要 Acquire 重读前一 head，与写者的 Release 顺序配合；
所有自己的路由记录生成完后才发布自己的端点。它仍只执行已有证书允许的
连续精确批次，AA 保留原两阶段协议，不需要给连续语料找空格切点。

每个非 AA 历史位置仍只作常数次验证和邻居读取，没有整份 corpus 快照。
低 31 位保存 ID 的前提在入口检查，超域整次调用回原 u32 两阶段；长度仍是
u32，未重新引入 255 限制。同 binary 的 `two-pass`、`tagged-two-pass`、
`tagged-fused` 分别控制原路径、标记成本和真正融合，std/aHash 均保留。
证明及适用边界见[端点审查](FUSED_ENDPOINT_SNAPSHOT_REVIEW.md)。

[owned_aa_bitmap](experiments/radical/owned_aa_bitmap/DESIGN.md)只在选中 AA
的历史位置数 H 至少为物理语料 N 的约 1/16，且位图载荷不超过 selected
posting 堆载荷时启用。起点只能由历史 posting 验证后置位；直接扫重复 ID
标签可能把长 token 尾部误认成起点。各线程按 4096 个物理位置分块，摘要传递
整条 AA run 的奇偶，随后流式生成路由，再流式重放写入，不存 Plan Vec。

密集路径每次只做 O(H) 总扫描，额外位图约 N/8 字节且只有一份；摘要前缀
是 O(N/4096) 的串行工作。原子 word 争用和负载不均仍会限制并行度。稀疏或
超预算时回退原排序，所以完整混合算法仍可能有 O(H log H) 排序项。
容量守卫并非严格瞬时 RSS 上限，详见[位图审查](AA_DENSE_PARITY_REVIEW.md)。

## 验证与首轮小测

融合版通过 14 项 Rust 测试、strict Clippy、240 次标准完整 oracle，以及
18 次 chunk=3/7 的 W4 并发完整轨迹；位图版通过 13 项 Rust 测试、strict
Clippy、160 次标准 oracle 和 6 次额外对照。后者包含加权 AA 跨 4096 位置
边界、位图和回退同时触发、token 长度超过 255；另有 4 MiB unary 正确性
检查，它不是 4 MiB 性能数据。局部交错模型只验证顺序一致调度，另有独立
Release/Acquire 论证，未冒充弱内存硬件测试。

[30 次小测](batch_results/radical-fused-bitmap-quick-v1/README.md)每格只有
一次，全部完整轨迹与 fingerprint 匹配。所有版本使用 aHash、固定 W1/W4
进程亲和；HWM 在训练后、trace 构造前采样。

| 256 KiB / 512 规则 | 原 two-pass W1 / W4 | tagged-two-pass W1 / W4 | tagged-fused W1 / W4 |
|---|---:|---:|---:|
| EN | 0.05926 / 0.04589 s | 0.05528 / 0.05378 s | 0.05285 / 0.04332 s |
| ZH | 0.03272 / 0.02124 s | 0.02974 / 0.02757 s | 0.02439 / 0.01704 s |

英文 68/69 批、中文 76/87 批实际融合。非 AA 临时起点逻辑峰值从
59,796/8,932 字节变为零；AA 仍可能保留计划。四核 HWM 原路径→融合为
英文 9.12→8.96 MiB、中文 8.42→7.97 MiB。该小测的自身 W1→W4 仅约
1.22×/1.43×。同窗公平直接串行 CF32 aHash checked 为英文 0.03778 秒、
中文 0.01911 秒，融合 W4 对其英文更慢、中文更快，不能省略这个对照。

| 65,536 字符 / W4 / n=1 | sort → bitmap 调用 | sort → bitmap HWM | sort → 混合算法 Plan 容量峰值 |
|---|---:|---:|---:|
| unary A | 0.01086 → 0.00814 s | 4.71 → 3.56 MiB | 1,048,576 → 65,536 B |
| 交替 AB | 0.00810 → 0.01085 s | 4.01 → 3.88 MiB | 524,288 → 65,536 B |

两例位图峰值均 8,200 字节，分别在 4/3 个早期 AA 批次启用，之后各有
12 批回退，所以混合算法的 Plan 峰值不为零。位图阶段自己没有逐匹配 Plan。
自然英文 **零批启用位图**、一批回退，但 sort/adaptive W4 却测出
0.05525/0.03725 秒；这个负控制清楚说明单次短调用有较大变异，不能把
差值归给一个根本没运行的位图算法。

## 有限较长输入复核

只为核实融合的小测趋势，另做了[20 次有限复核](batch_results/radical-fused-endpoint-long-v1/README.md)：
4 MiB、3000 规则、每格 n=2，同 binary 原 two-pass/tagged-fused，加公平
直接串行 CF32 aHash checked。20/20 完整轨迹匹配，没有扩大为旧算法全矩阵。

| 输入 | two-pass W1 / W4 | fused W1 / W4 | fused 自身 1→4 | 直接串行 / fused W4 |
|---|---:|---:|---:|---:|
| EN | 1.282 / 0.492 s | 1.245 / 0.549 s | 2.27× | 1.066 / 0.549 s（1.94×） |
| ZH | 0.559 / 0.328 s | 0.553 / 0.311 s | 1.78× | 0.648 / 0.311 s（2.08×） |

表中为两个观测的中位数，不是稳定排名。英文融合 W4 两次为 0.648/0.451 秒，
范围与 two-pass 的 0.519/0.465 秒交叠；中文两组范围也交叠。英文融合两次
初始化也从 0.119 变为 0.057 秒，波动并不只出现在被修改的规划阶段，不能
把所有差值都归因于融合。按中位数，融合 W1 仅略快，W4 英文更慢、中文略快，
没有形成通用速度收益或四核 3× 的证据。

非 AA 临时起点载荷从英文 1,243,208、中文 113,260 字节变为零，实际融合
240/249 批、2,889,948/638,508 次匹配。W4 训练 HWM two-pass→fused 为
英文 94.75→92.66 MiB、中文 99.85→97.28 MiB；英文 W1 却为
78.92→80.03 MiB，不能把去掉一个数组等同于进程峰值必降。
融合中的 `apply_seconds` 接近零只表示写入移进了 plan；英文 W4 的
plan 中位数 0.234 秒，高于原 0.201 秒，写与读的额外成本仍须承担。

## 当前整合决定

保留独立实现及同 binary 控制，尚不合并为默认。

后续 [word-cache 原型](experiments/radical/owned_aa_bitmap_cache/DESIGN.md)
让每个 scatter 任务暂存一个 bitmap word，遇到另一 word 或任务结束才发出
一次原子 OR；不要求 posting 排序，只增加每个活动任务的常数状态。
18 项 Rust 测试、strict Clippy、162 次完整 oracle 及
[8 次轻量调用](batch_results/radical-aa-bitmap-cache-quick-v1/README.md)全部通过。
unary/AB 四核的原子 OR 次数分别从 122,876/57,341 降到 4,121/3,081，
约少 29.8/18.6 倍；四个对照的 scatter 子阶段均下降，但完整调用方向不一致。
AB 四核从 0.00780 变为 0.01112 秒，不能用 unary 的 0.01586→0.00823 秒
宣布普遍提速。这是工作量减少的证据，不是稳定端到端排名。

[复用 producer 表做归约](OWNER_ACCUMULATOR_NEXT.md)已完成 12 次小测，
插入入口减少但完整调用无一致收益，暂不组合。

融合端点加 word-cache 位图的[组合原型](experiments/radical/owned_endpoint_bitmap_combo/DESIGN.md)
已通过 23 项 Rust 测试、strict Clippy 和 485 次完整 oracle，包含同次训练
非 AA 融合、dense AA、稀疏回退和长度 >255 的切换。
[16 次小测](batch_results/radical-endpoint-bitmap-combo-quick-v1/README.md)
证明两类临时数组的节省可以共存。AB 64 KiB 的 W4 原/组合调用为
0.00862/0.00634 秒，非 AA 起点载荷代理 131,072→0 字节，AA Plan
容量峰值 524,288→65,536 字节，额外 bitmap 峰值 8,200 字节；剩余
Plan 来自 12 批稀疏回退。组合自身 W1/W4 为 0.00863/0.00634 秒，
仍只有约 1.36×，不能拿同核模式切换当多核扩展。EN/ZH 均未启用 dense AA，
其 bitmap 开关差值不构成算法收益。这是保留组合选项的机制证据，n=1
不支持将其推为通用速度默认。

[region 投影有序 posting](REGION_ORDERED_FUSION_NEXT.md)已完成独立验证：
16 项 Rust 测试、strict Clippy、168 次完整 oracle、8 次轻量调用均通过。
它使相邻物理位置归同一任务，不复制每 key 的索引，也不把切点当成 token
边界。但首轮自然语料没有净速度收益，暂不与 bitmap 组合。

| 256 KiB / 512 规则 / n=1 | dynamic W1 / W4 | region W1 / W4 | region 自身 1→4 |
|---|---:|---:|---:|
| EN | .06327 / .03824 s | .06336 / .04004 s | 1.58× |
| ZH | .02910 / .02187 s | .03024 / .02540 s | 1.19× |

两模式均是同 binary tagged-fused，只有任务归属、posting 投影与 AA 重分组
等 region 结构不同。W4 region 的进程 CPU 时间亦较高，英文 .12972 对
dynamic .10314 秒、中文 .07634 对 .07176 秒；HWM 基本相近。
按 `total_visits / sum_each_batch_max_region_visits` 计算，英文 3.69、
中文 2.59；这是每条记录等成本时的静态负载上界，绝不是实际加速比。
英文较均衡仍未变快，中文则另有倾斜，不能只用「减少远端访问」解释结果。
二分 worker 时间之和 .00116/.00058 秒不是 wall time。详情见
[region 小测](batch_results/radical-region-fused-quick-v1/README.md)。

下一步保留[区域边界快照](REGION_BOUNDARY_SNAPSHOT_REVIEW.md)这一更明确
的所有权方向：跨区读使用批前数值快照、跨区写 join 后回放，而不只是更换
任务排序。目前只有证明与[实现计划](REGION_BOUNDARY_SNAPSHOT_NEXT.md)，
尚无 Rust 训练器或计时；所有跨区写完成前建立下一批快照会错误复活旧 pair，
这个反例已经纳入设计边界。
