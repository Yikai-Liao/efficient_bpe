# 区域快照与直接新键汇总

本轮让两个 Sol 分别实现独立 Rust 原型，由统一执行者负责 Cargo、oracle 与
轻量计时。没有改动另一份 tokenizer benchmark，也没有运行 4 MiB 或历史全矩阵。
结论：两条协议都通过精确性验证，但这轮未显示一致的速度收益，不切换默认。
四核 3× 目标仍未达到，Goal 继续。
两个原型都保留精确 greedy 的规则顺序、加权频率和最终 token；源码、二进制
与 fixture 的哈希以及原始结果保存在各自 gate/quick 目录。

## 区域边界快照

[实现](experiments/radical/owned_region_snapshot/DESIGN.md)在同一 binary 中保留
dynamic、原子 region、snapshot region。后两者使用相同固定切点和按区域投影
有序的位置表。snapshot 将非 AA 的语料访问改成独占切片：本地通过 get_mut
普通读写，远端从批前六 token 的数值窗口恢复邻居；越界端点写在 join 后回放。
每条 cut 至多被一个选中匹配跨度跨越，待写端点最多 2(T−1)，无需复制整份
语料或为每线程复制位置索引。AA 仍走原稳定规划和原子写回，并更新 cut anchor。

切点可以穿过长 token，这个协议不依赖空格或预分词。其限制也很具体：T 仍为
min(W,N)，任务可能不均；每批还要 O(TW) 路由头和约 2BT 次 posting 二分；
31 位 HEAD 域不满足时整调用回退原 full-u32 两阶段路径。去掉本地原子访问
不等于保证更快，新增窗口查询、计数器、边界检查和刷新都在调用计时内。

[正确性归档](batch_results/radical-region-snapshot-gate-v1/differential.json)：
23 项 Rust 测试通过，含 5 项局部模型；160 标准、10 定向和 1 dynamic smoke
完整轨迹通过。定向 `ab×129` 的四区训练确有 13 次远端查询和 1 次延迟端点写；
另覆盖超过 255 的 token、W>N、AA、sentinel 与域回退。Clippy 修正仅涉及
模型循环和冗余 map，没有训练结果失败；未运行 Miri 或 sanitizer。

## 新键直接暂存在 owner Entry

[实现](experiments/radical/owned_pending_entry/DESIGN.md)保留 staged、
fused-direct-combined、fused-direct-pending。两个 fused 路径使用相同的旧键
直接扣减、路由转置和连续提交，只有新键归约位置不同。pending 在 owner 私有
提交阶段借空 inline posting 的一个 u32 槽累计物理次数，Entry 中累计 u64
权重；完整归约后删除不合格键，分配并填满 posting，最后才向候选堆发布。
Entry 和 SmallPosting 不增宽，没有复制永久索引。

它消除临时新键 HashMap，但保留 touched/expected Vec；期望工作仍为
O(D+B+births)，没有渐近时间改善。B 个新键中只有 E 个合格时，B≫E 可能让
永久表短暂扩张并留下分配高水位，所以同 binary 的 combined 才是归因基线。
旧 4 MiB 自然文本 fixture 的 weight=2/minimum=2 等效无权 minimum=1，无法
凭它证明高阈值下也节省内存。本轮 256 KiB fixture 实际 weight=1、minimum=2，
还另加英文 minimum=16 的有限对照；两种 fixture 不应混作仅大小不同的缩放实验。

[正确性归档](batch_results/radical-pending-entry-gate-v1/differential.json)：
14 项 Rust 测试、240 标准和 24 定向完整轨迹通过。高阈值定向输入真实触发
B=256、E=0。首次容量测试错误地假定删除前后公开 capacity 相等；观测为
448→192，已修正测试及报告口径。它不是训练轨迹错误，也不能据此说分配已
收缩。公开 capacity 是不重新分配可容纳的元素数，不能当作物理 bucket 数。

## 28 次小测的结果

仅 EN/ZH 各 256 KiB、512 规则、aHash/lazy、每格 n=1。W1 的整个进程限制在
CPU 5；W4 为 CPU 0/1/2/5，包含协调者。call 包括训练验证、初始化、提交、
最终解码与清理，排除 JSON 解析和轨迹指纹；CPU 计时取同一段。训练 VmHWM
在训练返回后、生成 trace 前采样，仍含进程启动和 JSON 解析的历史峰值。
所有 28 次完整 trace 均匹配；高阈值另有一次串行正确性参考，不纳入计时表。
快照小测内部以 dynamic 完整轨迹为参考，其 12 个指纹另与 pending 小测同输入的
[直接串行参考交叉核对](batch_results/radical-region-snapshot-quick-v1/cross-screen-serial-check.json)一致。
这些单次小调用只能作机制筛选，不能作稳定排名或服务器扩展性结论。

[区域快照小测](batch_results/radical-region-snapshot-quick-v1/README.md)，单位 ms：

| 语料 | 模式 | W1 call | W4 call | 自身 W1/W4 | W4 CPU | W4 HWM MiB |
|---|---|---:|---:|---:|---:|---:|
| EN | dynamic | 56.20 | 45.87 | 1.23× | 120.36 | 8.93 |
| EN | atomic region | 63.20 | 53.62 | 1.18× | 108.74 | 8.78 |
| EN | snapshot region | 60.66 | 61.98 | 0.98× | 157.01 | 9.01 |
| ZH | dynamic | 25.01 | 32.70 | 0.76× | 89.16 | 8.36 |
| ZH | atomic region | 36.45 | 41.05 | 0.89× | 91.61 | 7.94 |
| ZH | snapshot region | 29.14 | 22.16 | 1.32× | 70.06 | 8.30 |

snapshot 的 W4 窗口载荷两例都是 600 字节；英文有 21 次边界查询和 2 次
延迟写，中文为 2 次查询、0 次延迟写。英文普通本地读/写为 942246/326814，
中文为 156147/56898。常数规模的通信确实成立，但整体访问协议仍未获得一致
调用收益；ZH 相对原子 region 的单次大幅改善不能盖过 EN 的退步和小测波动。

[pending 小测](batch_results/radical-pending-entry-quick-v1/README.md)，单位 ms：

| 语料 | 提交方式 | W1 call | W4 call | 自身 W1/W4 | W4 CPU | W4 HWM MiB |
|---|---|---:|---:|---:|---:|---:|
| EN | staged | 65.60 | 44.28 | 1.48× | 132.74 | 8.37 |
| EN | direct + combined | 63.83 | 31.12 | 2.05× | 88.22 | 8.80 |
| EN | direct + pending | 69.19 | 36.17 | 1.91× | 88.80 | 8.80 |
| ZH | staged | 31.73 | 19.98 | 1.59× | 66.43 | 7.82 |
| ZH | direct + combined | 31.17 | 22.35 | 1.39× | 66.49 | 7.99 |
| ZH | direct + pending | 36.86 | 19.75 | 1.87× | 63.74 | 7.79 |

同窗口直接串行 CF32/aHash/checked 为 EN 47.94 ms、ZH 24.03 ms。pending
W4 相对它仅约 1.33×/1.22×；这与 pending 自身 W1/W4 是不同分母。临时
新键汇总载荷代理在 W4 EN 从 43008 降为 32768 字节，ZH 从 26880 降为
16384 字节；它不包含永久表、路由、expected Vec 或分配器开销，不能直接
作为总内存节省。普通 minimum=2 时，EN 的累计 B/E 为 33681/19258，ZH 为
15364/5338，确有大量被拒绝的出生键。多线程 D 随任务归属略有变化，不是固定常量。

额外 EN minimum=16 对照只跑 W4：combined/pending 的 call 为 24.25/40.36 ms，
CPU 为 80.62/82.91 ms，HWM 为 7.070/7.145 MiB。B=33681、E=3306，约九成
新键被拒绝。提交后公开 owner 容量峰值从 combined 的 3261 增至 pending 的
7162，约 2.20×。这是完整训练中的公开容量差异，**还不能证明永久分配真的
扩容**：初始 retain 后可能已有 tombstone，新键插入后在同一分配上重排或
复用槽也能恢复可用容量。各值是跨 owner 求和后再跨批取最大，未必出现在同一批。
首次从空表插入再删除的 unit 反例仍有价值，但要证明自然训练的底层分配污染，
还缺分配器观测。也不能以单次 wall 差距宣称稳定慢了 1.66×。

## 下一轮的选择

[固定微区](REGION_TASK_GRANULARITY_NEXT.md)把 T 与 W 分开，尝试让更多独占
区域任务由 W 个线程动态领取，以改善固定四区的负载；必须同时承担增加的
分区搜索和路由头，不能移动 cuts 后仍假装 posting 对任意坐标有序。

[不可变 pair row](IMMUTABLE_PAIR_ROWS_REVIEW.md)审查发现，小 row 的目录开销
和热门 row 的 owner 倾斜可能抵消哈希节省，暂不直接替换永久表。
[posting 分级存储](IMMUTABLE_POSTING_STORAGE_NEXT.md)则利用每个 pair 只出生
一次、位置表此后只会失效的性质设计磁盘封存；它尚未实现。累计位置载荷的
O(N) 界不自动保证顺序 I/O，也没有解决 AA 排序、构建峰值及频率/heap 常驻。
