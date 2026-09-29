# 在写入中恢复批前邻居：待证实的融合规划方案

状态：已在独立 [owned_fused_endpoint](experiments/radical/owned_fused_endpoint/DESIGN.md)
实现，并通过 14 项 Rust 测试、strict Clippy、240 次标准完整 oracle 和 18 次小粒度
并发完整 oracle；见[门控记录](batch_results/radical-fused-endpoint-gate-v1/differential.json)。
局部交错模型与独立内存序论证见[审查](FUSED_ENDPOINT_SNAPSHOT_REVIEW.md)。
[首轮小测及有限 n=2 复核](ENDPOINT_BITMAP_REPORT.md)未确认通用速度收益，保留独立原型。
原设计目标是在精确批次内把每个 occurrence 的
规划与端点写入放到同一个任务，省掉全批 plan→apply 屏障、有效起点临时数组和第二遍
apply 遍历。它不同于已有的 owner commit/apply 重叠实验。本轮只考虑非 AA 批次；AA
仍用现有排序、全局 run parity 和两阶段路径。

## 批次和表示的前提

批次仍是已有证明允许的连续 greedy 前缀：任意两个实际匹配不能共享 token。
每个新 token ID 对应的旧 `(a,b)`、两者长度在启动任务前登记，批内不会变化。
计数、posting 出生和 owner 提交在所有任务结束后进行；只提前执行语料端点写入。

用 u32 高位标记 head，低 31 位存 token ID。所有稳定 reader 都取低 31 位；新
token 的首端点保存 `id|HEAD`，末端点保存裸 id。初始单 token 可标 HEAD（首尾
重合），合并 token 的物理长度至少为 2，首尾不会重合。无需每批去标记，也没有
语料大小的附加数组。整个调用必须先检查 token ID 域小于 2^31，否则回退到原有
完整 u32 两阶段路径。长度仍为 u32，不能把这称为解决任意 u32 ID 的无代价编码。

对匹配 `(a,b)` 起点 p，q=p+len[a]，t=q+len[b]，依次写：

1. `corpus[p] = fresh_id|HEAD`；
2. 若 len[b]=1，`corpus[q]=fresh_id`；否则 `corpus[q]=0`；
3. 若 len[b]>1，`corpus[t-1]=fresh_id`。

写用 Release，需要重建的邻居读用 Acquire。尤其读到步骤 2 的清零后，重读 p
必须看见步骤 1。仅凭 x86 的表现不足以证明跨架构正确性。

## 已知批前 token 末端位置的解码

记 fresh_begin 为本批最小新 ID。在一个**已知批前 token 的末端** e 读取 raw：

- ID < fresh_begin：就是批前旧 ID；
- 新 ID 且有 HEAD：返回该新规则左侧 a；这只可能是旧 a 长度为 1；
- 新 ID 且无 HEAD：返回右侧 b。

批前 token 的末端不会在此次写入中被清零。因此这个函数不需要重试，也不需要
知道任意 stale 物理位置的意义。不能把它扩展为任意位置 reader 而不补证明。

## 每个有效匹配需要的邻居

自己 posting 的 p、q 仍用低位 ID 检查确为旧 `(a,b)`。别的合法匹配不共享其
token；新写入 ID 也不可能等于旧 a/b。是否足以排除 stale interior 误命中需要
沿用并复核现有 posting 不复生不变量，不得仅以“原来就是这样”带过。

左邻：解码 `p-1` 得旧 L，计算其起点 `before=p-len[L]`。如需判断左边匹配是否
也被选择，再解码 `before-1` 得旧 K，用 selected[(K,L)] 判定；边界 0 单独处理。

右邻：t 是自己匹配后的批前 token 起点。因为匹配之间不重叠，它不可能是另一
被选匹配的右 constituent，故 t 不会被清零，也不会变成新 token 的裸尾 ID。

- t 读到新 HEAD：右边匹配 `(C,D)` 已开始，旧 C 和其最终新 ID 都由规则表取得；
- t 读到旧 C：若 C=0 则边界；否则看 u=t+len[C] 的旧 D，以判断 selected[(C,D)]。
- u 若是非零旧 ID，直接用；若是新 HEAD，用规则左 constituent；若是新裸尾，
  用规则右 constituent（右 constituent 原长度为 1）。
- u 若是 0，可能是原始边界，也可能是 `(C,D)` 把长度>1 的 D 起点清掉了。
  此时 Acquire 重读 t：若是新 HEAD，从该规则取得旧 D；若仍是旧 C，则是原始
  边界。读到清零的 Acquire 与其 Release 配合，禁止遗漏此前的新 HEAD 写入。

这个特殊 reader 利用已知 C 起点来区分 0；不引入通用任意位置恢复算法。整个
occurrence 所需的旧邻居和最终 birth key 都计算完成后，才发布自身端点写入。

## 工作流、控制与风险

任务扫描 posting → 验证 → 重建邻居 → 发出与原 plan 相同的 delta/birth 记录 →
立即写入自身三个端点。所有任务 join 后再执行原 owner commit/fill。必须证明
路由去重、相邻新 token 的 birth 归属与 suppression 在任意写入顺序下保持原义。

第一版需要同 binary 的 tagged-two-pass 与 tagged-fused 两个控制，外加冻结
owner aHash 的绝对耗时参考。tagged-two-pass 分离表示与屏障变化成本。预期省掉
4 字节/有效匹配的临时起点载荷、第二遍遍历和一次全局阶段边界；代价是更多按位
掩码、fresh 分支、Acquire/Release 及偶尔的重读。是否净收益须测量。

先独立枚举小型 interleaving 模型，再接完整 trainer。反例集应覆盖：ABAB 相邻
同规则、不同相邻规则、左右长度 1/2/>255、head/clear/tail 的各种部分完成、最左
最右 sentinel、stale posting、空间证书允许的类型冲突、ID 超 31 位回退。并发
模型需要明确区分顺序一致交错验证与 Release/Acquire 的独立内存序证明。

## 已有两阶段实现的计时边界

在 [aHash 4 MiB、3000 规则、每格两次的原始记录](batch_results/radical-local-hash-v1/integer-long.jsonl) 中，分别对完整调用、`plan_seconds` 和 `apply_seconds` 取中位数：

| 输入 | workers | 完整调用 | plan | apply | 最大单批有效起点数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN | 1 | 1.310 s | 0.717 s（54.7%） | 0.0554 s（4.2%） | 310802 |
| EN | 4 | 0.533 s | 0.238 s（44.6%） | 0.0251 s（4.7%） | 310802 |
| ZH | 1 | 0.585 s | 0.163 s（27.8%） | 0.0108 s（1.8%） | 28315 |
| ZH | 4 | 0.303 s | 0.0746 s（24.6%） | 0.00955 s（3.2%） | 28315 |

这两个计时区间顺序执行，互不嵌套。非 AA 路径的 `plan_seconds` 计
`prepare_batch`，但此前的 `FlatTask` 构建不在其中；`apply_seconds` 只计
`apply_batch` 的端点写入，后续 `commit_routes` 不在其中。AA 路径的 plan
还含规划和路由。累计 `flat_tasks` 为 EN 3614、ZH 3017；最大单批任务数
分别为 88、50。`peak_task_starts` 与最大单批有效起点数相同。

融合没有可保证的正收益，实际也可能更慢。即使让整个 apply 区间免费，完整调用的
算术节省上限也只有 1.8%–4.7%；即使让 plan 与 apply 两个区间都免费，
EN 的上限为 49.3%–58.9%，ZH 为 27.8%–29.6%。实际规划、邻居检查、
路由和写入都仍需执行，因此这些上限不能当作预期收益；两次测量也不足以
建立稳定的性能排序。
