# 融合端点快照的独立正确性审查

结论：在草案限定的**非 AA、实际匹配互不共享 token**的完整精确批次内，
我没有找到可阻断的顺序一致交错反例。这个结论要求下列读取与写入契约全部落实；
它不是现有 trainer 的正确性证明，也不是弱内存平台的实测结果。

## 最容易漏掉的交错

右侧选中 `(C,D)` 已写 `t=fresh|HEAD`，随后因 `len[D]>1` 写
`u=0`。左侧 occurrence 仍可先读到旧 `C`，再读到 `u=0`。若直接把零
判成 piece 边界，就漏掉 `(fresh_left,fresh_right)` 出生边。因此零分支
必须 **Acquire 读取 u 后再 Acquire 重读 t**；只读一次 t 无法保证正确。
当前草案包含这一步。对于 `len[D]=1`，u 写入裸 fresh ID，直接按规则
右 constituent 解码旧 D，仍可用 selected[(C,D)] 得到最终 fresh ID。

若 `u` 是另一个选中 `(D,E)` 的 fresh HEAD，而 `(C,D)` 未选，
它代表旧 D 的起点；解码得到 D 后，selected[(C,D)] 为假，自己的右邻仍是
旧 C。不能把“在 u 看见任意 fresh”推断为 C 已合并。若 t 本身为 fresh
HEAD，它才代表 `(C,D)` 已启动，旧 C 和最终 fresh ID 均来自该规则。

左侧选中 `(K,L)` 与自己相邻时，`p-1` 是旧 L 的末端：`len[L]=1`
会变为裸 fresh ID，`len[L]>1` 在第三次写前仍是旧 L、写后才是裸 fresh
ID，始终解码为 L。`before-1` 是旧 K 的末端：`len[K]=1` 可变为 fresh
HEAD，解码为 K；更长 K 的末端保持旧 K。于是 selected[(K,L)] 始终可
抑制右侧 occurrence 的旧左边扣减与新左边出生。左侧 occurrence 负责旧
`(L,A)` 的一次扣减和 `(fresh_left,fresh_self)` 的一次出生。同规则 ABAB
也是这两种相邻情况；key 相同不改变位置归属。

## stale posting 与空间前提

草案自己验证时必须比较 **去 HEAD 位后的完整 ID**：p 仍是旧 a，
q=p+len[a] 仍是旧 b。批内写入到任何位置的值只可能是零或本批 fresh ID，
去标记后均不等于批前旧 a/b。因此若两次读仍分别为 a/b，两个格在批前
也必为 a/b；并发写不能使 stale posting 复生。还要用稳定语料的归纳：
这里的已选 key 必须全部在批前确定，fresh ID 不能在同一批再成为已选 key。
一个曾有效的旧 pair 只有其一个 constituent 被合并才会失效，此时其
起点 p 或右起点 q 会被改成零/新 ID；后来没有操作能写回旧 ID。
一个仍为有效 a 起点的 p，其固定长度的下一物理位置 q 正是下一活 token
的起点。前批任务全部 join 并对本批读建立可见性，是这个归纳的必要前提。

一旦自己通过验证，其 a/b token 区间不能被本批其他合法匹配写入，
所以在自己发布前不会再失效。此处需要的证书是**实际匹配不共享 token**，
不要求所选 pair 类型完全不同；空间探测放宽的类型冲突也可适用。
但若 AA、部分选中同 key 出现、或仅有候选 key 而未证明所有实际匹配
不交叠，上述“selected key ⇒ 该邻接 occurrence 已执行”不成立，
birth suppression 不能照搬。

## 内存序与实现边界

对右邻清零的证明依赖同一线程 `store(t, fresh|HEAD, Release)`
**先于** `store(u, 0, Release)`。读取 u 的零值若来自后者，Acquire
与其同步；先前 head 写因而 happens-before 随后的 t 重读。t 是批内
唯一写者，原子修改序及 read-write coherence 保证重读见 fresh HEAD。
初始 u 若为 sentinel 零，`(C,D)` 根本不是有效选中匹配，t 不会由该
规则改写；重读旧 C 可确认为边界。模型只枚举顺序一致调度，不能替代
这段 Release/Acquire 论证。对其他单格 fresh 读取，可直接从该值及启动
前发布的不可变规则表解码，不依赖另一格是否已经写完。
此时边界位于 C **之后**，用于出生边的最终右邻仍为 C；只有再往右看的
`next_id` 才是零。

实现时应一次性验证 `initial_lengths.len() + max_merges <= 2^31`（用
checked 算术；最大实际分配 ID 为和减一），超域整轮走原两阶段 u32
表示。所有稳定读者——初始计数、AA 两阶段、权重以外的邻居读取、owner
出生校验、最终解码——都要先去 HEAD 位；当前批的特殊邻居读者再根据
raw tag 与 `fresh_begin` 复原**旧** constituent。sentinel 0 不带标记。
不得将“已知批前 token 末端”的解码器用于任意内部 stale 位置。
AA 保留排序、奇偶选择和两阶段写入；两个模式每批结束后都必须完成 join，
下一批选择/规划才能读取 corpus。若任何任务报错，可丢弃已消费的私有
训练调用；不能把部分改写的 corpus 继续用于下一批。

## 局部交错模型和后续测试

[模型脚本](experiments/radical/fused_endpoint_model/interleavings.py) 对右侧
`HEAD→clear/tail` 和左侧 `HEAD→clear/tail` 分别枚举写顺序与自适应
读顺序的 132 个顺序一致前缀；长度 1、2、257、fresh HEAD/裸尾、sentinel、
同侧外向合并均覆盖。运行结果为 17 组局部情形全部通过。它把读写格抽象
为名称，未模拟整个语料或弱内存；上述 stale 归纳和 Release/Acquire
证明独立于模型。

trainer 落地时还需要完整规则/频率/最终 token oracle：相邻同规则
ABAB、相邻不同规则、隔一个旧 token 的外向合并、空间证书允许的类型
冲突、长于 255 的旧 token、piece 首尾、旧 posting 大量 stale、AA 与
非 AA epoch 交替，以及 ID 域边界和超域整轮回退。对照必须同时有
tagged-two-pass 和 tagged-fused，才能把标记表示成本与融合收益分开。
