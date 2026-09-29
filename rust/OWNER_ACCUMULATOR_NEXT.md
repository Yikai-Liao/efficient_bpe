# 复用最大 producer 表，消掉一次汇总插入

已在独立 [owned_reuse_accumulator](experiments/radical/owned_reuse_accumulator/DESIGN.md)
实现，同 binary 保留 staged、fused-fresh、fused-reuse。12 项 Rust 测试、
strict Clippy 和 240 次完整轨迹 oracle 已通过；[12 次轻量性能筛选](batch_results/radical-reuse-accumulator-quick-v1/README.md)
也已完成，完整轨迹均匹配，但没有一致速度收益，暂不组合到默认路径。
它针对 owner 频率归约，而非端点读写。旧 aHash 4 MiB W4 小样本里该阶段
约占英文 13%、中文 23% 完整调用，不能由阶段占比推断实际加速幅度。

现在每个 producer 已经按 owner 存有 `HashMap<key,Delta>`。owner 又建立空的
combined 表，把所有 producer 的 key/weight/count 再插一次，使用它决定频率和
新 posting 分配，之后还要遍历原 producer 表填入 birth chain。能否选最大
producer 表直接作为 combined，省去复制该表及新建整张汇总表？

## 所有权与链头是核心问题

把本批 route 按 owner 转移所有权，只移动表头和 born Vec 头，不复制 payload。
每个 owner 独占 W 个输入 route；挑 `delta.len()` 最大者，把其 map 移出为
累加表，原 born Vec 留在本 owner。逐项读其他 map，把 weight/count 累到这个表。

不能照搬原 `Delta`：其 head 是 producer 私有 born Vec 的下标。对原最大表已有
key，保留原 head；对只在其他表出现的 key，新插入项 head 必须为 MAX。这样
累加表的 weight/count 是**全 owner 总数**，但 head 仍只表示最大 producer 的
链。其他表及其链头在填充结束前原样保留。切勿将一个外来 head 当成最大表的
born 下标，也不能用已累加的 occurrences 校验该单条局部链长。

频率归约后，每个 eligible 新 key 按总 occurrences 分配一次 posting。先填最大
producer 的链，再填其他 producer 的链；所有链合计长度必须与总 occurrences
相等。局部链的索引、终止、总遍历量仍须单独检查，不能因少了一份局部 count
就取消循环界和初始化完整性检查。旧 key 不应有 born head，新 key 整批结束
之前不向外发布。若任一检查失败，丢弃此次私有训练状态，不能继续下一批。

独立审查补充：现有 born 节点是头插，非终止 `next` 必须小于当前 cursor，可
用这个严格下降条件及 born Vec 长度界排除循环。每条链必须始终在它所属 producer
的 Vec 中走；总填充数与 accumulator 的总 occurrences 核对。若要额外验证原
最大 producer 的每 key 计数，需要在覆盖前保存它，不能把已有总数误当局部值。
最大表也不能再次作为 foreign 表重复参与归约。

## 可预期的节省与控制

若一个 owner 的输入表 key 数分别为 k_i，则原 combined 至少做 sum(k_i) 次
累加入口，新版只需 sum(k_i)-max(k_i) 次；仍有所有 union key 的频率处理和
必要 birth 填充。它不保证完整调用得到同比节省：复用表可能扩容，随机种子会
改变布局，表头转置和链校验也有开销。哈希操作仍是期望常数，不改善最坏查找界。

表头转置临时空间 O(W²)，不复制 W 份语料或永久索引。payload 的潜在收益是
不再同时持有完整 combined 和所有原 map；rehash 期间新旧 bucket 同时存在，
仍须报告容量代理及训练 HWM，不能仅凭少一个变量宣布峰值一定下降。
容量指标必须显式加上被移出的 accumulator；只求原 outputs 中的表容量会漏算。
选择最大 len 只保证省最多累加入口，不保证其现有 capacity 足够容纳 union。

为隔离原因，应先以同一个 owner 独占输入、连续 reduce/fill 的内核比较
`fresh-map` 与 `reuse-largest`；另保留原 staged 控制作绝对参照。不能把同时
删除全局 reduce→fill 屏障的收益全部归因于复用。原 `owned_fused` 已探索连续
提交，`owned_fused_direct` 已探索旧 key 直接扣减；这里新增的是将 producer
已有的 map 作为总量表，以及在同一个 Delta 中明确区分总计数与局部链头。

## 首轮结果

256 KiB、512 规则、aHash、每格 n=1，同 binary 保留三个控制。W4 英文
staged/fused-fresh/fused-reuse 为 0.05746/0.04542/0.04555 秒，中文为
0.02330/0.01756/0.03155 秒。reuse 的汇总插入确实减少：英文跳过最大输入
表的 23,091 个入口、其余访问 43,591；中文跳过 16,092、其余访问 14,471。
但每个 producer 的任务分配可变，两次运行的局部表入口总数不必相等；这些
计数不能直接相减推断 wall time。W4 训练 HWM 的 staged/fresh/reuse 为
英文 8.60/8.96/9.00 MiB、中文 8.02/8.09/7.60 MiB，亦不支持普遍降低峰值。

这版仍保留新键 expected Vec、链界检查和最终 count 验证；fresh/reuse
的连续 owner 提交都包含表头转置。只有同一连续提交内核的 fresh→reuse
才较接近单独考察复用效果，不能把 staged→reuse 全部差值算给少建一张表。
