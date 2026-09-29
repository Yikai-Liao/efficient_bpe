# 下一原型：只替换非 AA 的跨区访问协议

待实现。先使用[边界快照审查](REGION_BOUNDARY_SNAPSHOT_REVIEW.md)的严格
协议，独立克隆 `owned_region_fused`，只比较 atomic-region 与
snapshot-region 的非 AA 批次。选规则、hash、owner 提交、posting 投影和
AA 路径全部相同，避免把新的 owner 或位图改动混进来。这个实验首先验证
能否将并发访问收敛到独占 region 与 O(T) 边界通信，不预设速度收益。

最小实现可以继续使用 `Vec<AtomicU32>` 作为 backing storage，因而不必
同时重写初始化、AA、最终解码及域回退。进入非 AA 独占阶段前，先复制
数值快照，再按物理 cuts 将 corpus 拆成独占的 `&mut [AtomicU32]`。
每个 region 任务对本地格通过
[`AtomicU32::get_mut()`](https://doc.rust-lang.org/core/sync/atomic/type.AtomicU32.html#method.get_mut)
读取或写入普通 u32；该接口由独占可变借用保证不会与 atomic 访问并发，不需要重新解释
整个 slice 的 raw pointer，也不需要另一份 N 大小的数组。一个私有
region accessor 持有本地 slice、全局起点和边界快照；read 返回复制的
u32 值，可接收 `&mut self` 以使用 get_mut。不要为了保留原函数签名而
偷偷留下整个 corpus 的共享引用。

远端读只允许命中本 region 上下 cut 的旧 token 描述符，失配即返回错误。
远端写只记录坐标和值；任务 join 后由协调者应用至多 2(T−1) 次写，随后
更新 anchor、完成 owner commit，才进入下一批。AA 暂保留现有全局规划
和原子写回，完成后同样更新 anchor。这是非 AA 访问协议的消融，不能把
实验命名或报告成所有阶段都已移除 Atomic。

严格 cuts 初版取 T=min(W,N)，其逻辑 region 输出数可能少于 key-owner
数 W，必须在初始化、commit 与 route 头计量中区分 T 和 W。或者一开始
保持 T=W 并完整定义零长度 region；不能仅跳过空任务却让输出编号错位。
HEAD 域不满足时整次调用回已有 dynamic/two-pass，报告 effective 模式。

在完整 trainer 前，先用可执行小模型验证每种端点查询与写回、anchor
更新和邻区处理顺序；尤其覆盖待写右起点尚未清零、长 token 跨多 cut、
sentinel、stale posting、AA→非 AA 切换及空物理区间。模型过后再接入
Rust 全轨迹 oracle。测量普通本地读写次数、边界查询与快照容量、延迟
写数、各 region 工作量、完整 call/CPU/HWM；局部读路径的范围检查如确为
热点，再在明确证明的范围内单独消融 unchecked 访问，不同时修改协议。
