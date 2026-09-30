# 自适应切区与出生位置重放：40 调用筛选

状态：40/40 完整轨迹一致，标准/定向门控全部通过。结论见[研究报告](../../ADAPTIVE_REPLAY_REPORT.md)。这是 n=2 的小样本筛选，不是最终性能排名；没有执行完整旧矩阵或 4 MiB 扩展。

- `screen.jsonl`：逐调用完整训练 wall、进程 CPU、训练后/trace 前 VmHWM、模式与所有内部指标、CLI、输入/二进制 SHA、CPU 亲和性。自然语料和同窗口直接串行完整轨迹比较；AB 与同窗 chain 和旧冻结 fingerprint 比较。
- `summary.json`：每格原始两轮值、独立中位数、自身 W1→W4 和公平直接串行/W4，后两种比值明确区分。
- `frozen-config.json`：同 binary fixed/adaptive 和 chain/replay 的控制参数。执行器先核对门控及 binary SHA，再核对源码和输入。
- `new-source-snapshot.tar.gz` / `new-source-hashes.json`：两 crate 的 Rust/Cargo/DESIGN 冻结源码。`shared-source-provenance.json`：26 个共享文件与基座 c1d8fc3 逐字一致、编译器版本和重建说明。
- 两个正确性门控分别在 `../radical-adaptive-cuts-gate-v1`、`../radical-replay-birth-gate-v1`。`oracle_standard.py` 复用独立 naive 全量重计数参考。

256 KiB EN/ZH，512 规则。W1 CPU 5；W4 CPU 0/1/2/5。aHash/lazy/global/tagged-fused/atomic region k1；直接串行为 CF32/aHash/checked。seed 20260930 随机交错，第二轮反转第一轮次序。完整 call 包含线程池、初始化及收尾；不含输入 JSON 读取和 trace 写入。n=2 中位数为两值算术中点，原始范围不是置信区间。

adaptive 的访问最大值累计和 EN/ZH 降 2.25%/27.86%，全调用速度未一致改善；原 replay 虽删除出生节点，自然语料 W4 较慢，AB 较快。总 RSS 未证明一致下降。不要跨两个不同 family 的控制混算某个开关的因果效果。

复现：在基座上覆盖源码 capsule，按 Cargo manifest release 构建（`--offline --locked --target-dir rust/target`），将产物放到 config 中的 ignored reruns 路径。编译环境改变时须记录新 binary SHA；不要覆写本次归档。在新的输出目录配置并运行门控、`snapshot_sources.py`、`run_screen.py` 和 `summarize.py`；脚本对归档输出使用独占创建，防止静默覆盖。
