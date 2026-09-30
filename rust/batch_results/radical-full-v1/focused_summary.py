"""Summarize the focused window; keep the stopped matrix exploratory."""

import collections
import json
from pathlib import Path
import statistics

from run import OUT, ROOT, RUST, sha, verify_row


FIELDS = ["call_seconds", "call_cpu_seconds", "train_vm_hwm_mib", "outer_call_seconds",
          "mean_occupied_cores", "init_seconds", "plan_seconds", "select_seconds",
          "frequency_reduce_seconds", "birth_group_fill_seconds", "replay_fill_seconds",
          "posting_visits", "grouped_birth_nodes", "replay_count_peak_heap_bytes"]


def summarize(rows):
    result = {"n": len(rows)}
    for field in FIELDS:
        values = [r[field] for r in rows if field in r]
        if len(values) == len(rows):
            result[field] = {"median": statistics.median(values),
                             "min": min(values), "max": max(values), "values": values}
    return result


def main():
    destination = OUT / "focused"
    progress = json.loads((destination / "progress.json").read_text())
    assert progress["status"] == "passed"
    config = json.loads((OUT / "config.json").read_text())
    fixtures = {r["case_id"]: r for r in json.loads((OUT / "fixtures.json").read_text())}
    references = json.loads((OUT / "references.json").read_text())
    rows = [json.loads(line) for line in (destination / "measurements.jsonl").read_text().splitlines()]
    original = [json.loads(line) for line in (OUT / "measurements.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (OUT / "warmups.jsonl").read_text().splitlines()]
    assert len(rows) == 36 and len(original) == 135 and len(warmups) == 135
    for row in original + warmups + rows:
        verify_row(row, config, fixtures, (row["case_id"], row["mode"], row["workers"]), references)
    assert all(r["complete_trace_match"] for r in warmups)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["case_id"], row["mode"], row["workers"]].append(row)
    assert len(grouped) == 12
    cells = []
    for (case, mode, workers), values in sorted(grouped.items()):
        assert {r["repeat"] for r in values} == {1, 2, 3}
        primary = [r for r in values if r["repeat"] in (2, 3)]
        assert len(primary) == 2
        assert all(r["measurement_window"] == "focused_completion" for r in primary)
        cells.append({"case_id": case, "mode": mode, "workers": workers,
                      "primary_focused_repeats_2_3": summarize(primary),
                      "all_three_observations": summarize(values)})
    env = json.loads((OUT / "environment.json").read_text())
    for path, digest in env["workspace_source_sha256"].items():
        assert sha(ROOT / path) == digest
    for case in {c["case_id"] for c in cells}:
        assert sha(RUST / fixtures[case]["file"]) == fixtures[case]["fixture_sha256"]
    for mode in {c["mode"] for c in cells}:
        assert sha(ROOT / config["modes"][mode]["binary"]) == config["modes"][mode]["binary_sha256"]
    unique_measured = original + [r for r in rows if r["measurement_window"] == "focused_completion"]
    result = {
        "status": "passed", "stopped_original_matrix_rows": len(original),
        "focused_new_calls": progress["new_calls"], "focused_reused_rows": progress["reused_rows"],
        "unique_measured_calls": len(unique_measured), "primary_rows": 24, "primary_n_per_cell": 2,
        "focused_elapsed_seconds": progress["elapsed_seconds"],
        "full_fingerprint_matches": len(unique_measured), "complete_trace_warmups": len(warmups),
        "max_sampled_vm_swap_kib": max(r["sampled_peak_vm_swap_kib"] for r in unique_measured),
        "max_process_major_faults": max(r["process_major_faults"] for r in unique_measured),
        "cells": cells,
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {"serial_cf32_checked": "CF32 checked 串行 W1",
             "serial_cf16_unchecked": "CF16 unchecked 串行 W1",
             "owner_ahash": "旧 owner aHash W4", "cuts_adaptive": "自适应切分 W4",
             "birth_chain": "出生链 W4", "birth_replay_inline": "内联计数重放 W4"}
    lookup = {(c["case_id"], c["mode"]): c["primary_focused_repeats_2_3"] for c in cells}
    lines = ["# 16 MiB、32,000 合并：核心方案复核", "",
             "原计划矩阵已停止：200 个配置、1,000 次正式测量、200 次完整轨迹预热超出了本次对比需要。原有 135 条正式结果及 135 次完整轨迹校验保留为探索性数据，不能称为完成了整个矩阵。", "",
             f"收敛后的执行新增 {progress['new_calls']} 次训练，用时 {progress['elapsed_seconds']:.1f} 秒。12 个配置各保留三条观测，其中 8 条来自原窗口；主表仅使用补测窗口的第 2、3 轮，共 24 条，每格 n=2，避免旧窗口的较高耗时影响排序。其余 12 条保留作敏感性观察，见 summary.json。", "",
             "输入为同 revision 的英文、中文 Wikipedia 各约 16 MiB，连续单 piece、权重 1，不预分词；每份均实际完成 32,000 次合并，最低频率 2。英文/中文字符数为 16,709,040/6,806,761；字节数相近不代表同等位置工作量。", "",
             "源码及二进制固定于原记录的 39c4a265d17af815621c352af3d605f3a8fc6eb4；使用已门控 release 二进制，补测前后核对源码、输入和二进制哈希。进程逐个运行，W1 限 CPU 5，W4 限 CPU 0/1/2/5。机器提供六个 KVM vCPU、一个可见 NUMA node。", "",
             "## 完整训练调用耗时", "",
             "单位为秒，括号为两次观测范围。包含初始化、线程池与训练收尾；不包含 JSON 读取、结果指纹编码和 trace 写出。子进程完整时间另存。n=2 用于本轮方案比较，不能证明普遍稳定排序或统计显著性。", "",
             "| 方案 | 英文中位数（范围） | 中文中位数（范围） |",
             "|---|---:|---:|"]
    for mode, name in names.items():
        values = []
        for case in ("en-16m-continuous", "zh-16m-continuous"):
            v = lookup[case, mode]["call_seconds"]
            values.append(f"{v['median']:.3f}（{v['min']:.3f}–{v['max']:.3f}）")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines += ["", "## CPU 和训练后内存高水位", "",
              "内存单位 MiB；是训练后、输出前的进程 VmHWM，包含输入解析此前的高水位。它不是纯训练载荷，也不是算法的硬内存上界。CPU/wall 是平均占用核数，不是多核加速比。", "",
              "| 方案 | 英文 CPU 秒 / CPU÷wall / HWM | 中文 CPU 秒 / CPU÷wall / HWM |",
              "|---|---:|---:|"]
    for mode, name in names.items():
        values = []
        for case in ("en-16m-continuous", "zh-16m-continuous"):
            v = lookup[case, mode]
            values.append(f"{v['call_cpu_seconds']['median']:.3f} / {v['mean_occupied_cores']['median']:.2f} / {v['train_vm_hwm_mib']['median']:.1f}")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines += ["", "## 本轮判断", ""]
    for case, label in (("en-16m-continuous", "英文"), ("zh-16m-continuous", "中文")):
        timing = {m: lookup[case, m]["call_seconds"]["median"] for m in names}
        parallel = min((m for m in names if not m.startswith("serial")), key=timing.get)
        serial = min((m for m in names if m.startswith("serial")), key=timing.get)
        lines.append(f"- {label}本窗最快并行候选为 {names[parallel]}，{timing[parallel]:.3f} 秒；相对本窗最快直接串行为 {timing[serial]/timing[parallel]:.2f}×。内联重放/出生链的耗时比为 {timing['birth_replay_inline']/timing['birth_chain']:.2f}。这些是不同实现的整次调用比较，不能把差值全部归因于单个改动。")
    lines += ["", "内联重放不是已确认的通用最优。它删除物理 BirthNode，却增加第二次历史位置扫描、写入目录组织和保留 selected 的时间；计数内联降低局部元数据成本，无法保证整次调用更快。chain 与 inline 来自同一 replay-counts 二进制，本窗对照能直接检验出生填充模式。adaptive 与 chain 来自不同原型，不能将两者差距解释成 adaptive 开关的因果收益；本轮没有补跑 fixed 控制。", "",
              "本轮没有为每种并行模式补齐 W1，因此不报告自身 W1→W4 扩展率。公平直接串行→W4 也没有达到 3×，总体研究目标仍未完成。CF16 unchecked 仅在本次符号 ID 域可容纳的输入中参与比较，不能无条件推广到更大词表。", "",
              f"全部 {len(unique_measured)} 次独立正式调用均匹配完整规则、频率及最终 token 指纹；原窗口 {len(warmups)} 个配置另有完整 trace 相等验证。补测没有重做大 trace 序列化。正式进程最大采样 VmSwap 为 {result['max_sampled_vm_swap_kib']} KiB，最大 major faults 为 {result['max_process_major_faults']}。", "",
              "原配置、输入 manifest、冻结源码/二进制出处和运行环境见同目录。focused/measurements.jsonl 保留来源窗口、实际 CLI、每次 wall/CPU/HWM、完整指纹及算法指标；focused/summary.json 包含主窗口与全部三条观测的独立汇总。", "",
              "复现补测：在该目录输入和已冻结二进制均存在时，先保存或移走已有 focused/ 输出，再从仓库根目录运行 `python3 rust/batch_results/radical-full-v1/focus.py`，随后运行 `python3 rust/batch_results/radical-full-v1/focused_summary.py`。原 run.py/summarize.py/audit.py/plot.py 属于已停止的完整矩阵，不用于此次收敛结果。", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    print("\n".join(lines[17:29]))


if __name__ == "__main__":
    main()
