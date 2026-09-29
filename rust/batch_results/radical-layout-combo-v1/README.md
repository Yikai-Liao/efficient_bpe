# Final limited layout screen: grouped births plus inline postings

This archive compares four exact greedy BPE layouts on the same certified-batch, pair-owned trainer: frozen `owned_lazy`, `grouped_lazy` (birth positions grouped by final pair key), `inline_lazy` (two positions inline per posting key), and the new `combo_lazy` that combines grouping with inline postings. It also samples the best previously identified direct scalar for each language. Other researched probe, counted-delta-only, and shard variants remain archived separately; no full matrix was restarted.

The final combo source passed 9/9 debug library tests, including empty reserved heap, inline/heap growth and take/drop, all `u32` payload bits, AA/weighted exact cases, and both grouped chain/count and token-key invariants. It passed strict Clippy and release build. Two independent read-only reviews found no blocking unsafe or exactness issue. The Python full-recount oracle matched **80/80 complete rule traces and final token sequences** for combo lazy/eager, W1/W4, and 20 cases. All 16 short quick calls matched the frozen native scalar's complete trace, fingerprint, and fixture SHA-256. Every call in the later 36-call screen matched the previously validated complete-training fingerprint and fixture SHA-256.

The entire process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. All binaries used Rust release `thin` LTO and one codegen unit, checked input, `chunk_size=4096` for the radical variants, and `min_frequency=2`. The 256 KiB continuous inputs trained 512 rules. The final 4 MiB continuous inputs trained 3000 rules; each of four layouts ran W1/W4 twice and each direct scalar ran twice, **36 training calls total**. The second round reversed the first seeded-shuffle order. `call_seconds` covers the complete training call after JSON parsing. `train_vm_hwm_mib` is sampled immediately after training, before fingerprint formatting; it remains a process high-water mark including startup and parsing.

The 256 KiB quick screen used one call per cell:

| Input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered`) | 0.07719 / 8.41 | — |
| EN | Owned lazy | 0.09930 / 8.88 | 0.07551 / 9.07 |
| EN | Grouped lazy | 0.09996 / 8.12 | 0.06165 / 8.57 |
| EN | Inline lazy | 0.11216 / 8.49 | 0.06672 / 8.64 |
| EN | Combo lazy | 0.08755 / 8.05 | **0.05461 / 8.34** |
| ZH | Best direct scalar (`combined_filtered`) | 0.03378 / 6.17 | — |
| ZH | Owned lazy | 0.05051 / 8.29 | 0.03600 / 8.93 |
| ZH | Grouped lazy | 0.04799 / 7.97 | 0.03068 / 8.87 |
| ZH | Inline lazy | 0.04661 / 7.15 | 0.02576 / 8.00 |
| ZH | Combo lazy | 0.04051 / 6.87 | **0.02457 / 7.62** |

The final 4 MiB table reports the **median of two** call times, with the min–max observations in brackets. These ranges are not confidence intervals or proof of a stable ranking.

| Input | Variant | W1 seconds [range] | W4 seconds [range] | W4 training VmHWM MiB [range] |
|---|---|---:|---:|---:|
| EN | Direct scalar (`combined_filtered_halfword`) | 1.676 [1.631–1.722] | — | — |
| EN | Owned lazy | 2.280 [2.241–2.319] | 0.851 [0.850–0.853] | 102.57 [97.97–107.18] |
| EN | Grouped lazy | 1.931 [1.818–2.044] | 0.726 [0.679–0.772] | 105.63 [104.07–107.20] |
| EN | Inline lazy | 2.247 [2.175–2.319] | 0.895 [0.809–0.982] | 97.82 [95.83–99.82] |
| EN | Combo lazy | 1.845 [1.818–1.873] | **0.680 [0.659–0.701]** | **94.32 [94.09–94.55]** |
| ZH | Direct scalar (`combined_filtered`) | 1.039 [0.990–1.089] | — | — |
| ZH | Owned lazy | 1.047 [0.989–1.105] | 0.503 [0.480–0.525] | 115.47 [115.47–115.47] |
| ZH | Grouped lazy | 1.058 [1.057–1.059] | 0.444 [0.430–0.459] | 116.19 [115.57–116.80] |
| ZH | Inline lazy | 0.993 [0.978–1.008] | 0.493 [0.417–0.569] | 98.17 [96.27–100.06] |
| ZH | Combo lazy | 0.850 [0.782–0.919] | **0.404 [0.400–0.409]** | 100.20 [99.55–100.84] |

Ratios of these medians put combo W4 at 2.47× EN and 2.57× ZH the direct scalar throughput, and 1.25×/1.24× faster than frozen owned W4. Its W1-to-W4 ratios are 2.71× EN and 2.10× ZH under the stated total CPU budgets. The two combo W4 times were close in both languages; several other cells were less consistent, especially inline EN/ZH W4. A larger repeated sample would be needed for a statistical speed claim. Compared with owned, combo's median training VmHWM is about 8.25 MiB lower EN and 15.27 MiB lower ZH; these are process-peak differences, not net allocator measurements. Inline alone had the lowest ZH peak in this screen (98.17 MiB versus combo's 100.20 MiB), so the combination does not minimize memory in both languages.

All four layouts produce exactly the same batch count, maximum width, generated/stored births, and historical posting visits: EN 251 batches, width 47, 5,705,752 births, 4,482,863 visits; ZH 275, width 50, 1,274,766 births, 806,617 visits. Grouping changes how birth positions reach their final posting key, and inlining changes their storage, without changing the greedy rule order. In the W4 calls, the old birth-decode timer's EN median was 0.221 s; grouped and combo replaced it with 0.046/0.056 s grouped-fill timers. On ZH the old 0.066 s decode became 0.027/0.029 s grouped fill. Decode and grouped fill are alternative phases in different implementations. Some other reported timers nest within broader phases—for example, initial route/owner work within initialization—so summing every stage does not reconstruct the full-call time.

The final eligible-pair map had 215,430 inline posting keys and 121,270 heap-backed keys EN, and 347,954 inline / 106,945 heap keys ZH. Combo's actual allocated heap-position capacity was 5.21 million EN and 1.63 million ZH, compared with the old Vec-backed layout's 7.35 million and 3.21 million slots. The inline keys occupy their entry payload without separate two-position Vec allocations. Temporary grouped birth nodes are 8 bytes each, so this representation is a memory tradeoff rather than an across-the-board reduction in every buffer.

`differential.json`, `native-quick.jsonl`, `quick.jsonl`, `screen.jsonl`, their sidecars, `summary.json`, and `checks.json` preserve raw observations, both repeats, exact commands, affinities, binary/fixture hashes, stage metrics, and assertions. `run_differential.py`, `run_quick.py`, and `run_screen.py` reproduce the focused work from frozen ignored binaries in `rust/target/reruns/`. `new-source-snapshot.tar.gz` preserves the combo crate; referenced prior snapshots preserve all controls. Large binaries are not staged. No 16 MiB input or full matrix was run.
