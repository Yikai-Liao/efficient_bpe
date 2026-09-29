# Inline two-position postings

This archive screens an independent exact pair-owned batch implementation whose posting entry stores up to two `u32` positions inline in a 16-byte `SmallPosting`. Larger lists use uniquely owned heap storage. The merge algorithm, certified batches, owner routing, and lazy/eager heap policies match the frozen owned v5 baseline. The previous owned, counted-delta, and grouped binaries were reused unchanged as controls.

The final source passed 7/7 debug library tests, including layout, inline-to-heap growth, move/take/drop, cross-thread ownership transfer, AA/weighted edge cases, and exact reference comparison. It passed strict Clippy and release build. An independent read-only unsafe review found no blocking ownership or undefined-behavior issue. The Python full-recount oracle matched **80/80 complete rule traces and final token sequences** for inline lazy/eager, W1/W4, and 20 cases. All 20 quick calls additionally matched the frozen native scalar's complete trace, fingerprint, and fixture SHA-256.

Every process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. Both 256 KiB continuous fixtures trained 512 rules at `min_frequency=2`, checked input, `chunk_size=4096`, and Rust release `thin` LTO/one codegen unit. `call_seconds` includes the complete training call after JSON parsing. `train_vm_hwm_mib` is read immediately after training, before trace/fingerprint formatting; it is still a process high-water mark including startup and parsing. Each timing cell is **one short run**, not a stable ranking.

| 256 KiB input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered_halfword`) | 0.08690 / 7.82 | — |
| EN | Frozen owned lazy | 0.09312 / 8.84 | 0.09903 / 8.83 |
| EN | Counted deltas lazy | 0.09300 / 8.77 | 0.07479 / 9.01 |
| EN | Grouped births lazy | 0.08863 / 8.45 | 0.05760 / 8.95 |
| EN | Inline postings lazy | 0.09731 / 8.29 | 0.05830 / 8.89 |
| EN | Inline postings eager | 0.09698 / 8.61 | 0.08629 / 9.07 |
| ZH | Best direct scalar (`combined_filtered_halfword`) | 0.03203 / 6.00 | — |
| ZH | Frozen owned lazy | 0.04878 / 8.39 | 0.03324 / 9.12 |
| ZH | Counted deltas lazy | 0.05621 / 8.27 | 0.03141 / 8.99 |
| ZH | Grouped births lazy | 0.05819 / 8.18 | 0.03468 / 8.97 |
| ZH | Inline postings lazy | 0.03931 / 7.14 | 0.02934 / 7.94 |
| ZH | Inline postings eager | 0.03690 / 7.11 | 0.02889 / 7.82 |

All controls and inline variants agree exactly on batch count, maximum width, generated/stored births, and historical posting visits: EN 69 batches, width 22, 284,407 generated births, 269,984 stored births, 202,877 visits; ZH 87, width 26, 51,001, 40,975, 29,670. The layout does not change rule order or rewrite work. The EN frozen-owned W4 call, 0.099 s, is high relative to its other archived 256 KiB samples (about 0.056–0.064 s); it must not be used to infer a large inline speedup.

At the end of W4 training, EN had 4,243 inline and 12,174 heap-backed eligible pair keys; ZH had 5,717 inline and 7,335 heap-backed keys. The inline keys each held exactly two positions on these `min_frequency=2` fixtures. Actual allocated heap-position capacity was 371,496 EN and 70,780 ZH, compared with the old Vec-backed baseline's 388,468 and 93,648 slots. In this short screen, training VmHWM was materially lower for ZH inline lazy (7.94 versus old 9.12 MiB) but not for EN (8.89 versus 8.83 MiB). The smaller entry size and fewer small Vec allocations are concrete layout effects; process RSS and sub-0.1-second timing need larger controlled inputs before a speed claim.

`differential.json`, `native-quick.jsonl`, `quick.jsonl`, their sidecars, and `checks.json` contain validation, exact commands, affinities, binary/fixture hashes, and raw observations. `run_differential.py` and `run_quick.py` reproduce the focused checks from ignored binaries in `rust/target/reruns/`. `new-source-snapshot.tar.gz` and `new-source-hashes.json` preserve the new crate at build time, including `small_posting.rs`; earlier archive snapshots preserve the controls. Only `owned_inline/DESIGN.md` received a documentation-only validation update afterward; its before/after hashes are in `checks.json`, while lib/main/Cargo and the binary remain unchanged. Large binaries are not staged. No standalone 4 MiB or 16 MiB inline timing was run; the planned joint candidate screen will compare inline with grouped and their combination under the same budget.
