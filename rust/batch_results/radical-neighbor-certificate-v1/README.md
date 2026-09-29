# Exact birth-neighbor certificate gate

`owned_neighbor_sketch` compares the original type-conflict certificate with `birth-neighbor64`, an additional conservative summary of final birth-edge neighbors. The same frozen binary runs both modes. This archive is a correctness gate and one-repeat 256 KiB mechanism screen; no large-input performance screen was run.

The final source passed 14/14 debug library tests, strict all-target Clippy, and release build. A Python full-recount oracle matched **80/80 complete merge-rule traces and final token sequences** (20 cases × two certificates × W1/W4). All eight quick runs also matched the native reference's complete trace and fingerprint. Exact batch boundaries differ, so intermediate birth and posting counts can legitimately differ; the complete output trace is the semantic requirement.

| Input | Workers | Type call s | Sketch call s | Type→sketch rounds | Negative admissions | Extra sketch visits | Type→sketch HWM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN | 1 | 0.0819 | 0.0941 | 69→60 | 15 | 530,725 | 8.16→8.68 |
| EN | 4 | 0.0476 | 0.0538 | 69→60 | 15 | 530,725 | 8.61→9.06 |
| ZH | 1 | 0.0380 | 0.0407 | 87→82 | 9 | 149,188 | 7.04→7.99 |
| ZH | 4 | 0.0305 | 0.0266 | 87→82 | 9 | 149,188 | 7.91→8.84 |

The sketch expands a valid exact batch: EN maximum width rises 22→27, while ZH remains 26; conservative positive stops number 56 EN and 58 ZH. EN selected/final summary popcounts are 16,244/136,385, versus 10,915/72,034 ZH. The retained `Entry` tuple grows from 32 to 40 bytes. Final summary payload is 131,336 bytes EN and 104,416 bytes ZH; this is not the whole map's allocated memory. Extra summary reads comprise EN 260,980 initial plus 269,745 birth visits, and ZH 108,314 plus 40,874. Ordinary posting visits stay equal within each input. EN generates 284,407→284,169 birth records and ZH 51,001→50,900 because batch boundaries change.

One quick run per cell cannot establish a stable speed ordering. The certificate reduces round count in both inputs, yet its extra reads, bit operations, and memory offset some saved barriers: EN is slower in this run, while ZH W4 is faster and ZH W1 slightly slower. These data justify keeping the sketch as an exact experimental option, not replacing the type certificate by default. No 4 MiB or full-matrix result is claimed.

Both modes used `--heap-policy lazy`, a 4,096-position task chunk, and 512 requested rules. W1 was pinned to CPU 5; W4 to CPUs `[0,1,2,5]`. `call_seconds` covers the full training call, and `train_vm_hwm_mib` is the process high-water mark sampled before fingerprint and trace construction. `call_cpu_seconds` and all phase metrics remain in raw `quick.jsonl`; `summary.json` extracts the certificate and memory counters.

`new-source-snapshot.tar.gz` and `new-source-hashes.json` preserve the independent crate. `shared-source-provenance.json` identifies the byte-identical shared Rust source and base commit needed to reconstruct it. Frozen binaries live in ignored `rust/target/reruns/radical-neighbor-certificate-v1/`; hashes are in `checks.json`. The raw quick and oracle results, environment sidecar, and scripts are included. Run `finalize.py` to verify provenance and completeness without repeating training.
