# Boundary-topology microbenchmark summary

Source: `formal/topology.jsonl` (SHA-256 `9f9cf50d05ad4b6a7719ea6c13cde973e4f8ee2ed5e4ca8d313ccd0f937f970a`). Reproduce with `python3 rust/tools/ablation_micro_summarize.py` from the repository root.

Validated 3300 rows in 660 groups; each variant × length × pattern × bounds group has five repetitions. There are 3150 timed rows and 150 explicit skips. All 30 length × pattern cases agree on checksum across every available variant, bound mode, and repetition. Actual positions and operation counts match the fixture formula below; buffer capacity is never below logical buffer size.

The fixture uses `blocks = max(1, floor(131072 / length))`, `positions = blocks × (length + 1) + 1`, and `operations = blocks × (length − 1)`. The extra positions are the initial sentinel and one separator per block. Timings below are the internal `seconds` field, with min/median/max over five isolated runs; process startup is excluded.

This microbenchmark replays a precomputed trace and times boundary inspection and merge only. It excludes BPE frequency counting, candidate selection, occurrence indexing, fixture creation, and final traversal. It does not rank complete trainers. The two 1-byte components have a different operation contract from the ID-bearing backends, and their timings should not be used as direct full-trainer speed comparisons.

Logical/capacity figures in the next table use length 32; exact byte counts for every group are in the JSON. They count backend vector buffers only, excluding Vec headers, allocator overhead, and the prepared trace.

| Variant | Logical bytes / position | Capacity bytes / position | Timed contract |
|---|---:|---:|---|
| `bitmap_u32` | 4.1876 | 4.1876 | `inspect_pair+merge_known` |
| `bytespans` | 1.0000 | 1.0000 | `prev+next+next+checked_merge_no_id` (component only) |
| `endpoints` | 4.0000 | 4.0000 | `inspect_pair+merge_known` |
| `full_clear` | 4.0000 | 4.0000 | `inspect_pair+merge_known` |
| `h25` | 2.5000 | 2.5000 | `inspect_pair+merge_known` |
| `h3` | 3.0000 | 3.0000 | `inspect_pair+merge_known` |
| `halfword` | 2.1876 | 2.1876 | `inspect_pair+merge_known` |
| `lean` | 4.0000 | 4.0000 | `inspect_pair+merge_known` |
| `linked12` | 12.0000 | 12.0000 | `inspect_pair+merge_known` |
| `linked16` | 16.0000 | 16.0000 | `inspect_pair+merge_known` |
| `u8_only` | 1.0000 | 1.0000 | `prev+next+next+merge_known_no_id` (component only) |

## Length boundaries

`ByteSpans` remains exactly 1 logical and capacity byte per original position at length 255, 256, 65,535, and 65,536. Its long spans use multiple endpoint tag bytes within that one-byte-per-position array; these figures exclude token IDs. The `bounds` switch has no effect on either component-only backend, whose calls remain checked. `u8_only` explicitly skips all five tested lengths above 255 (`u8_only supports length <=255`: 150 rows).

| Length | Actual positions | Operations | ByteSpans bytes | u8_only |
|---:|---:|---:|---:|---|
| 32 | 135,169 | 126,976 | 135,169 | measured |
| 63 | 133,121 | 128,960 | 133,121 | measured |
| 64 | 133,121 | 129,024 | 133,121 | measured |
| 128 | 132,097 | 130,048 | 132,097 | measured |
| 255 | 131,585 | 130,556 | 131,585 | measured |
| 256 | 131,585 | 130,560 | 131,585 | skipped |
| 1,024 | 131,201 | 130,944 | 131,201 | skipped |
| 8,192 | 131,089 | 131,056 | 131,089 | skipped |
| 65,535 | 131,073 | 131,068 | 131,073 | skipped |
| 65,536 | 131,075 | 131,070 | 131,075 | skipped |

## Chain clearing cost

On the chain trace, `full_clear` fills the entire merged interior on each operation. `lean` updates two or three endpoint cells. Both use the same four-byte ID array, and operation counts remain around 127k–131k across lengths. For a chain of length L, full clearing does Θ(L²) writes per block and lean does Θ(L); with this approximately fixed-size corpus, that means Θ(NL) versus Θ(N) boundary writes. The measured medians reflect this difference; they are not a prediction of whole-trainer speed.

| Length | Full clear checked (ms) | Lean checked (ms) | Ratio |
|---:|---:|---:|---:|
| 32 | 3.271 | 1.273 | 2.6× |
| 63 | 3.327 | 1.332 | 2.5× |
| 64 | 3.257 | 0.986 | 3.3× |
| 128 | 3.630 | 1.082 | 3.4× |
| 255 | 4.410 | 1.046 | 4.2× |
| 256 | 3.700 | 1.108 | 3.3× |
| 1,024 | 5.731 | 1.035 | 5.5× |
| 8,192 | 18.151 | 1.336 | 13.6× |
| 65,535 | 493.766 | 1.111 | 444.4× |
| 65,536 | 483.544 | 1.058 | 457.0× |

Each cell's min/median/max, operation count, buffer and capacity bytes, and process peak RSS are in `topology-summary.json`. The `vm_hwm_mib` field is process peak RSS, not backend buffer size; fixture and runtime allocations also contribute. The selected layouts and timed contracts are defined in [`micro.rs`](../src/ablation/micro.rs), [`backends.rs`](../src/ablation/backends.rs), and [`spans.rs`](../src/ablation/spans.rs).
