# AA radix sort gate and light screen

`owned_aa_radix` keeps the exact owner trainer and offers `--aa-sort std|radix`. The std control calls Rayon `pool.install(par_sort_unstable)`; radix is a serial in-place `u32` sort. Both use the **same binary**, aHash, lazy heap, and the same prepared input. The final source passed 13/13 debug library tests, strict all-target Clippy, release build, and **160/160** complete Python recount oracle traces (20 cases × two sort modes × two hash modes × W1/W4).

The authorized quick screen used a dense unary piece, an alternating AB piece that crosses 4,096-position task boundaries, and one natural EN control. All **12 complete merge traces and final tokens** matched. Each cell ran once under a fixed total CPU budget, so timing is diagnostic.

| Input | Workers | Std call s | Radix call s | AA positions | Std sort ms | Radix sort ms | Std→radix HWM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unary A ×65,536 | 1 | 0.0120 | 0.0122 | 131,054 | 0.347 | 1.426 | 4.79→4.74 |
| Unary A ×65,536 | 4 | 0.0156 | 0.0117 | 131,054 | 1.293 | 2.042 | 4.53→4.54 |
| Alternating AB ×65,536 | 1 | 0.0105 | 0.0115 | 65,519 | 0.241 | 1.068 | 3.68→3.70 |
| Alternating AB ×65,536 | 4 | 0.0074 | 0.0085 | 65,519 | 0.960 | 1.066 | 4.00→4.06 |
| Natural EN 256 KiB | 1 | 0.0737 | 0.0541 | 250 | 0.013 | 0.020 | 8.38→8.40 |
| Natural EN 256 KiB | 4 | 0.0353 | 0.0411 | 250 | 0.038 | 0.030 | 8.76→8.76 |

The measured AA-sort share of the full call is 2.9%/8.3% for unary std W1/W4, 11.7%/17.5% for unary radix, 2.3%/13.0% for alternating std, and 9.3%/12.5% for alternating radix. Natural EN sorts only 250 positions and spends below 0.1% of its call in AA sorting; its whole-call differences cannot be attributed to this sort. In the two AA-heavy fixtures, radix's **sort phase itself** is slower than the Rayon std phase in all four cells. Unary W4's overall radix call happens to be faster, but its sort phase is slower, so that single overall timing does not establish a radix benefit. The aHash builder is fixed; this screen does not test another hasher or larger input.

The unary and AB pieces each have 65,536 input positions, weight 2, and at most 128 rules. Natural EN uses 262,144 source bytes and 512 rules. W1 ran on CPU 5; W4 on `[0,1,2,5]`. `call_seconds` covers the complete training call, `aa_sort_seconds` measures the sort subphase, and `train_vm_hwm_mib` is the process high-water mark sampled after training. CPU time, radix pass/swap counts, AA shares, and all raw outputs are in `quick.jsonl`; `summary.json` presents the comparisons.

`new-source-snapshot.tar.gz` and `new-source-hashes.json` preserve the independent crate; `shared-source-provenance.json` identifies the byte-identical shared Rust source and base commit. Frozen release binaries remain in ignored `rust/target/reruns/radical-aa-radix-gate-v1/`. `checks.json` records hashes and validation. Run `finalize.py` to recheck source, binary, and result completeness without new timing.
