# Serial integer-hash control

This independent crate measures one engineering change in the existing serial
`combined_filtered` and `combined_filtered_halfword` ablations: the hash-table
builder for integer pair keys. `--integer-hash std` uses Rust's `RandomState`;
`--integer-hash ahash` uses `ahash::RandomState`. Both are selected once for
the whole training call through separate monomorphized kernels. No lookup or
update branches on a runtime hash-mode enum.

## Source boundary

The crate depends on the root `efficient-bpe-rust` public `Prepared`,
`TrainOptions`, `TrainResult`, `TrainError`, and `validate_prepared` contract.
It imports the frozen `rust/src/ablation/backends.rs` and `queue.rs` by
read-only relative path, so corpus endpoint packing, halfword storage,
checked/unchecked access, heap ordering, lazy refresh, and tie-breaking are
the original implementations. It does not edit `rust/src/**`.

`src/index.rs` extracts only the original `Combined<u64>` counted-path
record layout (`u64 frequency` plus `Vec<u32> positions`), parameterizing its
single `HashMap` by `H: BuildHasher + Default`. `src/lib.rs` extracts only
`trainer::run_impl` with `Init::Counted`, `queue_mode=0`, and
`CERTIFIED=false`. It retains both count and eligible-position passes,
weighted `u64` checked additions/subtractions, per-rule fresh-key set,
left-to-right posting iteration, final token decoding, and original error
conditions. The initial counts, persistent Combined index, and per-rule
fresh-key set all use the selected builder; the latter two are also integer
key tables. No parallel code or global index is copied.

The halfword variant has one `Vec<u16>` text and `BitLinks`. A merged token
stores its full `u32` ID as high and low halfwords at the first two physical
positions it occupies. Its original constructor requires the initial
alphabet to fit below 65,536, but fresh merge IDs can use the full `u32`
domain. `--bounds checked|unchecked` selects the original backend
specialization for either layout. The CLI
accepts only `--workers 1`, since this kernel is serial. It emits the same
fingerprint and complete trace encoding as the native ablation CLI, along
with process CPU time and both train-time and final VmHWM. In particular,
train VmHWM is sampled immediately after `train` and before fingerprint and
trace allocations.

## Exactness and comparison

All rule choices remain maximum current weighted frequency, then smallest
packed pair key. A hasher changes iteration order when constructing the
initial heap or draining fresh keys, but heap order and subsequent refresh
recover the same global rule order. Posting order within each key remains
corpus order at initialization and append order after each merge. AA remains
the original sequential left-to-right non-overlapping case. Tests compare
both hashers and both bounds modes to the root native variants on weighted,
AA, adjoining, long-token, empty, and randomized complete traces; the
executor runs the oracle before timing.

For a fair algorithm comparison, use the same fixture, bounds mode,
min-frequency, rule limit, executable build flags, CPU affinity, and
fingerprint. The `std`/`ahash` pair inside this binary isolates the builder
change within serial execution. HashMap seed randomness and hardware AES
features can affect timings. AHash is not a worst-case constant-time
guarantee against adversarial keys. Neither hash mode alters the asymptotic
historical posting work or the serial dependency between greedy rules.
