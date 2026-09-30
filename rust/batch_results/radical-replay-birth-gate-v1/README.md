# Replay birth correctness gate

Frozen binary SHA-256: `4cba04756256643b73f8f08189544a951947296a51a201a15bce159142699073`.

- Rust lib tests: 27/27 passed. Final changes after tests remove one unused import and apply Clippy equivalents (`mem::take` and a collapsed conditional); no semantic change.
- Final `cargo fmt --check`, `cargo clippy --offline --locked --all-targets -- -D warnings`, and release build passed.
- `differential.json`: 80 independent naive full-trace matches (20 standard inputs × chain/replay × W1/W4, aHash).
- `directed.json`: 16 independent full-trace matches (eight cases × chain/replay, std hash), including AA, adjacent fresh tokens, filtered weighted births, u64 weights, token length 512, cross-region edges, W>N, and tagged-domain fallback.
- Replay physical BirthNodes, grouped nodes and route birth capacity are zero. Semantic birth records, stored postings and merges equal the control. Additional visits equal the second-pass counter. Each final posting slot was filled once.

Reproduce with the Cargo manifest under `rust/experiments/radical/owned_replay_birth`, then `oracle_standard.py` and this directory's `run_directed.py`. Binary is an ignored build artifact; source capsule and hashes accompany the performance screen. No performance conclusion is implied by this gate.
