# Lean endpoint update: correctness contract

`LeanEndpoints` is correct for the fused trainer's occurrence-index workflow, provided token IDs are strictly fresh (never reused) and `inspect_pair(pos, a, b)` is called only for a position previously recorded as a real start of pair `(a, b)`. It is not a general-purpose `alive(pos)` predicate. The code deliberately leaves some old endpoint IDs in the corpus.

## Why historical pair starts remain safe to inspect

The trainer seeds an occurrence list from actual adjacent-token starts. Later it adds only the two starts that a merge can create: the pair beginning at the old predecessor and the pair beginning at the new token. Thus every indexed position was once the start of an actual token with the queried left ID.

Consider a historical start `p` for token ID `a` as merges proceed. If that token is consumed as the left token, the merge overwrites `corpus[p]` with its fresh result ID. If it is consumed as the right token, `p` is the right token's start. For a one-position right token, the update writes the fresh result ID at `p`; for a longer right token, it writes zero there. In either case `corpus[p] != a` afterward. Fresh IDs ensure that a later token cannot make this old comparison true again. Therefore, at any queried historical start, `corpus[p] == a` means the original token at `p` is still live.

For such a live start, `p + token_len[a]` is exactly the next token start. The inherited `inspect_pair` checks that this position contains `b`, so it recognizes exactly the still-live adjacent pair. The proof depends on the restricted history of queried positions; endpoint contents elsewhere do not establish liveness.

## Why the reduced writes preserve endpoints

Let `right` be the right-token start and `after` the following token start or EOF. The merged token begins at `pos` and ends at `after - 1`; its previous live neighbor's endpoint at `pos - 1` remains unchanged.

When the right token has length one, `right == after - 1`: writing `new_id` once at `right` establishes both the consumed token's new start boundary and the merged token's final endpoint. When the right token is longer, its start at `right` must be cleared so it cannot be mistaken for a live token start, and `after - 1` must receive `new_id` as the merged token's endpoint. In both cases `corpus[pos] = new_id` establishes the new start. The old left-token endpoint at `right - 1` is interior to the merged token and need not be cleared. Any historical pair start that was exactly that right-token start has already been invalidated by the clear-or-overwrite rule above.

The right-boundary test is `after - right == 1`, so it handles a singleton right token regardless of the left token's length. No extra bitmap or alive array is needed. A merge therefore performs two corpus stores for a singleton right token, and three for a longer right token.

## Explicit limit: not a general alive check

For corpus `[0, 1, 2, 3, 4, 0]`, first merge `(1, 2)` into ID 5 of length 2, then merge `(5, 3)` into ID 6 of length 3. The lean writes leave `corpus[2] == 5`, the stale endpoint of the old left token. Position 2 was not a historical start for `(5, 4)`, but `inspect_pair(2, 5, 4)` computes the next offset from `token_len[5]` and sees ID 4 at position 4, returning a match. This is an intentional false positive outside the occurrence-index contract. The directed regression test records this limit so callers do not treat `inspect_pair` as a universal liveness query.

## Validation

`test_lean_endpoints.py` checks randomized training traces against full-recount `naive`, including weighted duplicate pieces; self-overlap and repeated `aaa` runs; historical-start queries through varied-length randomized merges; long-token merges and boundary traversal; the deliberate out-of-contract false positive; and a fresh ID crossing 65535 to 65536. These are small correctness checks, not performance benchmarks.
