# Sequential spatial certificate on owner postings

This crate copies the frozen owned trainer. Its only algorithmic change is a
candidate-selection option, `--spatial-probe off|budgeted` (default
`budgeted`). `off` uses the original conservative type certificate. Owner
indexing, lazy/eager heap policy, AA parity, dynamic execution, route buckets,
pos-only birth decoding, and final rule construction stay the same.

During a batch, candidates are still examined in exact global order by
checking every owner's current heap head. A non-AA candidate with no type
conflict joins immediately. If its type conflicts with a chosen rule, the
coordinator scans only that candidate's unique historical posting. It first
validates each position against the stable corpus. At a valid `(A,B)` match,
the only possible shared-token matches are `(X,A)` on its left and `(B,C)` on
its right. Membership in the chosen-key set detects either overlap; zero
sentinels exclude piece boundaries. A conflict stops the batch before the
candidate. If the complete posting is scanned without conflict, the
candidate joins. An AA candidate always stays a singleton, so overlapping
AA occurrences never use this probe.

Let `H` be the sum of historical posting lengths already selected in the
current batch, and `S` the number of positions probed in it. Before reading
another position, the coordinator checks `S < 2H`, using saturating
arithmetic. If the budget is exhausted before completing a candidate, that
candidate remains in its owner's heap and the batch stops. Every successful
addition increases `H` by its posting length. Thus each batch's additional
visits are at most twice its selected historical positions, and summing over
batches preserves this bound because a selected key is removed once. No
corpus-sized marks, global length metadata round, or extra worker dispatch
is required. The probe is sequential and can still lengthen the critical
path despite reducing synchronization.

`probe_calls`, `probe_visited`, `probe_stale_visits`, and the three terminal
outcomes (conflict, budget, proved disjoint) expose what selection did.
`probe_epoch_history_total` sums final selected history lengths over epochs;
`probe_epoch_budget_total` sums their twice-history limits.
`probe_seconds` is included within `select_seconds`, so the phase timers must
not be added as disjoint costs. The fixed-budget quick comparison should run
both modes from the same binary at identical worker count, chunk size, heap
policy, fixture, rule cap, and frequency threshold, followed by the complete
rule/final-token oracle.
