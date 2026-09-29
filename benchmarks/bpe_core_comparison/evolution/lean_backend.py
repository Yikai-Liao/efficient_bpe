"""Two/three-write endpoints under the fresh-ID historical-occurrence contract.

Token IDs are minted once and never reused. Inspect is called only for an
occurrence that once really had the queried left ID at this position. This
is intentionally not a general live-position predicate.
"""
from backends_fused import FusedEbpeEndpoints


class LeanEndpoints(FusedEbpeEndpoints):
    def merge_known(self, pos, right, after, new_id, new_len):
        corpus = self.corpus
        corpus[pos] = new_id
        if after - right == 1:
            corpus[right] = new_id
        else:
            corpus[right] = 0
            corpus[after - 1] = new_id
        return right
