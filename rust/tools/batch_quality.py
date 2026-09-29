#!/usr/bin/env python3
"""Compare BPE merge traces by token count on matched train and heldout text.

The tool targets continuous-single-piece cases. It reproduces the dense
Unicode-scalar alphabet used by common_fused.prepare, then applies the trace's
selected pairs in training order with left-to-right, non-overlapping matches.
Unknown heldout scalars get fresh IDs outside every merge rule and therefore
remain atomic. Encoding uses a linked-array representation and per-rule
position min-heaps to avoid rescanning the full text once per merge round.
"""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"
DATA = ROOT / "benchmarks/bpe_core_comparison/data"
DEFAULT_TRAIN_MANIFEST = RUST / "ablation_results/fixtures.json"
DEFAULT_HELDOUT_MANIFEST = RUST / "ablation_results/large-fixtures.json"
DEFAULT_SOURCE_MANIFEST = RUST / "ablation_results/large-sources.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def find_case(manifest_path, case_id):
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {case_id!r} in {manifest_path}")
    return matches[0]


def continuous_case_path(row, data_dir=DATA):
    if row.get("split") != "continuous-single-piece" or row.get("stored_piece_count", 1) != 1:
        raise ValueError(f"{row.get('case_id')} is not a single continuous piece")
    match = re.fullmatch(r"(en|zh)-(\d+m)-continuous", row.get("case_id", ""))
    if not match:
        raise ValueError(f"unsupported continuous case id: {row.get('case_id')!r}")
    lang, size = match.groups()
    path = data_dir / f"{lang}-{size}.txt"
    raw = path.read_bytes()
    if len(raw) != row.get("input_bytes") or sha256(raw) != row.get("input_sha256"):
        raise ValueError(f"raw-text size/hash disagrees with manifest row: {path}")
    return path, raw


def read_single_piece_weight(row, rust_root=RUST):
    """Read the weight array without parsing the fixture's large corpus array."""
    fixture_path = (rust_root / row["file"]).resolve()
    try:
        fixture_path.relative_to(rust_root.resolve())
    except ValueError as error:
        raise ValueError("prepared fixture path escapes the Rust project") from error
    body = fixture_path.read_bytes()
    if sha256(body) != row.get("fixture_sha256"):
        raise ValueError(f"prepared fixture hash disagrees with manifest: {fixture_path}")
    marker = b',"weights":'
    start = body.find(marker)
    if start < 0:
        raise ValueError(f"prepared fixture has no weights field: {fixture_path}")
    start += len(marker)
    end = body.find(b"]", start)
    if end < 0:
        raise ValueError(f"prepared fixture weights field is truncated: {fixture_path}")
    weights = json.loads(body[start:end + 1])
    if len(weights) != 1 or not isinstance(weights[0], int) or weights[0] <= 0:
        raise ValueError("quality scoring requires exactly one positive-weight piece")
    if len(weights) != row.get("weight_groups"):
        raise ValueError("weight group count differs from the fixture manifest")
    return weights[0]


def verify_upstream_revision(source_manifest_path, train_row, heldout_row):
    source_data = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_data.get("dataset") != "wikimedia/wikipedia":
        raise ValueError("expected the frozen Wikimedia Wikipedia source manifest")
    revision = source_data.get("revision")
    if not revision:
        raise ValueError("source manifest has no dataset revision")
    for row in (train_row, heldout_row):
        if row.get("source_revision", revision) != revision:
            raise ValueError(f"source revision mismatch for {row.get('case_id')}")
        match = re.fullmatch(r"(en|zh)-(\d+m)-continuous", row.get("case_id", ""))
        if not match:
            raise ValueError(f"unsupported continuous case id: {row.get('case_id')!r}")
        lang, size = match.groups()
        sizes = source_data.get("sources", {}).get(lang, {}).get("manifest", {}).get("sizes", {})
        source_size = sizes.get(size.removesuffix("m"))
        if not source_size:
            raise ValueError(f"source manifest lacks {row['case_id']}")
        if (source_size.get("bytes") != row.get("input_bytes") or
                source_size.get("sha256") != row.get("input_sha256")):
            raise ValueError(f"source manifest hash/size mismatch for {row['case_id']}")
    return revision


class _JsonCursor:
    """Small buffered cursor for the native trace's two numeric-array fields."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = b""
        self.pos = 0

    def _fill(self):
        if self.pos >= len(self.buffer):
            self.buffer = self.stream.read(1024 * 1024)
            self.pos = 0
            if not self.buffer:
                raise ValueError("unexpected end of native trace")

    def skip_ws(self):
        while True:
            self._fill()
            while self.pos < len(self.buffer) and self.buffer[self.pos] in b" \t\r\n":
                self.pos += 1
            if self.pos < len(self.buffer):
                return

    def peek(self):
        self.skip_ws()
        return self.buffer[self.pos]

    def expect(self, expected):
        self.skip_ws()
        if self.buffer[self.pos] != expected:
            raise ValueError(f"expected {chr(expected)!r} in native trace")
        self.pos += 1

    def read_key(self):
        self.expect(ord('"'))
        key_bytes = bytearray()
        while True:
            self._fill()
            end = self.buffer.find(b'"', self.pos)
            if end >= 0:
                key_bytes.extend(self.buffer[self.pos:end])
                self.pos = end + 1
                return key_bytes.decode("ascii")
            # Native keys are short ASCII names. Keep just the unfinished key
            # across a buffer boundary rather than copying the remaining trace.
            key_bytes.extend(self.buffer[self.pos:])
            self.pos = len(self.buffer)

    def read_nested_array(self):
        self.expect(ord('['))
        payload = bytearray(b"[")
        depth = 1
        while depth:
            self._fill()
            next_open = self.buffer.find(b"[", self.pos)
            next_close = self.buffer.find(b"]", self.pos)
            if next_open < 0 and next_close < 0:
                payload.extend(self.buffer[self.pos:])
                self.pos = len(self.buffer)
                continue
            if next_close < 0 or (next_open >= 0 and next_open < next_close):
                delimiter = next_open
                depth += 1
            else:
                delimiter = next_close
                depth -= 1
            payload.extend(self.buffer[self.pos:delimiter + 1])
            self.pos = delimiter + 1
        return json.loads(payload)

    def count_flat_array(self):
        """Count scalar items in a flat numeric array without retaining them."""
        self.expect(ord('['))
        commas = 0
        has_values = False
        while True:
            self._fill()
            closing = self.buffer.find(b"]", self.pos)
            end = len(self.buffer) if closing < 0 else closing
            part = self.buffer[self.pos:end]
            commas += part.count(b",")
            has_values = has_values or any(48 <= byte <= 57 for byte in part)
            self.pos = end if closing < 0 else closing + 1
            if closing >= 0:
                return commas + 1 if has_values else 0


def read_trace(path):
    """Read merge rules and count `final` IDs without materializing that array.

    serde_json may serialize the object's keys alphabetically (`final` before
    `merges`) when preserve_order is disabled. Parse both fields by key and
    stream-count the flat final IDs, so trace size does not create a second
    full-corpus Python list.
    """
    merges = None
    final_count = None
    with path.open("rb") as stream:
        cursor = _JsonCursor(stream)
        cursor.expect(ord('{'))
        while True:
            if cursor.peek() == ord('}'):
                cursor.expect(ord('}'))
                break
            key = cursor.read_key()
            cursor.expect(ord(':'))
            if key == "merges":
                if merges is not None:
                    raise ValueError("duplicate merges field in trace")
                merges = cursor.read_nested_array()
            elif key == "final":
                if final_count is not None:
                    raise ValueError("duplicate final field in trace")
                final_count = cursor.count_flat_array()
            else:
                raise ValueError(f"unexpected native trace field: {key!r}")
            separator = cursor.peek()
            if separator == ord(','):
                cursor.expect(ord(','))
            elif separator == ord('}'):
                continue
            else:
                raise ValueError("expected comma or object close in native trace")
    if merges is None or final_count is None:
        raise ValueError(f"trace must contain merges and final arrays: {path}")
    rows = merges
    merges = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"invalid merge row {index} in {path}")
        left, right, frequency = row
        if not all(isinstance(x, int) and x >= 0 for x in row):
            raise ValueError(f"non-integer/negative merge field at row {index}: {path}")
        if frequency == 0:
            raise ValueError(f"zero-frequency selected pair at row {index}: {path}")
        merges.append((left, right, frequency))
    return merges, final_count


def _heap_push(heap, item):
    heap.append(item)
    child = len(heap) - 1
    while child:
        parent = (child - 1) >> 1
        if heap[parent] <= item:
            break
        heap[child] = heap[parent]
        child = parent
    heap[child] = item


def _heap_pop(heap):
    result = heap[0]
    tail = heap.pop()
    if heap:
        parent = 0
        limit = len(heap)
        while True:
            child = parent * 2 + 1
            if child >= limit:
                break
            right = child + 1
            if right < limit and heap[right] < heap[child]:
                child = right
            if heap[child] >= tail:
                break
            heap[parent] = heap[child]
            parent = child
        heap[parent] = tail
    return result


def encode_count(text, alphabet_ids, merges, *, capture_ids=False):
    """Return BPE token count and count of heldout scalars absent in training.

    Pair rules are applied in exact order. Each rule consumes its current
    left-to-right non-overlapping occurrences. Node positions remain stable;
    only the two adjacent array links change on a merge.
    """
    if array("I").itemsize != 4 or array("i").itemsize != 4:
        raise RuntimeError("this implementation requires 32-bit array I/i")
    if len(text) >= (1 << 31):
        raise ValueError("text exceeds signed 32-bit node-index capacity")
    initial_count = len(alphabet_ids)
    rule_count = len(merges)
    max_rule_id = initial_count + rule_count
    pair_rank = {}
    for rank, (left, right, _frequency) in enumerate(merges):
        new_id = initial_count + 1 + rank
        if left == 0 or right == 0 or left >= new_id or right >= new_id:
            raise ValueError(f"merge {rank} references a missing/future token")
        pair = (left, right)
        if pair in pair_rank:
            raise ValueError(f"pair selected more than once in trace: {pair}")
        pair_rank[pair] = rank

    # Base IDs are 1..initial_count. Unknown heldout scalars receive unique
    # sentinels above every possible trained ID, so they cannot match a rule.
    unknown_ids = {}
    next_unknown = max_rule_id + 1
    tokens = array("I")
    for scalar in text:
        token = alphabet_ids.get(scalar)
        if token is None:
            token = unknown_ids.get(scalar)
            if token is None:
                token = next_unknown
                next_unknown += 1
                if token > 0xFFFFFFFF:
                    raise ValueError("token ID exceeds u32 range")
                unknown_ids[scalar] = token
        tokens.append(token)
    count = len(tokens)
    if not count:
        result = {"tokens": 0, "unseen_scalars": 0}
        if capture_ids:
            result["token_ids"] = []
        return result

    previous = array("i", range(-1, count - 1))
    following = array("i", range(1, count + 1))
    following[-1] = -1
    alive = bytearray(b"\x01") * count
    candidates = [array("I") for _ in merges]

    # Initial candidate positions occur in ascending order, so appending keeps
    # each compact min-heap valid without a heapify pass.
    for pos in range(count - 1):
        rank = pair_rank.get((tokens[pos], tokens[pos + 1]))
        if rank is not None:
            candidates[rank].append(pos)

    def add_candidate(left_pos, right_pos):
        if left_pos < 0 or right_pos < 0:
            return
        rank = pair_rank.get((tokens[left_pos], tokens[right_pos]))
        if rank is not None:
            _heap_push(candidates[rank], left_pos)

    for rank, (left_id, right_id, _frequency) in enumerate(merges):
        heap = candidates[rank]
        new_id = initial_count + 1 + rank
        while heap:
            pos = _heap_pop(heap)
            if not alive[pos]:
                continue
            right_pos = following[pos]
            if right_pos < 0 or tokens[pos] != left_id or tokens[right_pos] != right_id:
                continue
            before = previous[pos]
            after = following[right_pos]
            tokens[pos] = new_id
            alive[right_pos] = 0
            following[pos] = after
            if before >= 0:
                following[before] = pos
            if after >= 0:
                previous[after] = pos
            add_candidate(before, pos)
            add_candidate(pos, after)
            count -= 1
    result = {"tokens": count, "unseen_scalars": len(unknown_ids)}
    if capture_ids:
        final_ids = []
        pos = 0
        while pos >= 0:
            if alive[pos]:
                final_ids.append(tokens[pos])
            pos = following[pos]
        result["token_ids"] = final_ids
    return result


def simple_encode_ids(text, alphabet_ids, merges):
    """Small clear oracle used only by --self-test."""
    ids = dict(alphabet_ids)
    next_unknown = len(alphabet_ids) + 1 + len(merges)
    result = []
    for scalar in text:
        if scalar not in ids:
            ids[scalar] = next_unknown
            next_unknown += 1
        result.append(ids[scalar])
    initial_count = len(alphabet_ids)
    for rank, (left, right, _frequency) in enumerate(merges):
        new_id = initial_count + 1 + rank
        updated = []
        i = 0
        while i < len(result):
            if i + 1 < len(result) and result[i] == left and result[i + 1] == right:
                updated.append(new_id)
                i += 2
            else:
                updated.append(result[i])
                i += 1
        result = updated
    return result


def self_test():
    alphabet = sorted(set("aabbcc"))
    ids = {scalar: i + 1 for i, scalar in enumerate(alphabet)}
    # Rules exercise overlap resolution, adjacent distinct merges, and a new
    # pair involving the output of an earlier rule.
    rules = [(ids["a"], ids["b"], 5),
             (ids["a"], ids["c"], 3),
             (ids["a"] + len(alphabet), ids["c"], 1)]
    samples = ["", "a", "aaa", "abab", "abcabc", "abac", "a🙂c🙂a"]
    rng = random.Random(20260929)
    samples.extend("".join(rng.choices("abc🙂", k=rng.randrange(40))) for _ in range(250))
    for text in samples:
        fast = encode_count(text, ids, rules, capture_ids=True)
        slow_ids = simple_encode_ids(text, ids, rules)
        if fast["tokens"] != len(slow_ids) or fast["token_ids"] != slow_ids:
            raise AssertionError((text, fast, slow_ids))
        unseen = len(set(text) - set(ids))
        if fast["unseen_scalars"] != unseen:
            raise AssertionError((text, fast["unseen_scalars"], unseen))
    random_checks = 0
    for seed in range(40):
        random_ids = {scalar: index + 1 for index, scalar in enumerate("abcd")}
        random_rules = []
        used_pairs = set()
        local = random.Random(seed)
        for _ in range(24):
            fresh = len(random_ids) + 1 + len(random_rules)
            pair = (local.randrange(1, fresh), local.randrange(1, fresh))
            while pair in used_pairs:
                pair = (local.randrange(1, fresh), local.randrange(1, fresh))
            used_pairs.add(pair)
            random_rules.append((*pair, local.randrange(1, 100)))
        for _ in range(20):
            text = "".join(local.choices("abcd🙂", k=local.randrange(60)))
            fast = encode_count(text, random_ids, random_rules, capture_ids=True)
            slow_ids = simple_encode_ids(text, random_ids, random_rules)
            if fast["token_ids"] != slow_ids:
                raise AssertionError((seed, text, fast, slow_ids))
            if fast["unseen_scalars"] != len(set(text) - set(random_ids)):
                raise AssertionError((seed, "unknown count"))
            random_checks += 1
    with tempfile.TemporaryDirectory(prefix="batch-quality-selftest-") as temp:
        for index, payload in enumerate((
                {"merges": rules, "final": [0, 4, 3, 0]},
                {"final": [0, 4, 3, 0], "merges": rules},
        )):
            trace_path = Path(temp) / f"trace-{index}.json"
            trace_path.write_text(json.dumps(payload, separators=(",", ":")),
                                  encoding="utf-8")
            if read_trace(trace_path) != (rules, 4):
                raise AssertionError("native trace parser failed for one key order")
        fixture_root = Path(temp) / "rust"
        fixture_path = fixture_root / "fixtures" / "piece.json"
        fixture_path.parent.mkdir(parents=True)
        fixture_body = b'{"corpus":[0,1,0],"initial_lengths":[1,1],"pivots":[1],"weights":[2]}\n'
        fixture_path.write_bytes(fixture_body)
        row = {"file": "fixtures/piece.json", "fixture_sha256": sha256(fixture_body),
               "weight_groups": 1}
        if read_single_piece_weight(row, fixture_root) != 2:
            raise AssertionError("prepared single-piece weight was not read")
    weighted = describe_count(7, 14, 14, 0, piece_weight=2)
    if weighted["tokens"] != 7 or weighted["weighted_token_mass"] != 14:
        raise AssertionError("piece weight must scale token mass, not re-encode the text")
    print(f"self-test passed: {len(samples)} fixed and {random_checks} randomized strings")


def parse_named_traces(values):
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError("each --trace must be LABEL=PATH")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError("each --trace must have a nonempty label and path")
        if any(row[0] == label for row in result):
            raise ValueError(f"duplicate trace label: {label}")
        result.append((label, Path(path)))
    if len(result) < 2:
        raise ValueError("provide at least two --trace LABEL=PATH values to compare")
    return result


def chars_to_ids(text):
    alphabet = sorted(set(text))
    return {scalar: index + 1 for index, scalar in enumerate(alphabet)}


def describe_count(count, raw_bytes, chars, unseen, *, piece_weight=1):
    weighted_mass = count * piece_weight
    return {
        "tokens": count,
        "piece_weight": piece_weight,
        "weighted_token_mass": weighted_mass,
        "bytes": raw_bytes,
        "unicode_scalars": chars,
        "unseen_scalars": unseen,
        "bytes_per_token": (raw_bytes / count) if count else None,
        "tokens_per_mib": (count * 1024 * 1024 / raw_bytes) if raw_bytes else None,
        "bytes_per_weighted_token": (raw_bytes / weighted_mass) if weighted_mass else None,
        "weighted_tokens_per_mib": (weighted_mass * 1024 * 1024 / raw_bytes)
        if raw_bytes else None,
    }


def run(args):
    train_row = find_case(args.train_manifest, args.train_case)
    heldout_row = find_case(args.heldout_manifest, args.heldout_case)
    train_path, train_raw = continuous_case_path(train_row, args.data_dir)
    heldout_path, heldout_raw = continuous_case_path(heldout_row, args.data_dir)
    revision = verify_upstream_revision(args.source_manifest, train_row, heldout_row)
    train_lang = train_row["case_id"].split("-", 1)[0]
    heldout_lang = heldout_row["case_id"].split("-", 1)[0]
    if train_lang != heldout_lang:
        raise ValueError("training and heldout cases must use the same language")
    if len(heldout_raw) <= len(train_raw):
        raise ValueError("heldout source must be longer than the training prefix")
    prefix = heldout_raw[:len(train_raw)]
    if prefix != train_raw or sha256(prefix) != train_row["input_sha256"]:
        raise ValueError("training text is not the exact byte prefix of heldout source")
    heldout_raw_bytes = heldout_raw[len(train_raw):]
    train_text = train_raw.decode("utf-8")
    heldout_text = heldout_raw_bytes.decode("utf-8")
    alphabet_ids = chars_to_ids(train_text)
    train_piece_weight = read_single_piece_weight(train_row)
    if len(alphabet_ids) != train_row.get("initial_alphabet"):
        raise ValueError("recreated Unicode alphabet size differs from fixture manifest")
    if train_row.get("corpus_positions") != len(train_text) + 2:
        raise ValueError("training fixture is not the declared one-piece Unicode sequence")
    if train_row.get("weight_groups") != 1:
        raise ValueError("quality scoring currently requires one weight group")
    if train_text.endswith("\x00") or "\x00" in train_text:
        raise ValueError("NUL is reserved as the prepared-corpus piece separator")

    traces = [(label, path, *read_trace(path))
              for label, path in parse_named_traces(args.trace)]
    reference = args.reference or traces[0][0]
    if reference not in {label for label, _, _, _ in traces}:
        raise ValueError(f"reference label {reference!r} was not provided")
    round_counts = {len(merges) for _, _, merges, _ in traces}
    if len(round_counts) != 1:
        raise ValueError(f"traces have different BPE merge-round counts: {sorted(round_counts)}")

    scored = []
    for label, trace_path, merges, trace_final_count in traces:
        for rank, (left, right, _frequency) in enumerate(merges):
            new_id = len(alphabet_ids) + 1 + rank
            if max(left, right) > len(alphabet_ids) + rank:
                raise ValueError(f"{label} trace references an ID outside its learned prefix")
            if left == 0 or right == 0 or left == right == new_id:
                raise ValueError(f"{label} trace has an invalid pair at round {rank}")
        train_count = encode_count(train_text, alphabet_ids, merges)
        if train_count["unseen_scalars"] != 0:
            raise AssertionError("training text produced unknown scalars")
        if trace_final_count != train_count["tokens"] + 2:
            raise ValueError(
                f"{label} training replay count {train_count['tokens']} disagrees with "
                f"native final trace length {trace_final_count} (expected tokens + two sentinels)"
            )
        heldout_count = encode_count(heldout_text, alphabet_ids, merges)
        scored.append({
            "label": label,
            "trace": str(trace_path),
            "merge_rounds": len(merges),
            "native_final_token_count": trace_final_count,
            "training": describe_count(train_count["tokens"], len(train_raw),
                                        len(train_text), train_count["unseen_scalars"],
                                        piece_weight=train_piece_weight),
            "heldout": describe_count(heldout_count["tokens"], len(heldout_raw_bytes),
                                       len(heldout_text), heldout_count["unseen_scalars"]),
        })
    ref_row = next(row for row in scored if row["label"] == reference)
    for row in scored:
        row["vs_reference"] = {}
        for part in ("training", "heldout"):
            base = ref_row[part]["tokens"]
            value = row[part]["tokens"]
            row["vs_reference"][part] = {
                "token_delta": value - base,
                "token_delta_percent": (100 * (value - base) / base) if base else None,
            }
            if part == "training":
                base_mass = ref_row[part]["weighted_token_mass"]
                value_mass = row[part]["weighted_token_mass"]
                row["vs_reference"][part]["weighted_token_mass_delta"] = value_mass - base_mass
                row["vs_reference"][part]["weighted_token_mass_delta_percent"] = (
                    100 * (value_mass - base_mass) / base_mass if base_mass else None
                )
    return {
        "method": "ordered BPE selected-pair replay; left-to-right non-overlapping per round",
        "unknown_heldout_scalars": "assigned unique IDs above trained vocabulary; never merge",
        "training_count_note": "encode the stored training piece once; weighted_token_mass = piece tokens * fixture weight",
        "source_revision": revision,
        "training_case": train_row["case_id"],
        "training_source": str(train_path),
        "training_sha256": train_row["input_sha256"],
        "training_bytes": len(train_raw),
        "heldout_case": heldout_row["case_id"],
        "heldout_source": str(heldout_path),
        "heldout_prefix_verified": True,
        "heldout_sha256": sha256(heldout_raw_bytes),
        "heldout_bytes": len(heldout_raw_bytes),
        "initial_alphabet_size": len(alphabet_ids),
        "reference": reference,
        "results": scored,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=False, default=[],
                        help="merge trace as LABEL=PATH; repeat for each model")
    parser.add_argument("--train-case", default="en-4m-continuous")
    parser.add_argument("--heldout-case", default="en-16m-continuous")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--heldout-manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--reference", help="baseline trace label; defaults to first --trace")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(args.trace) < 2:
        parser.error("provide at least two --trace LABEL=PATH options")
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
