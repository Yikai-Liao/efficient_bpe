"""Build immutable numeric fixtures for the native Rust ablation matrix.

The eleven published Rust baseline fixtures are reused byte-for-byte. Extra
cases live under the ignored rust/fixtures/ablation directory; rerunning this
tool verifies an existing file instead of overwriting it.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"
PYTHON_REWRITE = ROOT / "benchmarks/bpe_core_comparison/python_rewrite"
sys.path.insert(0, str(PYTHON_REWRITE))
from common_fused import prepare  # noqa: E402
from bench_fused import load_pieces  # noqa: E402

OLD_MANIFEST = RUST / "results/fixtures.json"
OUTPUT_DIR = RUST / "fixtures/ablation"
MANIFEST = RUST / "ablation_results/fixtures.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def wire_bytes(prepared):
    wire = dict(zip(("corpus", "initial_lengths", "pivots", "weights"),
                    (list(part) for part in prepared)))
    return (json.dumps(wire, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def write_immutable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"refusing to replace changed fixture: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(body)


def record(case_id, split, source, input_hash, input_bytes, prepared, body,
           *, rules=3000, min_frequency=2, weight_scale=1, extra=None):
    row = {
        "case_id": case_id,
        "dataset": case_id,
        "split": split,
        "source": source,
        "input_sha256": input_hash,
        "input_bytes": input_bytes,
        "file": str((OUTPUT_DIR / f"{case_id}.json").relative_to(RUST)),
        "fixture_sha256": sha256(body),
        "corpus_positions": len(prepared[0]),
        "initial_alphabet": len(prepared[1]) - 1,
        "weight_groups": len(prepared[2]),
        "rules": rules,
        "min_frequency": min_frequency,
        "weight_scale": weight_scale,
    }
    if extra:
        row.update(extra)
    return row


def add_prepared(rows, case_id, split, source, raw_bytes, prepared,
                 *, rules=3000, min_frequency=2, weight_scale=1, extra=None):
    body = wire_bytes(prepared)
    write_immutable(OUTPUT_DIR / f"{case_id}.json", body)
    rows.append(record(case_id, split, source, sha256(raw_bytes), len(raw_bytes),
                       prepared, body, rules=rules, min_frequency=min_frequency,
                       weight_scale=weight_scale, extra=extra))


def main():
    old_rows = json.loads(OLD_MANIFEST.read_text())
    rows = []
    for old in old_rows:
        path = RUST / old["file"]
        body = path.read_bytes()
        if sha256(body) != old["fixture_sha256"]:
            raise RuntimeError(f"legacy fixture hash mismatch: {path}")
        rows.append({
            "case_id": f"{old['dataset']}--{old['split']}",
            **old,
            "source": "legacy Rust/Python prepared snapshot; reused unchanged",
            "min_frequency": 2,
            "weight_scale": 1,
        })

    # A fixed 64x weight case exercises u64 counts and scaled thresholds without
    # changing the physical occurrence stream. Keep the source fixture intact.
    data_dir = ROOT / "benchmarks/bpe_core_comparison/data"
    source_row = next(row for row in old_rows
                      if row["dataset"] == "en-1m" and row["split"] == "regex")
    source_path = RUST / source_row["file"]
    source_raw = (data_dir / "en-1m.txt").read_bytes()
    if sha256(source_raw) != source_row["input_sha256"]:
        raise RuntimeError("en-1m source hash does not match the frozen fixture manifest")
    source_data = json.loads(source_path.read_bytes())
    source_data["weights"] = [weight * 64 for weight in source_data["weights"]]
    prepared = (source_data["corpus"], source_data["initial_lengths"],
                source_data["pivots"], source_data["weights"])
    add_prepared(rows, "weighted64-en-1m-regex", "regex-weighted64",
                 f"prepared from {source_row['file']}; each piece weight x64",
                 source_raw, prepared,
                 rules=3000, min_frequency=128, weight_scale=64)

    # Complete lexicographic greedy chains at two missing scales.
    for length in (2000, 4000):
        word = "".join(chr(0x1000 + i) for i in range(length, 0, -1))
        raw = (word + "\n" + word).encode("utf-8")
        prepared = prepare([word, word])
        add_prepared(rows, f"chain-{length}", "descending-id-chain",
                     "two identical descending-codepoint pieces", raw, prepared,
                     rules=length, min_frequency=2)

    # Unsplit full source texts remain one piece; whitespace and punctuation are
    # ordinary symbols. Repeating the exact piece gives it weight two, not a
    # sequence of tokenizer-produced boundaries.
    for dataset in ("en-4m", "zh-4m"):
        raw = (data_dir / f"{dataset}.txt").read_bytes()
        old_source = next(row for row in old_rows if row["dataset"] == dataset
                          and row["split"] == "regex")
        if sha256(raw) != old_source["input_sha256"]:
            raise RuntimeError(f"{dataset} source hash does not match legacy fixture")
        text = raw.decode("utf-8")
        prepared = prepare([text, text])
        add_prepared(rows, f"{dataset}-continuous", "continuous-single-piece",
                     f"{dataset}.txt decoded as one piece; no regex/paragraph split",
                     raw, prepared, rules=3000, min_frequency=2,
                     extra={"stored_piece_count": 1,
                            "whitespace_and_punctuation_are_symbols": True})

    # A single run and alternating run are stress cases for long token spans
    # without pre-tokenization. Each identical source piece is weighted twice.
    for case_id, text in (("single-run-a-65536", "a" * 65536),
                          ("single-piece-ab-65536", "ab" * 32768)):
        raw = text.encode("utf-8")
        prepared = prepare([text, text])
        add_prepared(rows, case_id, "continuous-single-piece", "generated single piece x2",
                     raw, prepared, rules=128, min_frequency=2,
                     extra={"stored_piece_count": 1})

    # Mix long shared runs, short tokens, and one-off boundaries so low-frequency
    # filtering and occurrence storage see both dense and sparse pair sets.
    mixed = ["a" * 4096 + "b", "a" * 64 + "c", "ab", "ac",
             "b" * 511 + "d", "d" * 3, "rare-z", "rare-y"]
    mixed_raw = "\n".join(mixed).encode("utf-8")
    add_prepared(rows, "mixed-long-short-rare", "regex-piece-stress",
                 "generated weighted pieces with long runs and rare edges",
                 mixed_raw, prepare(mixed), rules=512, min_frequency=2,
                 extra={"stored_piece_count": len(Counter(mixed))})

    # Many low-frequency distinct bigrams alongside a high-frequency repeated
    # pair provide pressure on filtered/arena indices and priority queues.
    sparse_pairs = []
    for i in range(512):
        left = chr(0x1000 + 2 * i)
        right = chr(0x1000 + 2 * i + 1)
        sparse_pairs.append(left + right)
    sparse_pairs.extend(["abababab"] * 8)
    sparse_raw = "\n".join(sparse_pairs).encode("utf-8")
    add_prepared(rows, "rare-pair-pressure-512", "sparse-bigram-stress",
                 "512 unique bigrams plus repeated ab pieces", sparse_raw,
                 prepare(sparse_pairs), rules=64, min_frequency=2,
                 extra={"stored_piece_count": len(Counter(sparse_pairs)),
                        "unique_singleton_bigrams": 512})

    manifest_body = (json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        if MANIFEST.read_bytes() != manifest_body:
            raise RuntimeError(f"refusing to replace changed manifest: {MANIFEST}")
    else:
        with MANIFEST.open("xb") as stream:
            stream.write(manifest_body)
    print(json.dumps({"fixtures": len(rows), "new_fixtures": len(rows) - len(old_rows),
                      "manifest": str(MANIFEST), "output_dir": str(OUTPUT_DIR)}))


if __name__ == "__main__":
    main()
