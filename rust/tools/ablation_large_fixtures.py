"""Snapshot verified 16 MiB source texts and prepare one-piece Rust fixtures.

This generator is intentionally separate from the 20-case small fixture
manifest. It validates immutable source revision/content hashes and refuses to
replace any existing snapshot, fixture, or manifest with different bytes.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path("/tmp/tokenizers-bpe-bench/data/text")
DATA_DIR = ROOT / "benchmarks/bpe_core_comparison/data"
FIXTURE_DIR = ROOT / "rust/fixtures/ablation"
SOURCE_MANIFEST_OUT = ROOT / "rust/ablation_results/large-sources.json"
FIXTURE_MANIFEST_OUT = ROOT / "rust/ablation_results/large-fixtures.json"
EXPECTED_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
EXPECTED = {
    "en": {
        "bytes": 16_776_969,
        "sha256": "41b20e33253a4db8c66b95536cbadc86c79072e81eb9e5cd59ef136490abdec9",
    },
    "zh": {
        "bytes": 16_777_136,
        "sha256": "cdee18c95c05586054cfd8e364d3391bb4ca0c2e46cc89d6e4e159d2c0794dc9",
    },
}
sys.path.insert(0, str(ROOT / "benchmarks/bpe_core_comparison/python_rewrite"))
from common_fused import prepare  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json_immutable(path, obj):
    body = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"refusing to replace changed metadata: {path}")
        return sha256_bytes(body)
    with path.open("xb") as stream:
        stream.write(body)
    return sha256_bytes(body)


def copy_immutable(source, destination, expected_hash, expected_bytes):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_hash:
            raise RuntimeError(f"existing snapshot differs from verified source: {destination}")
        return
    # Use an exclusive create; a concurrent or earlier file can never be
    # silently replaced. The caller verified source size/hash immediately
    # before this copy.
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_hash:
        raise RuntimeError(f"copied snapshot failed verification: {destination}")


def write_corpus_array(stream, values, chunk_size=250_000):
    stream.write(b"[")
    for start in range(0, len(values), chunk_size):
        if start:
            stream.write(b",")
        chunk = values[start:start + chunk_size]
        stream.write(",".join(map(str, chunk)).encode("ascii"))
    stream.write(b"]")


def write_prepared_immutable(path, prepared):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Existing large fixtures are never replaced; verification is by hash
        # over the newly encoded deterministic output using a temporary file.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="large-bpe-fixture-", delete=False) as tmp:
                temp_path = Path(tmp.name)
                write_prepared_stream(tmp, prepared)
            if sha256_file(path) != sha256_file(temp_path):
                raise RuntimeError(f"refusing to replace changed fixture: {path}")
            return sha256_file(path), path.stat().st_size
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    with path.open("xb") as stream:
        write_prepared_stream(stream, prepared)
    return sha256_file(path), path.stat().st_size


def write_prepared_stream(stream, prepared):
    corpus, initial_lengths, pivots, weights = prepared
    stream.write(b'{"corpus":')
    write_corpus_array(stream, corpus)
    stream.write(b',"initial_lengths":')
    stream.write(json.dumps(initial_lengths, separators=(",", ":")).encode("ascii"))
    stream.write(b',"pivots":')
    stream.write(json.dumps(pivots, separators=(",", ":")).encode("ascii"))
    stream.write(b',"weights":')
    stream.write(json.dumps(weights, separators=(",", ":")).encode("ascii"))
    stream.write(b"}\n")


def source_record(language):
    manifest_path = SOURCE_DIR / f"{language}-manifest.json"
    text_path = SOURCE_DIR / f"{language}-16m.txt"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = EXPECTED[language]
    size_record = manifest.get("sizes", {}).get("16", {})
    if manifest.get("revision") != EXPECTED_REVISION:
        raise RuntimeError(f"unexpected dataset revision in {manifest_path}")
    if manifest.get("language") != language or manifest.get("dataset") != "wikimedia/wikipedia":
        raise RuntimeError(f"unexpected dataset/language metadata in {manifest_path}")
    if size_record.get("path") != text_path.name:
        raise RuntimeError(f"unexpected 16 MiB path in {manifest_path}")
    if size_record.get("bytes") != expected["bytes"] or size_record.get("sha256") != expected["sha256"]:
        raise RuntimeError(f"manifest size/hash mismatch for {language}-16m")
    if text_path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"source size mismatch: {text_path}")
    actual_hash = sha256_file(text_path)
    if actual_hash != expected["sha256"]:
        raise RuntimeError(f"source SHA-256 mismatch: {text_path}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest": manifest,
        "text_path": str(text_path),
        "text_sha256": actual_hash,
        "text_bytes": expected["bytes"],
    }


def build_fixture(language, source):
    text_path = Path(source["text_path"])
    raw = text_path.read_bytes()
    if sha256_bytes(raw) != source["text_sha256"]:
        raise RuntimeError(f"source changed after validation: {text_path}")
    text = raw.decode("utf-8")
    # One complete source document is one piece of weight 1. Its whitespace,
    # punctuation, and line breaks remain ordinary input symbols.
    prepared = prepare([text])
    case_id = f"{language}-16m-continuous"
    fixture_path = FIXTURE_DIR / f"{case_id}.json"
    fixture_hash, fixture_bytes = write_prepared_immutable(fixture_path, prepared)
    snapshot_path = DATA_DIR / f"{language}-16m.txt"
    copy_immutable(text_path, snapshot_path, source["text_sha256"], source["text_bytes"])
    return {
        "case_id": case_id,
        "dataset": f"{language}-16m",
        "split": "continuous-single-piece",
        "source": str(snapshot_path.relative_to(ROOT)),
        "input_sha256": source["text_sha256"],
        "input_bytes": source["text_bytes"],
        "file": str(fixture_path.relative_to(ROOT / "rust")),
        "fixture_sha256": fixture_hash,
        "fixture_bytes": fixture_bytes,
        "corpus_positions": len(prepared[0]),
        "initial_alphabet": len(prepared[1]) - 1,
        "weight_groups": len(prepared[2]),
        "stored_piece_count": 1,
        "piece_weight": 1,
        "rules": 3000,
        "min_frequency": 2,
        "whitespace_and_punctuation_are_symbols": True,
        "pretokenization": None,
        "source_revision": EXPECTED_REVISION,
        "source_manifest_sha256": source["manifest_sha256"],
    }


def main():
    sources = {language: source_record(language) for language in ("en", "zh")}
    source_copy = {
        "dataset": "wikimedia/wikipedia",
        "revision": EXPECTED_REVISION,
        "source_directory": str(SOURCE_DIR),
        "sources": sources,
    }
    write_json_immutable(SOURCE_MANIFEST_OUT, source_copy)
    rows = [build_fixture(language, sources[language]) for language in ("en", "zh")]
    write_json_immutable(FIXTURE_MANIFEST_OUT, rows)
    print(json.dumps({"fixtures": [row["case_id"] for row in rows],
                      "source_manifest": str(SOURCE_MANIFEST_OUT),
                      "fixture_manifest": str(FIXTURE_MANIFEST_OUT),
                      "snapshots": [row["source"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
