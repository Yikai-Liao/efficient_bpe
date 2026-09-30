"""Create the full-scale matrix from verified local snapshots; sources stay read-only."""

from array import array
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
SOURCE = Path("/tmp/tokenizers-bpe-bench/data/text")
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
sys.path.insert(0, str(RUST / "tools"))
from ablation_large_fixtures import write_prepared_immutable  # noqa: E402
from ablation_differential import prepare  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural(language, size, existing):
    name = f"{language}-{size}m-continuous"
    source_manifest = SOURCE / f"{language}-manifest.json"
    metadata = json.loads(source_manifest.read_text())
    source = SOURCE / metadata["sizes"][str(size)]["path"]
    expected = metadata["sizes"][str(size)]
    assert metadata["language"] == language and metadata["revision"] == REVISION
    assert metadata["dataset"] == "wikimedia/wikipedia"
    assert source.stat().st_size == expected["bytes"] and sha(source) == expected["sha256"]
    if name in existing:
        row = existing[name].copy()
        assert row["input_sha256"] == expected["sha256"]
        assert sha(RUST / row["file"]) == row["fixture_sha256"]
    else:
        snapshot = ROOT / f"benchmarks/bpe_core_comparison/data/{language}-{size}m.txt"
        if not snapshot.exists():
            with source.open("rb") as src, snapshot.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        assert sha(snapshot) == expected["sha256"]
        text = snapshot.read_bytes().decode("utf-8")
        prepared = prepare([text])
        fixture = RUST / f"fixtures/full/{name}.json"
        fixture_hash, fixture_bytes = write_prepared_immutable(fixture, prepared)
        row = {"case_id": name, "file": str(fixture.relative_to(RUST)),
               "source": str(snapshot.relative_to(ROOT)), "input_bytes": expected["bytes"],
               "input_sha256": expected["sha256"], "fixture_sha256": fixture_hash,
               "fixture_bytes": fixture_bytes, "corpus_positions": len(prepared[0]),
               "initial_alphabet": len(prepared[1]) - 1}
    row.update(language=language, size_mib=size, category="natural", rules=32000,
               min_frequency=2, stored_piece_count=1, piece_weight=1,
               pretokenization=None, source_revision=REVISION,
               source_manifest=str(source_manifest), source_manifest_sha256=sha(source_manifest))
    return row


def synthetic(name, pattern):
    size = 4 << 20
    values = array("I", [0])
    values.extend(array("I", pattern) * (size // len(pattern)))
    values.append(0)
    prepared = values, [1] * (max(pattern) + 1), [1], [1]
    path = RUST / f"fixtures/full/{name}.json"
    fixture_hash, fixture_bytes = write_prepared_immutable(path, prepared)
    return {"case_id": name, "category": "synthetic", "pattern": pattern,
            "size_mib": 4, "input_bytes": size, "input_unit": "ASCII source bytes",
            "file": str(path.relative_to(RUST)), "fixture_sha256": fixture_hash,
            "fixture_bytes": fixture_bytes, "corpus_positions": len(values),
            "initial_alphabet": max(pattern), "rules": 32000, "min_frequency": 2,
            "stored_piece_count": 1, "piece_weight": 1, "pretokenization": None}


def main():
    existing = {}
    for name in ("fixtures.json", "large-fixtures.json"):
        existing.update({row["case_id"]: row for row in
                         json.loads((RUST / "ablation_results" / name).read_text())})
    rows = [natural(language, 16, existing) for language in ("en", "zh", "de", "ja")]
    rows += [natural(language, 4, existing) for language in ("en", "zh")]
    rows += [synthetic("unary-4m", [1]), synthetic("ab-4m", [1, 2])]
    assert len(rows) == 8
    assert all(row["initial_alphabet"] + row["rules"] <= 65535 for row in rows)
    with (OUT / "fixtures.json").open("x") as output:
        json.dump(rows, output, indent=2)
        output.write("\n")
    print(json.dumps({"fixtures": [row["case_id"] for row in rows]}))


if __name__ == "__main__":
    main()
