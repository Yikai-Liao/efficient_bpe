"""Archive the exact independent-crate sources used for this measurement."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CRATES = ("owned_fused", "owned_scatter", "owned_direct")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = []
    for crate in CRATES:
        base = ROOT / "rust/experiments/radical" / crate
        files.extend(sorted(base.rglob("*.rs")))
        files.extend(base / name for name in ("Cargo.toml", "Cargo.lock", "DESIGN.md"))
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("incomplete source set")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    with (OUT / "new-source-hashes.json").open("x") as output:
        json.dump(hashes, output, indent=2, sort_keys=True)
        output.write("\n")
    with tarfile.open(OUT / "new-source-snapshot.tar.gz", "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    print(json.dumps({"files": len(files), "source_snapshot_sha256":
                      sha(OUT / "new-source-snapshot.tar.gz")}))


if __name__ == "__main__":
    main()
