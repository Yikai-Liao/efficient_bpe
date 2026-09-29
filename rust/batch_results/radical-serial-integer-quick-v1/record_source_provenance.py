"""Verify all measured source bytes against the committed frozen baseline."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = "7d6eb8105683e5817cf981cf6b19822777aa4df0"


def command(*args):
    return subprocess.check_output(args, cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    names = command(
        "git", "ls-tree", "-r", "--name-only", BASE, "--",
        "rust/src", "rust/experiments/aa_parity.rs",
        "rust/experiments/radical/serial_integer_hash",
        "rust/experiments/radical/owned_integer_hash",
        "rust/Cargo.toml", "rust/Cargo.lock",
    ).decode().splitlines()
    assert len(names) > 30
    hashes = {}
    for name in names:
        current = (ROOT / name).read_bytes()
        committed = command("git", "show", f"{BASE}:{name}")
        assert current == committed, (name, "source differs from base")
        hashes[name] = sha(current)
    old_archives = {
        "serial_source_snapshot": "rust/batch_results/radical-serial-integer-gate-v1/new-source-snapshot.tar.gz",
        "owner_source_snapshot": "rust/batch_results/radical-local-hash-v1/new-source-snapshot.tar.gz",
        "ahash_build_context": "rust/batch_results/radical-local-hash-v1/ahash-build-context.json",
        "ahash_cargo_fingerprint": "rust/batch_results/radical-local-hash-v1/ahash-cargo-fingerprint.json",
        "rustc_target_cfg": "rust/batch_results/radical-local-hash-v1/rustc-target-cfg.txt",
    }
    archive_hashes = {name: sha((ROOT / path).read_bytes())
                      for name, path in old_archives.items()}
    report = {
        "git_base_commit": BASE,
        "git_head_before_measurement": command("git", "rev-parse", "HEAD").decode().strip(),
        "all_files_match_base_commit_byte_for_byte": True,
        "files_sha256": hashes,
        "prior_archive_paths": old_archives,
        "prior_archive_sha256": archive_hashes,
        "RUSTFLAGS_set": "RUSTFLAGS" in os.environ,
        "CARGO_ENCODED_RUSTFLAGS_set": "CARGO_ENCODED_RUSTFLAGS" in os.environ,
        "rustc_version_verbose": command("/root/.cargo/bin/rustc", "-Vv").decode().strip(),
        "cargo_version": command("/root/.cargo/bin/cargo", "-V").decode().strip(),
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
        "reconstruction": "Check out git_base_commit and use the frozen binary hashes in the screen environment sidecar.",
    }
    with (OUT / "source-provenance.json").open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"tracked_source_files": len(names), "matches_commit": True}))


if __name__ == "__main__":
    main()
