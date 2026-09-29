"""Record shared and inherited context sources from the committed baseline."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = "d757e69bf8aa48d267cfaeb4f658a0051908e28d"


def command(*args):
    return subprocess.check_output(args, cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    tracked = command("git", "ls-tree", "-r", "--name-only", BASE,
                      "--", "rust/src", "rust/experiments/radical/owned_context").decode().splitlines()
    names = sorted(set(tracked + ["rust/Cargo.toml", "rust/Cargo.lock",
                                   "rust/experiments/aa_parity.rs"]))
    hashes = {}
    for name in names:
        current = (ROOT / name).read_bytes()
        committed = command("git", "show", f"{BASE}:{name}")
        if current != committed:
            raise AssertionError((name, "differs from base commit"))
        hashes[name] = sha(current)
    report = {
        "git_base_commit": BASE,
        "git_head_during_measurement": command("git", "rev-parse", "HEAD").decode().strip(),
        "scope": "rust/src/**, rust/Cargo.toml and Cargo.lock, rust/experiments/aa_parity.rs, inherited owned_context crate",
        "all_files_match_base_commit_byte_for_byte": True,
        "files_sha256": hashes,
        "rustc_version_verbose": command("/root/.cargo/bin/rustc", "-Vv").decode().strip(),
        "cargo_version": command("/root/.cargo/bin/cargo", "-V").decode().strip(),
        "reconstruction": "Check out git_base_commit, then overlay new-source-snapshot.tar.gz from this archive.",
    }
    with (OUT / "shared-source-provenance.json").open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"shared_files": len(hashes), "match_base_commit": True}))


if __name__ == "__main__":
    main()
