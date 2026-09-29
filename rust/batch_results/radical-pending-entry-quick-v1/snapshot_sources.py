"""Archive the frozen pending owner implementation and its shared-source base."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BASE = "af7f05d5cc57bf44cad7af84c9cef931d4f1ed37"
CRATE = RUST / "experiments/radical/owned_pending_entry"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT)


def main():
    files = sorted((CRATE / "src").glob("*.rs"))
    files.extend(CRATE / name for name in ("Cargo.toml", "Cargo.lock", "DESIGN.md"))
    assert all(path.is_file() for path in files)
    hashes = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in files}
    with (OUT / "new-source-hashes.json").open("x") as output:
        json.dump(hashes, output, indent=2, sort_keys=True)
        output.write("\n")
    with tarfile.open(OUT / "new-source-snapshot.tar.gz", "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    tracked = command("git", "ls-tree", "-r", "--name-only", BASE,
                      "--", "rust/src").decode().splitlines()
    shared = sorted(set(tracked + ["rust/Cargo.toml", "rust/Cargo.lock",
                                   "rust/experiments/aa_parity.rs"]))
    shared_hashes = {}
    for name in shared:
        current = (ROOT / name).read_bytes()
        assert current == command("git", "show", f"{BASE}:{name}"), name
        shared_hashes[name] = sha(current)
    report = {"git_base_commit": BASE,
              "git_head_during_measurement": command("git", "rev-parse", "HEAD").decode().strip(),
              "all_shared_files_match_base_commit_byte_for_byte": True,
              "shared_files_sha256": shared_hashes,
              "rustc_version_verbose": command("/root/.cargo/bin/rustc", "-Vv").decode().strip(),
              "cargo_version": command("/root/.cargo/bin/cargo", "-V").decode().strip(),
              "reconstruction": "Check out git_base_commit, then overlay new-source-snapshot.tar.gz."}
    with (OUT / "shared-source-provenance.json").open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"new_files": len(files), "shared_files": len(shared),
                      "source_snapshot_sha256": sha((OUT / "new-source-snapshot.tar.gz").read_bytes())}))


if __name__ == "__main__":
    main()
