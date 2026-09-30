"""Read-only audit of frozen sources, binary hashes and measured repetitions."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = "c1d8fc3bf4fb31bfa59afd545b37a7697c15ec1f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    for name, calls in (("radical-adaptive-replay-v1", 40), ("radical-replay-counts-v1", 26)):
        directory = ROOT / "rust/batch_results" / name
        env = json.loads((directory / "screen.jsonl.environment.json").read_text())
        config = json.loads((directory / "frozen-config.json").read_text())
        assert sha((directory / "frozen-config.json").read_bytes()) == env["config_sha256"]
        snapshot = directory / env.get("source_snapshot_filename", "new-source-snapshot.tar.gz")
        assert sha(snapshot.read_bytes()) == env["source_snapshot_sha256"]
        with tarfile.open(snapshot, "r:gz") as archive:
            for path, digest in env["source_sha256"].items():
                assert sha(archive.extractfile(path).read()) == digest
                if path.endswith((".rs", "Cargo.toml", "Cargo.lock")):
                    assert sha((ROOT / path).read_bytes()) == digest
        final_hashes = json.loads((directory / "new-source-hashes.json").read_text())
        assert all(sha((ROOT / path).read_bytes()) == digest for path, digest in final_hashes.items())
        with tarfile.open(directory / "new-source-snapshot.tar.gz", "r:gz") as archive:
            for path, digest in final_hashes.items():
                assert sha(archive.extractfile(path).read()) == digest
        provenance = json.loads((directory / "shared-source-provenance.json").read_text())
        assert provenance["git_base_commit"] == BASE
        assert len(provenance["shared_files_sha256"]) == 26
        for path, digest in provenance["shared_files_sha256"].items():
            old = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
            assert sha(old) == digest == sha((ROOT / path).read_bytes())
        for family, settings in config["families"].items():
            assert sha((ROOT / settings["binary"]).read_bytes()) == env["binary_sha256"][family]
            for path in settings.get("gates", []):
                gate = json.loads((ROOT / path).read_text())
                assert gate["status"] == "passed" and gate["binary_sha256"] == settings["binary_sha256"]
        rows = [json.loads(line) for line in (directory / "screen.jsonl").read_text().splitlines()]
        assert len(rows) == calls
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["rows"] == calls
        for group in summary["groups"]:
            observed = sorted((row for row in rows if
                               (row["case_id"], row["mode_label"], row["workers"]) ==
                               (group["case_id"], group["mode"], group["workers"])),
                              key=lambda row: row["repeat"])
            assert [row["repeat"] for row in observed] == [1, 2]
            assert all(row["full_trace_match"] and row["fingerprint"] == group["fingerprint"]
                       for row in observed)
            assert [row["call_seconds"] for row in observed] == group["raw_call_seconds"]
        print(name, "passed:", calls, "calls; capsules, binaries, gates and shared base match")


if __name__ == "__main__":
    main()
