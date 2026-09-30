"""Lightweight host observations; do not alter benchmark or kill other tasks."""

import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent


def main():
    previous = None
    with (OUT / "host-monitor.jsonl").open("x") as output:
        while True:
            progress = json.loads((OUT / "progress.json").read_text())
            ticks = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
            fields = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "monotonic_seconds": time.monotonic(),
                      "progress": progress, "cpu_ticks": ticks,
                      "loadavg": Path("/proc/loadavg").read_text().strip()}
            if previous:
                difference = [value - old for value, old in zip(ticks, previous)]
                total = sum(difference)
                fields["aggregate_steal_fraction"] = difference[7] / total if total else 0
                fields["aggregate_busy_fraction"] = (total - difference[3] - difference[4]) / total if total else 0
            previous = ticks
            for name in ("cpu", "memory"):
                path = Path("/proc/pressure") / name
                if path.exists():
                    fields[name + "_pressure"] = path.read_text().strip()
            output.write(json.dumps(fields) + "\n")
            output.flush()
            if progress["status"] == "passed" or (OUT / "failure.json").exists():
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
