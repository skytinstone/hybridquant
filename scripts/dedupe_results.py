"""Remove duplicate rows from results/*.jsonl, keeping the first row of each run.

Duplicates appear when a runner is started twice on the same results file (both copies append the
same runs). Run this only on files whose step has finished; a file that is still being appended to
must not be rewritten.

    python scripts/dedupe_results.py                      # every results/*.jsonl
    python scripts/dedupe_results.py results/grid_0.8b.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os

KEYS = ("model", "label", "seed", "state_bits", "layer_idx", "context_len", "depth", "idx")


def run_key(row: dict) -> tuple:
    return tuple(row.get(k) for k in KEYS)


def dedupe_file(path: str, dry_run: bool = False) -> tuple[int, int]:
    seen, keep, n = set(), [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            k = run_key(json.loads(line))
            if k in seen:
                continue
            seen.add(k)
            keep.append(line if line.endswith("\n") else line + "\n")
    if len(keep) < n and not dry_run:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return n, len(keep)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="jsonl files (default: results/*.jsonl)")
    p.add_argument("--dry-run", action="store_true", help="report only, do not rewrite")
    args = p.parse_args(argv)
    paths = args.paths or sorted(glob.glob("results/*.jsonl"))
    for path in paths:
        n, kept = dedupe_file(path, args.dry_run)
        flag = "" if kept == n else ("  (would remove %d)" % (n - kept) if args.dry_run else "  (removed %d)" % (n - kept))
        print(f"{path:45s} {n:6d} -> {kept:6d}{flag}")


if __name__ == "__main__":
    main()
