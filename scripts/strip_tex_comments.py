"""Make an upload-ready copy of the LaTeX sources with every comment removed.

Full-line comments (lines whose first non-blank character is %) are dropped; trailing comments are cut at the
first unescaped %; \\% stays. Verbatim/comment environments are not special-cased, so keep code listings out
of the sources or check the output. Every .tex and .bib under the source directory is processed; other files
(figures, .bst, .cls, .sty) are copied unchanged.

    python scripts/strip_tex_comments.py paper/src paper/arxiv
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TRAILING = re.compile(r"(?<!\\)%.*$")


def strip_line(line: str) -> str | None:
    if line.lstrip().startswith("%"):
        return None
    out = TRAILING.sub("", line).rstrip()
    return out + "\n"


def strip_file(src: Path, dst: Path) -> tuple[int, int]:
    kept, dropped = [], 0
    for line in src.read_text(encoding="utf-8").splitlines(keepends=True):
        s = strip_line(line)
        if s is None:
            dropped += 1
        else:
            kept.append(s)
    dst.write_text("".join(kept), encoding="utf-8")
    return len(kept), dropped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="directory with the LaTeX sources")
    p.add_argument("dst", help="output directory (created; existing files overwritten)")
    args = p.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    for f in sorted(src.rglob("*")):
        if f.is_dir() or ".git" in f.parts:
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in (".tex", ".bib"):
            kept, dropped = strip_file(f, out)
            print(f"{rel}: {kept} lines kept, {dropped} comment lines removed")
        else:
            shutil.copy2(f, out)
    leftovers = [str(q.relative_to(dst)) for q in dst.rglob("*.tex") if "PENDING" in q.read_text(encoding="utf-8")]
    if leftovers:
        print("WARNING: [PENDING] markers remain in:", ", ".join(leftovers))


if __name__ == "__main__":
    main()
