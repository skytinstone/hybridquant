"""Figure 1 of the paper: one hybrid block (3 GDN + 1 attention), the two inference memories and the two
store-time quantization hooks. Pure drawing, no data.

    python analysis/fig_overview.py            # writes paper/figures/overview.pdf and .png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

GDN = "#4C72B0"      # blue
ATT = "#DD8452"      # orange
MEM = "#F4F4F4"      # memory boxes
HOOK = "#C44E52"     # red, quantization hooks
INK = "#222222"
NOTE = "#555555"


def box(ax, x, y, w, h, fc, ec=INK, lw=0.9, radius=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw))


def arrow(ax, p, q, color=INK, lw=0.9, ms=7, rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=(0, (3, 2)) if dashed else "-"))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 4.75); ax.axis("off")

    # ---- left: the hybrid block, bottom to top ----------------------------------------------------
    lx, lw_, lh = 0.6, 2.2, 0.6
    ys = [0.5, 1.3, 2.1, 2.9]
    for y, name, c in zip(ys, ["GDN layer"] * 3 + ["Attention layer"], [GDN] * 3 + [ATT]):
        box(ax, lx, y, lw_, lh, c, ec="none")
        ax.text(lx + lw_ / 2, y + lh / 2, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    cx = lx + lw_ / 2
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (cx, y0 + lh), (cx, y1), ms=6)
    arrow(ax, (cx, 0.12), (cx, ys[0]), ms=6);            ax.text(cx + 0.12, 0.2, r"$x_t$", fontsize=9, va="center")
    arrow(ax, (cx, ys[-1] + lh), (cx, 3.95), ms=6);      ax.text(cx + 0.12, 3.88, r"$o_t$", fontsize=9, va="center")
    bx = lx - 0.2
    ax.plot([bx, bx - 0.1, bx - 0.1, bx], [ys[0], ys[0], ys[-1] + lh, ys[-1] + lh], color=INK, lw=0.8)
    ax.text(bx - 0.17, (ys[0] + ys[-1] + lh) / 2, r"$\times\,L/4$", ha="right", va="center", fontsize=8.5, rotation=90)

    # ---- middle: the two hooks ---------------------------------------------------------------------
    hx, hw, hh = 3.45, 0.85, 0.5
    sy = ys[1] + lh / 2 - hh / 2          # state hook, level with the middle GDN layer
    ky = ys[3] + lh / 2 - hh / 2          # KV hook, level with the attention layer
    for y, lab in ((sy, r"$Q_{b_s}$"), (ky, r"$Q_{b_{kv}}$")):
        box(ax, hx, y, hw, hh, "white", ec=HOOK, lw=1.3)
        ax.text(hx + hw / 2, y + hh / 2, lab, ha="center", va="center", fontsize=10, color=HOOK)
    ax.text(hx + hw / 2, (sy + hh + ky) / 2, "store-time\nquantizers", ha="center", va="center", fontsize=6.8,
            color=HOOK, linespacing=1.1)
    for y in ys[:3]:                                                   # GDN writes -> state hook
        arrow(ax, (lx + lw_, y + lh / 2), (hx, sy + hh / 2), color=GDN, ms=6)
    arrow(ax, (lx + lw_, ys[3] + lh / 2), (hx, ky + hh / 2), color=ATT, ms=6)   # attention write -> KV hook
    # error-feedback residual loop under the state hook
    ax.add_patch(FancyArrowPatch((hx + hw * 0.8, sy), (hx + hw * 0.2, sy), connectionstyle="arc3,rad=-0.8",
                                 arrowstyle="-|>", mutation_scale=5, color=HOOK, lw=0.7, linestyle=(0, (2, 1.5))))
    ax.text(hx + hw / 2, sy - 0.42, r"EF residual $r_t$", ha="center", va="top", fontsize=6.5, color=HOOK)

    # ---- right: the two memories -------------------------------------------------------------------
    mx, mw = 4.85, 5.05
    tx = mx + 0.15                       # text column
    gx = mx + 3.75                       # glyph column
    body = dict(ha="left", va="top", fontsize=7.0)
    note = dict(ha="left", va="top", fontsize=6.8, style="italic", color=NOTE, linespacing=1.15)

    # KV cache (top)
    ky0, kh = 2.55, 1.65
    box(ax, mx, ky0, mw, kh, MEM, ec=ATT, lw=1.2, radius=0.1)
    ax.text(tx, ky0 + kh - 0.14, "KV cache", ha="left", va="top", fontsize=9, fontweight="bold", color=ATT)
    ax.text(tx, ky0 + kh - 0.48, r"one entry per past token: $(\tilde k_i, \tilde v_i)$, $i \leq t$", **body)
    ax.text(tx, ky0 + kh - 0.76, r"$|\mathrm{KV}|(T) = 2\,L_a H_{kv}\, d\; T$,  grows with $T$", **body)
    ax.text(tx, ky0 + kh - 1.08, "rounding error is committed once\nand never changes", **note)
    for i in range(7):                                                 # token cells, newest filled
        ax.add_patch(FancyBboxPatch((gx + i * 0.16, ky0 + 0.62), 0.12, 0.42, boxstyle="square,pad=0",
                                    fc=ATT if i == 6 else "white", ec=ATT, lw=0.7))
    ax.text(gx + 3.5 * 0.16, ky0 + 0.5, r"tokens $1 \ldots t$", ha="center", va="top", fontsize=6.5, color=NOTE)

    # recurrent state (bottom)
    sy0, sh = 0.4, 1.65
    box(ax, mx, sy0, mw, sh, MEM, ec=GDN, lw=1.2, radius=0.1)
    ax.text(tx, sy0 + sh - 0.14, "Recurrent state", ha="left", va="top", fontsize=9, fontweight="bold", color=GDN)
    ax.text(tx, sy0 + sh - 0.48, r"one matrix per GDN layer: $\tilde S_t \in \mathbb{R}^{H \times d_k \times d_v}$", **body)
    ax.text(tx, sy0 + sh - 0.76, r"$|S| = L_g\, H\, d_k d_v$,  fixed, independent of $T$", **body)
    ax.text(tx, sy0 + sh - 1.08, "rewritten every token: error is damped by\n"
            r"the decay $\alpha_t$, replenished each step (Eq. 9)", **note)
    for r in range(3):                                                 # state matrix glyph
        for c in range(4):
            ax.add_patch(FancyBboxPatch((gx + 0.3 + c * 0.2, sy0 + 0.62 + r * 0.2), 0.16, 0.16,
                                        boxstyle="square,pad=0", fc="white", ec=GDN, lw=0.6))
    ax.text(gx + 0.3 + 0.38, sy0 + 1.3, r"$\tilde S_t$", ha="center", va="bottom", fontsize=7, color=NOTE)

    # writes (hook -> memory) and reads (memory -> layer, dashed, routed outside)
    arrow(ax, (hx + hw, ky + hh / 2), (mx, ky0 + kh / 2), color=HOOK, lw=1.0)
    arrow(ax, (hx + hw, sy + hh / 2), (mx, sy0 + sh / 2), color=HOOK, lw=1.0)
    ax.text((hx + hw + mx) / 2, (ky + hh / 2 + ky0 + kh / 2) / 2 + 0.08, "write", ha="center", va="bottom", fontsize=6.5, color=HOOK)
    ax.text((hx + hw + mx) / 2, (sy + hh / 2 + sy0 + sh / 2) / 2 + 0.08, "write", ha="center", va="bottom", fontsize=6.5, color=HOOK)
    arrow(ax, (mx, ky0 + kh - 0.1), (lx + lw_, ys[3] + lh - 0.08), color=ATT, lw=0.8, rad=-0.32, dashed=True)
    arrow(ax, (mx, sy0 + 0.1), (lx + lw_, ys[0] + 0.08), color=GDN, lw=0.8, rad=-0.32, dashed=True)
    ax.text(3.85, 4.5, "read (dequantized)", ha="center", va="center", fontsize=6.5, color=ATT)
    ax.text(3.85, -0.08, "read (dequantized)", ha="center", va="center", fontsize=6.5, color=GDN)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"overview.{ext}", bbox_inches="tight", dpi=300)
    print("wrote", os.path.relpath(OUT / "overview.pdf", ROOT), "and .png")


if __name__ == "__main__":
    main()
