#!/usr/bin/env python3
"""Plot examples/data/ctm_vs_bmps_ising11_h0.01.csv.

Companion to examples/ctm_vs_bmps_ising_field.jl, which generates the CSV.
Mirrors Fig. 3 of the "Matrix Product Belief Propagation" draft: finite 11x11
classical Ising in a field, single-site magnetisation at a corner and at the
centre, CTMRG (:cut, :cycle) against boundary MPS at matched bond dimension.

    python3 examples/plot_ctm_vs_bmps.py            # writes the PNG next to the CSV
    python3 examples/plot_ctm_vs_bmps.py --show     # and opens a window

Needs only numpy + matplotlib.
"""

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NAME = "ctm_vs_bmps_ising11_h0.01.csv"
# Works whether the CSV sits beside this script or in the repo's examples/data/.
CSV = next((p for p in (HERE / NAME, HERE / "data" / NAME) if p.exists()),
           HERE / "data" / NAME)

# Categorical slots 1-3 of the reference palette, in fixed order. Colour follows the
# METHOD, so it stays put if a series is dropped.
CUT, CYCLE, BMPS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"

# A relative error of exactly 0 cannot be drawn on a log axis. The reference is itself
# a ratio of two 11x11 contractions, so nothing below ~1e-14 is meaningful anyway;
# clip to the floor and shade that region rather than silently dropping points.
FLOOR = 1e-16
REF_FLOOR = 1e-14


def read_csv(path):
    if not path.exists():
        sys.exit(f"missing {path}\nGenerate it first:\n"
                 f"  julia --project=. --startup-file=no examples/ctm_vs_bmps_ising_field.jl")
    with open(path) as fh:
        rows = list(csv.DictReader(r for r in fh if not r.startswith("#")))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    cols["chi"] = cols["chi"].astype(int)
    return cols


def error_panel(ax, chi, series, title):
    """`series` entries are (label, y, colour) or (label, (y, lo, hi), colour).

    The banded form is used by the RBIM dataset, where y is a median over disorder
    realisations and lo/hi are the min/max across them.
    """
    for label, y, colour in series:
        if isinstance(y, tuple):
            y, lo, hi = y
            ax.fill_between(chi, np.clip(lo, FLOOR, None), np.clip(hi, FLOOR, None),
                            color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(chi, np.clip(y, FLOOR, None), marker="o", markersize=5.5,
                linewidth=1.8, color=colour, label=label, clip_on=False, zorder=3)
    ax.axhspan(FLOOR, REF_FLOOR, color=INK_3, alpha=0.10, lw=0, zorder=0)
    ax.text(chi[-1], REF_FLOOR * 1.6, "exact-reference floor", ha="right", va="bottom",
            fontsize=7.5, color=INK_3)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.set_xlabel("bond dimension  $\\chi$", fontsize=9, color=INK_2)
    style(ax)


def style(ax):
    ax.grid(True, which="major", axis="both", color=INK_3, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--csv", default=str(CSV))
    args = ap.parse_args()

    d = read_csv(pathlib.Path(args.csv))
    chi = d["chi"]

    # The RBIM dataset carries disorder bands (*_med/_lo/_hi) and a single site; the clean
    # Ising dataset carries two sites and no bands. One script, two schemas.
    if "cut_med" in d:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
        error_panel(axes[0], chi,
                    [(":cut", (d["cut_med"], d["cut_lo"], d["cut_hi"]), CUT),
                     (":cycle", (d["cycle_med"], d["cycle_lo"], d["cycle_hi"]), CYCLE),
                     ("boundary MPS", (d["bmps_med"], d["bmps_lo"], d["bmps_hi"]), BMPS)],
                    "RBIM 10x10, site (4,4)")
        axes[0].set_ylabel(r"$|\delta m| / |m|$", fontsize=9, color=INK_2)
        axes[1].axis("off")
        axes[1].text(0.0, 0.5,
                     "median over 5 disorder realisations;\n"
                     "band = min-max across them.\n\n"
                     "Negative bonds make the network\n"
                     "strongly non-Hermitian -- the regime\n"
                     "the draft picks the RBIM for.\n\n"
                     "bMPS freezes at 1.35e-11 for\n"
                     "chi = 12..16 (7 s.f. identical).",
                     fontsize=8.5, color=INK_2, va="center", ha="left")
        for ax in axes[:1]:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6)
        fig.suptitle("Finite 10x10 random-bond Ising at the Nishimori point, $h=0.01$ — "
                     "CTMRG vs boundary MPS", fontsize=11, color=INK)
        out = pathlib.Path(args.csv).with_suffix(".png")
        fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {out}")
        if args.show:
            plt.show()
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    error_panel(axes[0], chi,
                [(":cut", d["cut_corner"], CUT),
                 (":cycle", d["cycle_corner"], CYCLE),
                 ("boundary MPS", d["bmps_corner"], BMPS)],
                "CORNER site (1,1)")
    axes[0].set_ylabel(r"$|\delta m| / |m|$", fontsize=9, color=INK_2)

    error_panel(axes[1], chi,
                [(":cut", d["cut_centre"], CUT),
                 (":cycle", d["cycle_centre"], CYCLE),
                 ("boundary MPS", d["bmps_centre"], BMPS)],
                "CENTRE site (6,6)")

    # Separate panel, not a second y-axis: this is a different quantity. bMPS has no
    # analogue, so only two series appear here.
    ax = axes[2]
    ax.plot(chi, np.clip(d["marg_cut"], FLOOR, None), marker="o", markersize=5.5,
            linewidth=1.8, color=CUT, label=":cut", clip_on=False, zorder=3)
    ax.plot(chi, np.clip(d["marg_cycle"], FLOOR, None), marker="o", markersize=5.5,
            linewidth=1.8, color=CYCLE, label=":cycle", clip_on=False, zorder=3)
    ax.set_yscale("log")
    ax.set_title(r"stationarity residual  $\partial_X Z$", fontsize=10, color=INK, pad=8)
    ax.set_xlabel("bond dimension  $\\chi$", fontsize=9, color=INK_2)
    ax.set_ylabel("marginal_inconsistency", fontsize=9, color=INK_2)
    style(ax)

    for ax in axes:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6)

    fig.suptitle("Finite 11x11 classical Ising, $\\beta=0.4407$, $h=0.01$ — "
                 "CTMRG vs boundary MPS at matched $\\chi$",
                 fontsize=11, color=INK)

    out = pathlib.Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
