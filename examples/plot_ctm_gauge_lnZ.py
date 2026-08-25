"""Dependency-free Mike-style semilog gauge plots for the 5x5 and 9x9 PEPS."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_DATA_DIR = Path(
    r"C:\Users\Joey\Documents\Python\Projects\Work\QuantumPhysics\Data Analysis and Plotting\CTMRG\_PLOTS"
)
REFERENCES = {"5x5": -6.217866847854575, "9x9": -38.024120943923315}
METHODS = (
    ("bmps", 'Z<tspan baseline-shift="sub" font-size="20">bMPS</tspan>',
     "#2A78B8", "square"),
    ("cut", 'SVD-CTMRG w/ Z<tspan baseline-shift="sub" font-size="20">B</tspan>',
     "#D94B45", "circle"),
    ("cycle", 'eig-CTMRG / MP-BP w/ Z<tspan baseline-shift="sub" font-size="20">B</tspan>',
     "#1B9E77", "triangle"),
)
GAUGES = (
    ("original", "original", "", 1.00),
    ("vidal", "Vidal/BP", "12 6", 0.80),
    ("g1", "random G₁", "2 5", 0.65),
    ("g2", "random G₂", "13 4 2 4", 0.50),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(data_dir: Path, size: str,
              preconditioned_cycle: bool = False) -> list[dict[str, object]]:
    if size == "5x5":
        cycle_name = ("ising5x5_D3_gauge_cycle_kappa5_preconditioned.csv"
                      if preconditioned_cycle else
                      "ising5x5_D3_gauge_cycle_kappa5.csv")
        source = (read_csv(data_dir / cycle_name)
                  + read_csv(data_dir / "ising5x5_D3_gauge_cut_bmps_kappa5.csv"))
        gauge_map = {"original": "original", "vidal_bp": "vidal",
                     "random_g1": "g1", "random_g2": "g2"}
        rows = [{"chi": int(r["chi"]), "method": r["method"],
                 "gauge": gauge_map[r["gauge"]], "F": float(r["F"]),
                 "status": r["status"]} for r in source]
    else:
        source = read_csv(data_dir / "ising9x9_D3_gauge_lnZ_kappa5.csv")
        gauge_map = {("raw", "identity"): "original",
                     ("vidal", "symmetric"): "vidal",
                     ("raw", "random_a"): "g1", ("raw", "random_b"): "g2"}
        rows = [{"chi": int(r["chi"]), "method": r["method"],
                 "gauge": gauge_map[(r["preconditioner"], r["attack"])],
                 "F": float(r["F"]), "status": r["status"]} for r in source]
    expected = 8 * len(METHODS) * len(GAUGES)
    if len(rows) != expected:
        raise ValueError(f"{size}: expected {expected} rows, found {len(rows)}")
    for row in rows:
        row["error"] = max(abs(math.expm1(row["F"] - REFERENCES[size])), 1e-16)
    return rows


def marker(shape: str, x: float, y: float, color: str, opacity: float,
           converged: bool) -> str:
    fill = color if converged else "white"
    common = f'fill="{fill}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="2"'
    if shape == "square":
        return f'<rect x="{x-5}" y="{y-5}" width="10" height="10" {common}/>'
    if shape == "triangle":
        points = f"{x},{y-6} {x-5.5},{y+5} {x+5.5},{y+5}"
        return f'<polygon points="{points}" {common}/>'
    return f'<circle cx="{x}" cy="{y}" r="5" {common}/>'


def plot_size(data_dir: Path, size: str,
              preconditioned_cycle: bool = False) -> None:
    rows = load_rows(data_dir, size, preconditioned_cycle)
    width, height = 2048, 780
    left, right, top, bottom, gap = 155, 45, 125, 105, 75
    panel_w = (width - left - right - 2 * gap) / 3
    panel_h = height - top - bottom
    xmin, xmax, ymin_exp, ymax_exp = 3, 33, -16, 0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Times New Roman,serif;fill:#111}.tick{font-size:23px}'
        '.title{font-size:28px}.axis{font-size:31px}.legend{font-size:21px}'
        '.panel-label{font-size:31px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="42" text-anchor="middle" font-size="27">'
        f'{size.replace("x", " × ")} D = 3 TFIM PEPS, random-gauge κ = 5</text>',
    ]
    for panel, (method, title, color, shape) in enumerate(METHODS):
        if method == "cycle" and preconditioned_cycle:
            title = ('eig-CTMRG + Vidal precondition w/ '
                     'Z<tspan baseline-shift="sub" font-size="20">B</tspan>')
        x0 = left + panel * (panel_w + gap)
        y0 = top
        px = lambda x: x0 + (x - xmin) / (xmax - xmin) * panel_w
        py = lambda y: y0 + (ymax_exp - math.log10(max(1e-16, min(1.0, y)))) / 16 * panel_h
        for exponent in range(ymin_exp, ymax_exp + 1, 2):
            yy = py(10.0 ** exponent)
            svg.append(f'<line x1="{x0}" y1="{yy}" x2="{x0+panel_w}" y2="{yy}" '
                       'stroke="#d7d7d7" stroke-width="1"/>')
            if panel == 0:
                svg.append(f'<text class="tick" x="{x0-15}" y="{yy+8}" text-anchor="end">'
                           f'10<tspan dy="-9" font-size="16">{exponent}</tspan></text>')
        for chi in (8, 16, 24, 32):
            xx = px(chi)
            svg.append(f'<line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y0+panel_h}" '
                       'stroke="#ececec" stroke-width="1"/>')
            svg.append(f'<text class="tick" x="{xx}" y="{y0+panel_h+36}" '
                       f'text-anchor="middle">{chi}</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" '
                   'fill="none" stroke="#222" stroke-width="2"/>')
        svg.append(f'<line x1="{x0}" y1="{py(1e-16)}" x2="{x0+panel_w}" y2="{py(1e-16)}" '
                   'stroke="#777" stroke-width="2.5" stroke-dasharray="4 7"/>')
        svg.append(f'<text class="panel-label" x="{x0-22}" y="{y0-60}">'
                   f'({chr(ord("a")+panel)})</text>')
        svg.append(f'<text class="title" x="{x0+panel_w/2}" y="{y0-30}" '
                   f'text-anchor="middle">{title}</text>')
        for gauge, _, dash, opacity in GAUGES:
            series = sorted((r for r in rows if r["method"] == method and r["gauge"] == gauge),
                            key=lambda r: r["chi"])
            points = " ".join(f'{px(r["chi"])},{py(r["error"])}' for r in series)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                       f'stroke-opacity="{opacity}" stroke-width="3"{dash_attr}/>')
            for row in series:
                svg.append(marker(shape, px(row["chi"]), py(row["error"]), color, opacity,
                                  row["status"] == "ok"))
        lx, ly = x0 + panel_w - 184, y0 + 32
        svg.append(f'<rect x="{lx-15}" y="{ly-25}" width="184" height="126" rx="4" '
                   'fill="white" fill-opacity="0.90" stroke="#bbb"/>')
        for i, (_, label, dash, opacity) in enumerate(GAUGES):
            yy = ly + i * 28
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+42}" y2="{yy}" stroke="{color}" '
                       f'stroke-opacity="{opacity}" stroke-width="3"{dash_attr}/>')
            svg.append(marker(shape, lx + 21, yy, color, opacity, True))
            svg.append(f'<text class="legend" x="{lx+52}" y="{yy+7}">{label}</text>')
        svg.append(f'<text class="axis" x="{x0+panel_w/2}" y="{height-28}" '
                   'text-anchor="middle">Boundary Bond Dimension χ</text>')
    svg.append(f'<text class="axis" transform="translate(48 {top+panel_h/2}) rotate(-90)" '
               'text-anchor="middle">Relative Error in Z</text>')
    svg.append('</svg>')
    suffix = "_preconditioned" if preconditioned_cycle else ""
    (data_dir / f"ising{size}_D3_gauge_lnZ_kappa5{suffix}.svg").write_text(
        "".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sizes", nargs="+", choices=("5x5", "9x9"), default=("5x5", "9x9"))
    parser.add_argument("--preconditioned-cycle", action="store_true",
                        help="use the validated symmetric-gauge cycle CSV (5x5 only)")
    args = parser.parse_args()
    if args.preconditioned_cycle and args.sizes != ["5x5"]:
        parser.error("--preconditioned-cycle currently requires --sizes 5x5")
    for size in args.sizes:
        plot_size(args.data_dir, size, args.preconditioned_cycle)


if __name__ == "__main__":
    main()
