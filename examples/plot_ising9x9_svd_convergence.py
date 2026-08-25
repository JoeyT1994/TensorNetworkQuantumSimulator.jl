"""Dependency-free convergence summary for 9x9 D=3 SVD-CTMRG data."""

from __future__ import annotations

import csv
import math
from pathlib import Path


DATA_DIR = Path(
    r"C:\Users\Joey\Documents\Python\Projects\Work\QuantumPhysics\Data Analysis and Plotting\CTMRG\_PLOTS"
)
F_REF = -38.024120943923315


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def high_chi() -> list[dict[str, float]]:
    source = rows(DATA_DIR / "ising9x9_D3_meanX_highchi_reference.csv")
    source += rows(DATA_DIR / "ising9x9_D3_svd_norm_meanX_chi72_96.csv")
    return sorted(({"chi": int(r["chi"]),
                    "zerr": max(abs(math.expm1(float(r["F"]) - F_REF)), 1e-16),
                    "xerr": max(float(r["X_abs_error"]), 1e-16)}
                   for r in source if r["method"] == "cut"), key=lambda r: r["chi"])


def amplitudes() -> list[dict[str, float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows(DATA_DIR / "ising9x9_D3_svd_random_amplitudes.csv"):
        if row["method"] != "cut":
            continue
        grouped.setdefault(int(row["chi"]), []).append(max(float(row["relative_error"]), 1e-16))
    return [{"chi": chi, "mean": sum(values) / len(values),
             "lo": min(values), "hi": max(values)}
            for chi, values in sorted(grouped.items())]


def marker(x: float, y: float, color: str, radius: float = 5.0) -> str:
    return (f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" '
            f'fill="{color}" stroke="white" stroke-width="1.5"/>')


def panel(svg: list[str], index: int, title: str, ylabel: str,
          data: list[dict[str, float]], field: str, xlim: tuple[float, float],
          ylim: tuple[int, int], xticks: tuple[int, ...], color: str,
          band: bool = False) -> None:
    width, top, height, left, gap = 535, 125, 520, 135, 75
    x0 = left + index * (width + gap)
    y0 = top
    px = lambda x: x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * width
    py = lambda y: y0 + (ylim[1] - math.log10(max(10.0 ** ylim[0], y))) / (ylim[1] - ylim[0]) * height
    for exponent in range(ylim[0], ylim[1] + 1, 2):
        yy = py(10.0 ** exponent)
        svg.append(f'<line x1="{x0}" y1="{yy}" x2="{x0+width}" y2="{yy}" stroke="#ddd"/>')
        svg.append(f'<text class="tick" x="{x0-13}" y="{yy+7}" text-anchor="end">'
                   f'10<tspan dy="-8" font-size="15">{exponent}</tspan></text>')
    for tick in xticks:
        xx = px(tick)
        svg.append(f'<line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y0+height}" stroke="#eee"/>')
        svg.append(f'<text class="tick" x="{xx}" y="{y0+height+34}" text-anchor="middle">{tick}</text>')
    svg.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#222" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{py(1e-16)}" x2="{x0+width}" y2="{py(1e-16)}" stroke="#777" stroke-width="2" stroke-dasharray="4 7"/>')
    svg.append(f'<text class="label" x="{x0-24}" y="{y0-55}">({chr(97+index)})</text>')
    svg.append(f'<text class="title" x="{x0+width/2}" y="{y0-28}" text-anchor="middle">{title}</text>')
    if band:
        top_points = " ".join(f'{px(r["chi"]):.3f},{py(r["hi"]):.3f}' for r in data)
        bottom_points = " ".join(f'{px(r["chi"]):.3f},{py(r["lo"]):.3f}' for r in reversed(data))
        svg.append(f'<polygon points="{top_points} {bottom_points}" fill="{color}" fill-opacity="0.17"/>')
    points = " ".join(f'{px(r["chi"]):.3f},{py(r[field]):.3f}' for r in data)
    svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for row in data:
        svg.append(marker(px(row["chi"]), py(row[field]), color))
    svg.append(f'<text class="axis" x="{x0+width/2}" y="{y0+height+82}" text-anchor="middle">Boundary bond dimension χ</text>')
    svg.append(f'<text class="axis" transform="translate({x0-88} {y0+height/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')


def main() -> None:
    high = high_chi()
    amps = amplitudes()
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2048" height="780" viewBox="0 0 2048 780">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Times New Roman,serif;fill:#111}.tick{font-size:22px}.title{font-size:27px}.axis{font-size:28px}.label{font-size:30px;font-weight:bold}</style>',
        '<text x="1024" y="42" text-anchor="middle" font-size="28">9 × 9 D = 3 TFIM PEPS — SVD-CTMRG convergence</text>',
    ]
    panel(svg, 0, 'Norm Z = ⟨ψ|ψ⟩', 'Relative error in Z', high, 'zerr',
          (38, 98), (-16, -10), (40, 56, 72, 88, 96), '#2A78B8')
    panel(svg, 1, 'Lattice-average ⟨X⟩', 'Absolute error in ⟨X⟩', high, 'xerr',
          (38, 98), (-16, -10), (40, 56, 72, 88, 96), '#D95F02')
    panel(svg, 2, 'Five random amplitudes ⟨s|ψ⟩', 'Mean relative error', amps, 'mean',
          (1, 49), (-16, -2), (2, 8, 16, 24, 32, 40, 48), '#1B9E77', band=True)
    svg.append('<text x="1940" y="115" text-anchor="end" font-size="20" fill="#1B9E77">band: min–max across samples</text>')
    svg.append('</svg>')
    (DATA_DIR / "ising9x9_D3_svd_convergence_summary.svg").write_text("".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
