#!/usr/bin/env python3
"""Render per-condition accuracy across the four arms, from the sealed record.

Reads ``results/experiments/router-loop-v4-sealed-1.json`` and recomputes
every bar height from the retained correctness bits, so the figure cannot
drift from the artifact.

Output: ``docs/figures/arms-by-condition.svg``.

Form: the job is a magnitude comparison across two categorical dimensions
(4 arms x 3 conditions) with the zeros carrying the point — grouped bars.
The two zero bars are the finding, so they are drawn as explicit labeled
zeros rather than absent marks.

Color: categorical, four identities, adjacent pairlist (grouped bars place
only neighbours side by side). Validated:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" \
         --mode light
    -> ALL CHECKS PASS; aqua and yellow carry sub-3:1 contrast WARNs, so
       the relief rule applies and every bar is directly value-labeled.
"""

from __future__ import annotations

import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "arms-by-condition.svg"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

ARMS = (
    ("arch_full", "full architecture"),
    ("routing_only", "routing only"),
    ("discovery_only", "always discover"),
    ("generic", "no architecture"),
)
CONDITIONS = (
    ("in_library", "in-library", "the structure is in the library"),
    ("out_of_library", "out-of-library", "the structure is withheld"),
    ("null_control", "null control", "there is no structure"),
)

W, H = 900, 540
L, R = 92, 48
PY0, PY1 = 118, 404
PX0, PX1 = L, W - R

art = json.loads(
    (ROOT / "results/experiments/router-loop-v4-sealed-1.json").read_text()
)
rows = art["raw_rows"]
eligible = sorted({r["seed"] for r in rows})


def accuracy(condition: str, arm: str) -> float:
    bits = [
        1.0 if r["correct"] else 0.0
        for r in rows
        if r["condition"] == condition and r["arm"] == arm
    ]
    return statistics.fmean(bits) * 100.0


invocations = {
    arm: sum(
        r["discovery_invocations"] for r in rows if r["arm"] == arm
    )
    for arm, _ in ARMS
}

parts: list[str] = []
add = parts.append
add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}" font-family="system-ui,-apple-system,'
    f'&quot;Segoe UI&quot;,sans-serif" role="img" aria-label="Per-condition '
    f'accuracy for four arms across three conditions.">'
)
add(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
add(
    f'<text x="{L}" y="34" font-size="19" font-weight="600" fill="{INK}">'
    f"Two arms can never acquire a missing structure — by construction"
    f"</text>"
)
add(
    f'<text x="{L}" y="56" font-size="13.5" fill="{INK_2}">'
    f"Accuracy per condition, {len(eligible)} sealed seeds, loop-v4. The two "
    f"zeros are design constants, not training failures.</text>"
)

# legend
lx = L
for (arm, label), color in zip(ARMS, SERIES):
    add(
        f'<rect x="{lx}" y="72" width="11" height="11" rx="2.5" '
        f'fill="{color}"/>'
    )
    add(
        f'<text x="{lx + 17}" y="82" font-size="12.5" fill="{INK_2}">'
        f"{label}</text>"
    )
    lx += 26 + len(label) * 7.0

# gridlines
for v in (0, 25, 50, 75, 100):
    y = PY1 - v / 100.0 * (PY1 - PY0)
    add(
        f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" '
        f'stroke="{GRID if v else AXIS}" '
        f'stroke-width="{1 if v else 1.5}"/>'
    )
    add(
        f'<text x="{PX0 - 12}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" '
        f'text-anchor="end" font-variant-numeric="tabular-nums">{v}%</text>'
    )

group_gap = 96
gw = (PX1 - PX0 - group_gap * (len(CONDITIONS) - 1)) / len(CONDITIONS)
bar_gap = 2
bw = (gw - bar_gap * (len(ARMS) - 1)) / len(ARMS)

for gi, (cond, cond_label, cond_note) in enumerate(CONDITIONS):
    gx = PX0 + gi * (gw + group_gap)
    for bi, ((arm, _), color) in enumerate(zip(ARMS, SERIES)):
        v = accuracy(cond, arm)
        x = gx + bi * (bw + bar_gap)
        h = v / 100.0 * (PY1 - PY0)
        if h > 0:
            add(
                f'<rect x="{x:.1f}" y="{PY1 - h:.1f}" width="{bw:.1f}" '
                f'height="{h:.1f}" rx="4" fill="{color}"/>'
            )
            # square off the baseline end so the bar is anchored, not floating
            add(
                f'<rect x="{x:.1f}" y="{PY1 - min(h, 6):.1f}" '
                f'width="{bw:.1f}" height="{min(h, 6):.1f}" fill="{color}"/>'
            )
        else:
            add(
                f'<line x1="{x:.1f}" y1="{PY1:.1f}" x2="{x + bw:.1f}" '
                f'y2="{PY1:.1f}" stroke="{color}" stroke-width="3"/>'
            )
        add(
            f'<text x="{x + bw / 2:.1f}" y="{PY1 - h - 8:.1f}" font-size="11.5" '
            f'fill="{INK_2 if v else INK}" text-anchor="middle" '
            f'font-weight="{600 if not v else 400}" '
            f'font-variant-numeric="tabular-nums">{v:.0f}%</text>'
        )
    add(
        f'<text x="{gx + gw / 2:.1f}" y="{PY1 + 26:.1f}" font-size="13.5" '
        f'font-weight="600" fill="{INK}" text-anchor="middle">'
        f"{cond_label}</text>"
    )
    add(
        f'<text x="{gx + gw / 2:.1f}" y="{PY1 + 44:.1f}" font-size="12" '
        f'fill="{MUTED}" text-anchor="middle">{cond_note}</text>'
    )

note_y = PY1 + 86
add(
    f'<text x="{L}" y="{note_y}" font-size="12.5" fill="{INK_2}">'
    f"Routing-only has no discovery channel and the no-architecture model has "
    f"no certified synthesis channel, so neither can</text>"
)
add(
    f'<text x="{L}" y="{note_y + 18}" font-size="12.5" fill="{INK_2}">'
    f"ever acquire a withheld structure — that leg is a property of the "
    f"design, and the protocol declares it before any data.</text>"
)
add(
    f'<text x="{L}" y="{note_y + 42}" font-size="12" fill="{MUTED}">'
    f"The full architecture reaches this with "
    f"{invocations['arch_full']} discovery calls; always-discover needs "
    f"{invocations['discovery_only']}.</text>"
)
add("</svg>")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}  (n={len(eligible)})")
for cond, label, _ in CONDITIONS:
    cells = "  ".join(
        f"{name}={accuracy(cond, arm):5.1f}%" for arm, name in ARMS
    )
    print(f"  {label:16s} {cells}")
print(f"  discovery calls: {invocations}")
