#!/usr/bin/env python3
"""Render the learned alarm's Pareto frontier from the sealed artifacts.

Every plotted coordinate is recomputed from
``results/experiments/router-loop-v{2,3,4}-sealed-1.json`` rather than
transcribed, so the figure cannot drift from the record it depicts.

Output: ``docs/figures/alarm-frontier.svg`` (vector, for the paper and the
blog alike).

Form: the data's job is to show a two-error-mode tradeoff across three
sealed operating points, and to make one point's DOMINANCE of another
visible. Dominance is a spatial relation — up and to the right — so a
scatter is the form that carries it; a table cannot.

Color: categorical, three identities. Scatter compares all pairs, and only
the first three slots of the reference palette clear the all-pairs gates,
so the always-discover reference is drawn in muted ink as chrome rather
than taking a fourth categorical slot. Validated:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" \
         --mode light --pairs all
    -> ALL CHECKS PASS; aqua carries a sub-3:1 contrast WARN, so the
       relief rule applies and every point is directly labeled.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "alarm-frontier.svg"

# --- palette (reference instance, light surface) -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # slots 1-3, all-pairs validated

# --- geometry ------------------------------------------------------------
W, H = 900, 560
L, R, T, B = 92, 48, 78, 78
PX0, PX1 = L, W - R
PY0, PY1 = T, H - B
XD0, XD1 = 80.0, 102.0
YD0, YD1 = 55.0, 103.0


def sx(v: float) -> float:
    return PX0 + (v - XD0) / (XD1 - XD0) * (PX1 - PX0)


def sy(v: float) -> float:
    return PY1 - (v - YD0) / (YD1 - YD0) * (PY1 - PY0)


def load(tag: str) -> dict:
    path = ROOT / "results" / "experiments" / f"router-loop-{tag}-sealed-1.json"
    d = json.loads(path.read_text())
    desc = d["audit"]["descriptive"]
    acc = desc["accuracy_per_condition_arm"]
    inv = desc["discovery_invocations_per_condition_arm"]
    harm = [
        c for c in d["claims"]["claims"] if c["id"].endswith("inlibrary-harm")
    ][0]
    conditions = ("in_library", "out_of_library", "null_control")
    cal = desc.get("calibration") or {}
    return {
        "acquisition": acc["out_of_library/arch_full"] * 100.0,
        "in_library": acc["in_library/arch_full"] * 100.0,
        "e2e": sum(acc[f"{c}/arch_full"] for c in conditions) / 3.0 * 100.0,
        "invocations": sum(inv[f"{c}/arch_full"] for c in conditions),
        "reference_invocations": sum(
            inv[f"{c}/discovery_only"] for c in conditions
        ),
        "harm_seeds": round(-harm["estimate"] * harm["n"]),
        "n": harm["n"],
        "threshold": cal.get("threshold"),
    }


points = [
    dict(load("v2"), label="loop-v2", rule="frozen threshold 0.5", color=SERIES[0]),
    dict(load("v3"), label="loop-v3", rule="bounded FQ ≤ 0.02", color=SERIES[1]),
    dict(load("v4"), label="loop-v4", rule="cost-aware, 1:1", color=SERIES[2]),
]
ref_invocations = points[0]["reference_invocations"]

parts: list[str] = []
add = parts.append

add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}" font-family="system-ui,-apple-system,'
    f'&quot;Segoe UI&quot;,sans-serif" role="img" '
    f'aria-label="The learned alarm’s Pareto frontier across three '
    f'sealed operating points.">'
)
add(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

# --- titles --------------------------------------------------------------
add(
    f'<text x="{L}" y="34" font-size="19" font-weight="600" fill="{INK}">'
    f"The alarm is a one-parameter dial — the third setting pushed "
    f"the frontier</text>"
)
add(
    f'<text x="{L}" y="56" font-size="13.5" fill="{INK_2}">'
    f"Full-architecture accuracy per condition, {points[0]['n']} sealed seeds "
    f"per experiment. Up and to the right is better.</text>"
)

# --- grid and axes -------------------------------------------------------
for v in (60, 70, 80, 90, 100):
    y = sy(v)
    add(
        f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" '
        f'stroke="{GRID}" stroke-width="1"/>'
    )
    add(
        f'<text x="{PX0 - 12}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" '
        f'text-anchor="end" font-variant-numeric="tabular-nums">{v}%</text>'
    )
for v in (80, 85, 90, 95, 100):
    x = sx(v)
    add(
        f'<line x1="{x:.1f}" y1="{PY0}" x2="{x:.1f}" y2="{PY1}" '
        f'stroke="{GRID}" stroke-width="1"/>'
    )
    add(
        f'<text x="{x:.1f}" y="{PY1 + 22}" font-size="12" fill="{MUTED}" '
        f'text-anchor="middle" font-variant-numeric="tabular-nums">{v}%</text>'
    )
add(
    f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="{AXIS}" '
    f'stroke-width="1.5"/>'
)
add(
    f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="{AXIS}" '
    f'stroke-width="1.5"/>'
)
add(
    f'<text x="{(PX0 + PX1) / 2:.0f}" y="{H - 26}" font-size="13" '
    f'fill="{INK_2}" text-anchor="middle">Out-of-library acquisition '
    f'— what a false quiet costs →</text>'
)
add(
    f'<text transform="translate(26,{(PY0 + PY1) / 2:.0f}) rotate(-90)" '
    f'font-size="13" fill="{INK_2}" text-anchor="middle">'
    f'In-library accuracy — what a false alarm costs →</text>'
)

# --- always-discover reference (chrome, not a categorical series) --------
rx, ry = sx(100.0), sy(100.0)
add(
    f'<line x1="{PX0}" y1="{ry:.1f}" x2="{PX1}" y2="{ry:.1f}" '
    f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="2 4" '
    f'opacity="0.85"/>'
)
add(
    f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="5.5" fill="none" '
    f'stroke="{MUTED}" stroke-width="2"/>'
)
add(
    f'<text x="{rx - 14:.1f}" y="{ry - 12:.1f}" font-size="12.5" '
    f'fill="{INK_2}" text-anchor="end">always-discover: perfect, '
    f'{ref_invocations} discovery calls</text>'
)

# --- the dominance annotation -------------------------------------------
v2, v4 = points[0], points[2]
add(
    f'<line x1="{sx(v2["acquisition"]):.1f}" y1="{sy(v2["in_library"]):.1f}" '
    f'x2="{sx(v4["acquisition"]) - 16:.1f}" '
    f'y2="{sy(v4["in_library"]):.1f}" stroke="{INK}" stroke-width="1.5" '
    f'stroke-dasharray="5 4" opacity="0.5"/>'
)
add(
    f'<polygon points="{sx(v4["acquisition"]) - 16:.1f},'
    f'{sy(v4["in_library"]) - 4.5:.1f} {sx(v4["acquisition"]) - 6:.1f},'
    f'{sy(v4["in_library"]):.1f} {sx(v4["acquisition"]) - 16:.1f},'
    f'{sy(v4["in_library"]) + 4.5:.1f}" fill="{INK}" opacity="0.5"/>'
)
# The callout goes in the empty lower-left quadrant rather than squeezed
# between the connector and the reference line, where it collided with both.
gain = v4["acquisition"] - v2["acquisition"]
cx, cy = sx(84.2), sy(80.0)
add(
    f'<text x="{cx:.0f}" y="{cy:.0f}" font-size="14" font-weight="600" '
    f'fill="{INK}">loop-v4 dominates loop-v2</text>'
)
add(
    f'<text x="{cx:.0f}" y="{cy + 20:.0f}" font-size="12.5" fill="{INK_2}">'
    f'same in-library accuracy, same −1/{v2["n"]} harm,</text>'
)
add(
    f'<text x="{cx:.0f}" y="{cy + 37:.0f}" font-size="12.5" fill="{INK_2}">'
    f'+{gain:.1f} points of acquisition</text>'
)

# --- the three operating points ------------------------------------------
placements = {
    "loop-v2": ("start", 16, 6),
    "loop-v3": ("end", -16, 6),
    "loop-v4": ("middle", 0, 40),
}
for p in points:
    x, y = sx(p["acquisition"]), sy(p["in_library"])
    add(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{p["color"]}" '
        f'stroke="{SURFACE}" stroke-width="2"/>'
    )
    anchor, dx, dy = placements[p["label"]]
    tx, ty = x + dx, y + dy
    add(
        f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="13.5" font-weight="600" '
        f'fill="{INK}" text-anchor="{anchor}">{p["label"]}</text>'
    )
    add(
        f'<text x="{tx:.1f}" y="{ty + 17:.1f}" font-size="12" fill="{INK_2}" '
        f'text-anchor="{anchor}">{p["rule"]}</text>'
    )
    harm = p["harm_seeds"]
    add(
        f'<text x="{tx:.1f}" y="{ty + 33:.1f}" font-size="12" fill="{MUTED}" '
        f'text-anchor="{anchor}" font-variant-numeric="tabular-nums">'
        f'−{harm}/{p["n"]} harm · {p["invocations"]} calls · '
        f'{p["e2e"]:.1f}% e2e</text>'
    )

add(
    f'<text x="{L}" y="{H - 8}" font-size="11.5" fill="{MUTED}">'
    f"Recomputed from the sealed artifacts; h4 (bounded harm) failed at all "
    f"three points with one identical mechanism.</text>"
)
add("</svg>")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}")
for p in points:
    print(
        f"  {p['label']:8s} acquisition {p['acquisition']:6.2f}%  "
        f"in-library {p['in_library']:6.2f}%  e2e {p['e2e']:6.2f}%  "
        f"harm -{p['harm_seeds']}/{p['n']}  calls {p['invocations']}  "
        f"threshold {p['threshold']}"
    )
