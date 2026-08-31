#!/usr/bin/env python3
"""Deterministic SVG figure for M1 (no plotting dependency, byte-stable output).

    <venv python> m1/plot_m1.py [--metric q4_k_m_kl_mean] [--out m1/work/m1_optionality.svg]

One horizontal bar panel per requested metric, bars labelled by checkpoint, coloured by role
(reference / stressed / repaired / control). Values come from m1/work/m1_summary.json's "num"
blocks, which m1/report.py derives from the probe JSONs, so the figure can never disagree with
the table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROLE = {"base": "reference", "prep_base": "reference"}


def role_of(name: str) -> str:
    if name.endswith("_rep"):
        return "repaired"
    if name.startswith(("g4_", "g6_")):
        return "control"
    if ROLE.get(name):
        return ROLE[name]
    return "stressed" if name != "base" else "reference"


COLOR = {"reference": "#6b7280", "stressed": "#c0392b", "repaired": "#1e8449",
         "control": "#2471a3"}
ROW_H, LABEL_W, BAR_MAX, PAD = 22, 250, 620, 46


def panel(title: str, unit: str, items: list[tuple[str, float]], log_scale: bool) -> tuple[str, int]:
    if not items:
        return f'<text x="16" y="{PAD - 14}" font-size="14" font-weight="bold">{title}: no data</text>', 0
    vals = [max(v, 1e-12) for _, v in items]
    if log_scale:
        lo, hi = math.log10(min(vals)), math.log10(max(vals))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        scale = lambda v: 0.0 if hi == lo else (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo)
        ticks = [10 ** (lo + k * (hi - lo) / 4) for k in range(5)]
    else:
        hi = max(vals) or 1.0
        scale = lambda v: v / hi
        ticks = [hi * k / 4 for k in range(5)]
    h = PAD + ROW_H * len(items) + 34
    out = [f'<text x="16" y="{PAD - 22}" font-size="15" font-weight="bold">{title}</text>']
    for t in ticks:
        x = LABEL_W + scale(t) * BAR_MAX
        out.append(f'<line x1="{x:.1f}" y1="{PAD - 8}" x2="{x:.1f}" y2="{PAD + ROW_H * len(items):.0f}" '
                   f'stroke="#e5e7eb"/>'
                   f'<text x="{x:.1f}" y="{PAD + ROW_H * len(items) + 14:.0f}" font-size="10" '
                   f'text-anchor="middle">{t:.3g}</text>')
    out.append(f'<text x="{LABEL_W + BAR_MAX}" y="{PAD + ROW_H * len(items) + 30:.0f}" font-size="10" '
               f'text-anchor="end" fill="#6b7280">{unit}</text>')
    for i, (name, v) in enumerate(items):
        y = PAD + i * ROW_H
        w = max(1.0, scale(v) * BAR_MAX)
        out.append(f'<rect x="{LABEL_W}" y="{y}" width="{w:.1f}" height="{ROW_H - 8}" '
                   f'fill="{COLOR[role_of(name)]}" rx="2"/>')
        out.append(f'<text x="{LABEL_W - 8}" y="{y + ROW_H - 12}" font-size="11" text-anchor="end" '
                   f'font-family="monospace">{name}</text>')
        out.append(f'<text x="{LABEL_W + w + 6:.1f}" y="{y + ROW_H - 12}" font-size="10">{v:.4g}</text>')
    return "\n".join(out), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(common.WORK / "m1_summary.json"))
    ap.add_argument("--out", default=str(common.WORK / "m1_optionality.svg"))
    ap.add_argument("--metric", action="append", default=None,
                    help="metric key(s) in num; default: the quant-damage set that exists")
    a = ap.parse_args()
    summ = json.loads(Path(a.summary).read_text())
    defaults = [("q4_k_m_kl_mean", "Q4_K_M mean KL divergence vs its own f16 (nats)", True),
                ("q4_k_m_rel_dppl", "Q4_K_M relative perplexity damage", False),
                ("q8_0_rel_dppl", "Q8_0 relative perplexity damage", False),
                ("lora_r16_capture", "bounded LoRA r16 task capture", False)]
    wanted = a.metric or [k for k, _, _ in defaults
                          if any(k in v.get("num", {}) for v in summ.values())]
    titles = {k: t for k, t, _ in defaults}
    logs = {k: lg for k, _, lg in defaults}
    blocks, total_y = [], PAD + 60 * len([w for w in wanted if w not in titles])
    y = 20
    for key in wanted:
        items = [(n, v["num"][key]) for n, v in sorted(summ.items(), key=lambda kv: kv[0])
                 if key in v.get("num", {}) and isinstance(v["num"][key], (int, float))]
        svg, h = panel(titles.get(key, key), key, items, logs.get(key, True))
        blocks.append(f'<g transform="translate(0,{y})">{svg}</g>')
        y += h + 26
    legend = "".join(
        f'<rect x="{16 + i * 130}" y="{y}" width="11" height="11" fill="{COLOR[r]}"/>'
        f'<text x="{32 + i * 130}" y="{y + 10}" font-size="11">{r}</text>'
        for i, r in enumerate(["reference", "stressed", "repaired", "control"]))
    svg_doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{LABEL_W + BAR_MAX + 60}" '
               f'height="{y + 40}" viewBox="0 0 {LABEL_W + BAR_MAX + 60} {y + 40}">'
               f'<rect width="100%" height="100%" fill="#ffffff"/>'
               + "\n".join(blocks) + legend + "</svg>\n")
    Path(a.out).write_text(svg_doc)
    print(f"wrote {a.out} ({len(wanted)} panels, {len(summ)} checkpoints)")


if __name__ == "__main__":
    main()
