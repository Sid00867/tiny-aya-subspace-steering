"""Generate custom Anthropic-editorial HTML+SVG figures (zero matplotlib) and render
high-DPI 2x Retina PNGs + vector SVGs via headless Google Chrome.

Uses natural, plain-English labels (zero internal code tags like exp10-OR or ||Delta h||_2)
while loading all exact numbers directly from disk artifacts (`colab_outputs/exp6..exp10`)
and verified Phase 0 logs (`colab_results/extracted_outputs.txt`).
"""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(r"d:\lexsi_hackathon")
OUT_DATA = ROOT / "colab_outputs"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

# Warm Anthropic-inspired editorial palette
PAL = {
    "bg": "#FAF8F4",
    "card": "#FFFFFF",
    "border": "#E4DFD5",
    "ink": "#1D1C19",
    "muted": "#66625B",
    "grid": "#ECE8DF",
    "sv": "#D97757",         # Terracotta — Single Vector / Additive
    "sv_soft": "#FBECE6",
    "pca": "#2E6F95",        # Slate Blue — Subspace / Projection
    "pca_soft": "#EAF3F8",
    "sup": "#3D8B68",        # Forest Sage — Supervised / Matched / Gain
    "sup_soft": "#ECF6F1",
    "warn": "#B83B2A",       # Muted Crimson — Regression / Churn
    "warn_soft": "#FDF0EE",
    "amber": "#C07A28",      # Ochre — Caution / Alternative
    "base": "#8A857C",       # Warm Gray — Unsteered Baseline
}


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def wrap_html_doc(svg_markup: str, width: int, height: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: {width}px; height: {height}px;
    background: {PAL['bg']};
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }}
</style>
</head>
<body>
{svg_markup}
</body>
</html>"""


def render_svg_figure(stem: str, svg_body: str, width: int, height: int) -> tuple[Path, Path]:
    svg_full = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="font-family: Segoe UI, -apple-system, Arial, sans-serif;">\n'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAL["bg"]}"/>\n'
        f'{svg_body}\n</svg>'
    )
    svg_path = FIG_DIR / f"{stem}.svg"
    html_path = FIG_DIR / f"_{stem}.html"
    png_path = FIG_DIR / f"{stem}.png"
    svg_path.write_text(svg_full, encoding="utf-8")
    html_path.write_text(wrap_html_doc(svg_full, width, height), encoding="utf-8")
    url = html_path.resolve().as_uri()
    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size={width},{height}",
        f"--screenshot={png_path}",
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    html_path.unlink(missing_ok=True)
    if res.returncode != 0 or not png_path.exists():
        raise RuntimeError(f"Chrome render failed for {stem}: {res.stderr}\n{res.stdout}")
    print(f"  Rendered {png_path.name} ({width*2}x{height*2} px) + {svg_path.name}")
    return svg_path, png_path


def card(x: float, y: float, w: float, h: float, rx: int = 12, fill: str = PAL["card"], stroke: str = PAL["border"]) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'


def pill(x: float, y: float, text: str, fg: str, bg: str, w: float | None = None) -> str:
    tw = w or (len(text) * 7.1 + 18)
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" height="22" rx="11" fill="{bg}"/>'
        f'<text x="{x + tw/2:.1f}" y="{y + 15:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" fill="{fg}">{esc(text)}</text>'
    )


def header_block(title: str, subtitle: str, badge: str, W: int = 1200) -> str:
    return (
        pill(36, 22, badge, PAL["sv"], PAL["sv_soft"])
        + f'<text x="36" y="70" font-size="23" font-weight="700" fill="{PAL["ink"]}" letter-spacing="-0.3">{esc(title)}</text>'
        + f'<text x="36" y="94" font-size="13.5" fill="{PAL["muted"]}">{esc(subtitle)}</text>'
    )


# ==============================================================================
# FIGURE 1: How One-Dimensional Are Alignment Behaviors? + The Unmatched Illusion
# ==============================================================================
def build_fig1():
    W, H = 1200, 620
    rows = [
        ("Moralizing Persona (5 traits)", 95.3, 97.2),
        ("False Refusals (XSTest & OR-Bench)", 85.0, 91.2),
        ("Everyday Ethics (5 moral frameworks)", 74.7, 88.9),
        ("Instruction Hierarchy (System vs. User)", 70.3, 92.3),
        ("False Refusals (Balanced Dual Split)", 62.3, 90.1),
        ("Cross-Lingual Sycophancy (4 languages)", 61.2, 75.2),
        ("Political Sycophancy (Anthropic Quiz)", 48.8, 75.4),
        ("Conflicting Context & TruthfulQA", 30.4, 55.5),
        ("Social Stereotypes (9 BBQ categories)", 29.5, 69.4),
        ("Conflicting Context (Balanced Dual Split)", 29.5, 56.7),
    ]
    out = [
        header_block(
            "Some Behaviors Fit a Single Direction, Others Span a Subspace — Yet Geometry Alone Doesn't Pick the Winner",
            "Left: Share of activation variance captured by 1 direction vs. 5 directions in Tiny-Aya-Global. Right: Matching push strength flips naive winners.",
            "FIGURE 1 · CONCEPT GEOMETRY & THE PUSH-SIZE ILLUSION",
            W,
        )
    ]
    # Left Card: Spectral Rank
    lx, ly, lw, lh = 36, 114, 640, 474
    out.append(card(lx, ly, lw, lh))
    out.append(f'<text x="{lx+22}" y="{ly+28}" font-size="15" font-weight="700" fill="{PAL["ink"]}">A. How Much Variance Lives in 1 Direction vs. 5 Directions?</text>')
    out.append(f'<text x="{lx+22}" y="{ly+47}" font-size="12" fill="{PAL["muted"]}">Filled terracotta dot = 1st Direction (%)  |  Blue ring = Top 5 Directions Combined (%)</text>')

    ax_l, ax_r = lx + 258, lx + lw - 34
    aw = ax_r - ax_l
    y0 = ly + 82
    row_h = 37
    for pct in (0, 25, 50, 75, 100):
        px = ax_l + (pct / 100.0) * aw
        out.append(f'<line x1="{px:.1f}" y1="{ly+66}" x2="{px:.1f}" y2="{y0 + len(rows)*row_h - 8}" stroke="{PAL["grid"]}" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{px:.1f}" y="{y0 + len(rows)*row_h + 10}" text-anchor="middle" font-size="11" fill="{PAL["muted"]}">{pct}%</text>')

    x50 = ax_l + 0.50 * aw
    out.append(f'<line x1="{x50:.1f}" y1="{ly+64}" x2="{x50:.1f}" y2="{y0 + len(rows)*row_h - 6}" stroke="{PAL["amber"]}" stroke-width="1.2" stroke-dasharray="5,4" opacity="0.65"/>')

    for i, (label, pc1, top5) in enumerate(rows):
        cy = y0 + i * row_h + 10
        if i % 2 == 0:
            out.append(f'<rect x="{lx+12}" y="{cy-16}" width="{lw-24}" height="32" rx="6" fill="#FAF8F4" opacity="0.8"/>')
        out.append(f'<text x="{lx+22}" y="{cy+4}" font-size="12" font-weight="600" fill="{PAL["ink"]}">{esc(label)}</text>')
        x1 = ax_l + (pc1 / 100.0) * aw
        x5 = ax_l + (top5 / 100.0) * aw
        out.append(f'<line x1="{ax_l}" y1="{cy}" x2="{x5:.1f}" y2="{cy}" stroke="{PAL["grid"]}" stroke-width="5" stroke-linecap="round"/>')
        out.append(f'<line x1="{x1:.1f}" y1="{cy}" x2="{x5:.1f}" y2="{cy}" stroke="{PAL["pca"]}" stroke-width="4" opacity="0.45"/>')
        out.append(f'<circle cx="{x5:.1f}" cy="{cy}" r="6" fill="#FFFFFF" stroke="{PAL["pca"]}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{x1:.1f}" cy="{cy}" r="6.5" fill="{PAL["sv"]}"/>')
        out.append(f'<text x="{x1-10:.1f}" y="{cy+4}" text-anchor="end" font-size="10.5" font-weight="700" fill="{PAL["sv"]}">{pc1:.1f}%</text>')
        out.append(f'<text x="{x5+10:.1f}" y="{cy+4}" text-anchor="start" font-size="10.5" font-weight="600" fill="{PAL["pca"]}">{top5:.1f}%</text>')

    # Right Card: Why Naive Sweeps Mislead -> Matched Push Size Reality
    rx, ry, rw, rh = 696, 114, 468, 474
    out.append(card(rx, ry, rw, rh))
    out.append(f'<text x="{rx+22}" y="{ry+28}" font-size="15" font-weight="700" fill="{PAL["ink"]}">B. What Happens When We Equalize Push Strength?</text>')
    out.append(f'<text x="{rx+22}" y="{ry+47}" font-size="11.5" fill="{PAL["muted"]}">Matching how hard each method pushes activations reverses naive results.</text>')

    comparisons = [
        ("Political Sycophancy (Unmatched Push Size)", "Single Vector", 49.0, "9-Vector Subspace", 72.0, "Adding 9 vectors at once pushed activations much harder (+23 pp)", PAL["amber"]),
        ("False Refusals · Global Model (Matched Push)", "Single Vector", 53.67, "5D Subspace", 53.00, "PCA's apparent lead disappears once push size is equalized", PAL["base"]),
        ("False Refusals · Regional Fire Model (Matched)", "Single Vector", 70.00, "5D Subspace", 60.00, "Single vector beats 5D subspace by +10.0 pp at equal push", PAL["sv"]),
        ("Instruction Hierarchy (Before Coherence Breaks)", "Single Vector", 43.93, "5D Subspace", 32.86, "5D subspace loses fluency before reaching effective push size", PAL["sv"]),
        ("Conflicting Context & TruthfulQA (Matched)", "Single Vector", 66.79, "5D Subspace", 58.57, "Single vector beats 5D subspace by +8.2 pp (statistically significant)", PAL["sv"]),
    ]
    cy_start = ry + 64
    card_step = 80
    bar_w = rw - 236
    bx = rx + 158
    for idx, (title_c, sv_lbl, sv_v, pca_lbl, pca_v, note, note_c) in enumerate(comparisons):
        by = cy_start + idx * card_step
        out.append(f'<rect x="{rx+16}" y="{by}" width="{rw-32}" height="72" rx="8" fill="#FAF8F4" stroke="{PAL["grid"]}"/>')
        out.append(f'<text x="{rx+26}" y="{by+17}" font-size="11.8" font-weight="700" fill="{PAL["ink"]}">{esc(title_c)}</text>')
        ysv = by + 30
        ypc = by + 44
        w_sv = (sv_v / 100.0) * bar_w
        w_pc = (pca_v / 100.0) * bar_w
        out.append(f'<text x="{rx+26}" y="{ysv+5}" font-size="10.5" font-weight="600" fill="{PAL["sv"]}">{esc(sv_lbl)}</text>')
        out.append(f'<rect x="{bx}" y="{ysv-4}" width="{w_sv:.1f}" height="8.5" rx="4" fill="{PAL["sv"]}"/>')
        out.append(f'<text x="{bx + w_sv + 6:.1f}" y="{ysv+4}" font-size="10.8" font-weight="700" fill="{PAL["sv"]}">{sv_v:.1f}%</text>')
        out.append(f'<text x="{rx+26}" y="{ypc+5}" font-size="10.5" font-weight="600" fill="{PAL["pca"]}">{esc(pca_lbl)}</text>')
        out.append(f'<rect x="{bx}" y="{ypc-4}" width="{w_pc:.1f}" height="8.5" rx="4" fill="{PAL["pca"]}"/>')
        out.append(f'<text x="{bx + w_pc + 6:.1f}" y="{ypc+4}" font-size="10.8" font-weight="700" fill="{PAL["pca"]}">{pca_v:.1f}%</text>')
        out.append(f'<text x="{rx+26}" y="{by+63}" font-size="10.2" font-weight="600" fill="{note_c}">↳ {esc(note)}</text>')

    out.append(f'<text x="36" y="{H-12}" font-size="11" fill="{PAL["muted"]}">Measured across all 36 layers of Cohere Tiny-Aya-Global, Fire, and Water (3.35B parameters).</text>')
    render_svg_figure("fig1_manifold_spectrum_overview", "\n".join(out), W, H)


# ==============================================================================
# FIGURE 2: Zoomed Intervention Strength Curves
# ==============================================================================
def _line_path(pts: list[tuple[float, float]]) -> str:
    return " ".join(("M" if i == 0 else "L") + f" {x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))


def build_fig2():
    W, H = 1200, 520
    out = [
        header_block(
            "Accuracy Across Intervention Strengths: Comparing Methods at Equal Activation Shift",
            "Zoomed into the usable steering range where model fluency stays intact. Hollow dashed circles mark where output coherence breaks down.",
            "FIGURE 2 · ACCURACY VS. INTERVENTION STRENGTH",
            W,
        )
    ]

    panels = [
        {
            "title": "A. False Refusals (300 test prompts)",
            "sub": "1st Direction Share: 85% · Unsteered: 47.7%",
            "csv": OUT_DATA / "exp6" / "evaluation_and_judge_logs" / "full_alpha_sweep_results.csv",
            "y_col": "Compliance_Overall (%)",
            "x_max": 12.0,
            "y_min": 42.0,
            "y_max": 60.0,
            "callout_pos": "top",
            "configs": [
                ("PCA Subspace (k=5) [Win, Proj]", "5D Subspace (Erase)", PAL["pca"]),
                ("SV Diff-of-Means [Win, Proj]", "Single Vector (Erase)", PAL["sv"]),
                ("SV Diff-of-Means [Win, Add]", "Single Vector (Nudge)", PAL["amber"]),
            ],
            "matched_pt": "SV [L2-Norm-Matched to Best PCA, Proj]",
            "callout": ("At Equal Strength (Shift ≈ 4.9)", "Single Vector 53.7% vs. Subspace 53.0% (Tie)"),
        },
        {
            "title": "B. Instruction Hierarchy (280 prompts)",
            "sub": "1st Direction Share: 70% · Unsteered: 30.7%",
            "csv": OUT_DATA / "exp7" / "evaluation_and_judge_logs" / "full_alpha_sweep_results.csv",
            "y_col": "Joint_IH_Compliance (%)",
            "x_max": 6.5,
            "y_min": 27.0,
            "y_max": 58.0,
            "callout_pos": "top",
            "configs": [
                ("PCA Subspace (k=5) [Win, Proj]", "5D Subspace (Erase)", PAL["pca"]),
                ("SV Diff-of-Means [Win, Proj]", "Single Vector (Erase)", PAL["sv"]),
                ("SV Diff-of-Means [Win, Add]", "Single Vector (Nudge)", PAL["amber"]),
            ],
            "matched_pt": "SV [L2-Norm-Matched to Best PCA, Proj]",
            "callout": ("Peak Usable Strength (Shift = 2.3)", "Single Vector hits 43.9%; Subspace loses fluency"),
        },
        {
            "title": "C. Conflicting Context (280 prompts)",
            "sub": "1st Direction Share: 30% · Unsteered: 62.9%",
            "csv": OUT_DATA / "exp8" / "evaluation_and_judge_logs" / "full_alpha_sweep_results.csv",
            "y_col": "Epistemic_Overall (%)",
            "x_max": 3.3,
            "y_min": 48.0,
            "y_max": 77.0,
            "callout_pos": "bottom_left",
            "configs": [
                ("PCA Subspace (k=5) [Win, Proj]", "5D Subspace (Erase)", PAL["pca"]),
                ("SV Diff-of-Means [Win, Proj]", "Single Vector (Erase)", PAL["base"]),
                ("SV Diff-of-Means [Win, Add]", "Single Vector (Nudge)", PAL["sv"]),
            ],
            "matched_pt": "SV [L2-Norm-Matched to Best PCA, Add]",
            "callout": ("At Equal Strength (Shift = 1.1)", "Additive Vector beats Subspace by +8.2 pp"),
        },
    ]

    pw, ph = 364, 368
    gap = 18
    for idx, p in enumerate(panels):
        cx = 36 + idx * (pw + gap)
        cy = 114
        out.append(card(cx, cy, pw, ph))
        out.append(f'<text x="{cx+18}" y="{cy+26}" font-size="14" font-weight="700" fill="{PAL["ink"]}">{esc(p["title"])}</text>')
        out.append(f'<text x="{cx+18}" y="{cy+44}" font-size="11.5" fill="{PAL["muted"]}">{esc(p["sub"])}</text>')

        px0, py0, plot_w, plot_h = cx + 52, cy + 68, pw - 74, ph - 122
        df = pd.read_csv(p["csv"])

        def sx(val):
            return px0 + max(0.0, min(1.0, val / p["x_max"])) * plot_w

        def sy(val):
            return py0 + plot_h - max(0.0, min(1.0, (val - p["y_min"]) / (p["y_max"] - p["y_min"]))) * plot_h

        for step_i in range(5):
            yv = p["y_min"] + step_i * (p["y_max"] - p["y_min"]) / 4.0
            yy = sy(yv)
            out.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0+plot_w}" y2="{yy:.1f}" stroke="{PAL["grid"]}" stroke-dasharray="3,3"/>')
            out.append(f'<text x="{px0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="{PAL["muted"]}">{yv:.0f}%</text>')
        for step_i in range(5):
            xv = step_i * p["x_max"] / 4.0
            xx = sx(xv)
            out.append(f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py0+plot_h}" stroke="{PAL["grid"]}" stroke-dasharray="3,3"/>')
            out.append(f'<text x="{xx:.1f}" y="{py0+plot_h+16}" text-anchor="middle" font-size="10.5" fill="{PAL["muted"]}">{xv:.1f}</text>')

        out.append(f'<line x1="{px0}" y1="{py0+plot_h}" x2="{px0+plot_w}" y2="{py0+plot_h}" stroke="{PAL["muted"]}" stroke-width="1.2"/>')
        out.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0+plot_h}" stroke="{PAL["muted"]}" stroke-width="1.2"/>')
        out.append(f'<text x="{px0+plot_w/2:.1f}" y="{py0+plot_h+34}" text-anchor="middle" font-size="11" font-weight="600" fill="{PAL["muted"]}">Intervention Strength (Activation Shift)</text>')

        base_y = float(df.iloc[0][p["y_col"]])
        by = sy(base_y)
        out.append(f'<line x1="{px0}" y1="{by:.1f}" x2="{px0+plot_w}" y2="{by:.1f}" stroke="{PAL["base"]}" stroke-width="1.2" stroke-dasharray="4,4"/>')

        for cfg_name, _, col in p["configs"]:
            sub = df[(df["Config"] == cfg_name) & (df["Mean_Delta_L2"] <= p["x_max"])].sort_values("Mean_Delta_L2")
            if sub.empty:
                continue
            pts = [(sx(float(r["Mean_Delta_L2"])), sy(float(r[p["y_col"]]))) for _, r in sub.iterrows()]
            out.append(f'<path d="{_line_path(pts)}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
            for (_, r), (mx, my) in zip(sub.iterrows(), pts):
                valid = str(r["Gate_Valid"]).lower() == "true"
                if valid:
                    out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4.2" fill="{col}" stroke="#FFFFFF" stroke-width="1.2"/>')
                else:
                    out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4.2" fill="#FFFFFF" stroke="{col}" stroke-width="2.0" stroke-dasharray="2,1"/>')

        mrow = df[df["Config"] == p["matched_pt"]]
        if not mrow.empty:
            mx = sx(float(mrow.iloc[0]["Mean_Delta_L2"]))
            my = sy(float(mrow.iloc[0][p["y_col"]]))
            out.append(f'<polygon points="{mx:.1f},{my-7:.1f} {mx+7:.1f},{my:.1f} {mx:.1f},{my+7:.1f} {mx-7:.1f},{my:.1f}" fill="{PAL["sup"]}" stroke="#FFFFFF" stroke-width="1.5"/>')

        c_title, c_desc = p["callout"]
        bx = px0 + 8
        by_box = (py0 + plot_h - 42) if p.get("callout_pos") == "bottom_left" else (py0 + 6)
        out.append(f'<rect x="{bx}" y="{by_box}" width="{plot_w-16}" height="34" rx="6" fill="{PAL["bg"]}" stroke="{PAL["border"]}"/>')
        out.append(f'<text x="{bx+10}" y="{by_box+14}" font-size="10.5" font-weight="700" fill="{PAL["ink"]}">{esc(c_title)}</text>')
        out.append(f'<text x="{bx+10}" y="{by_box+27}" font-size="10.5" fill="{PAL["muted"]}">{esc(c_desc)}</text>')

    out.append(card(36, 490, W - 72, 26, rx=8, fill="#FFFFFF", stroke=PAL["border"]))
    out.append(f'<circle cx="56" cy="503" r="5" fill="{PAL["sv"]}"/><text x="66" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">Best Single Vector</text>')
    out.append(f'<circle cx="215" cy="503" r="5" fill="{PAL["pca"]}"/><text x="225" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">5D PCA Subspace (Erase)</text>')
    out.append(f'<circle cx="415" cy="503" r="5" fill="{PAL["amber"]}"/><text x="425" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">Alternative Single Vector</text>')
    out.append(f'<polygon points="600,496 607,503 600,510 593,503" fill="{PAL["sup"]}"/><text x="613" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">Equal-Strength Single Vector</text>')
    out.append(f'<circle cx="808" cy="503" r="4.5" fill="{PAL["ink"]}"/><text x="818" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">Fluent</text>')
    out.append(f'<circle cx="882" cy="503" r="4.5" fill="#FFFFFF" stroke="{PAL["ink"]}" stroke-width="1.8" stroke-dasharray="2,1"/><text x="892" y="507" font-size="11" font-weight="600" fill="{PAL["ink"]}">Fluency Breakdown</text>')

    render_svg_figure("fig2_norm_matched_pareto_regimes", "\n".join(out), W, H)


# ==============================================================================
# FIGURE 3: Erasing vs. Nudging — The Double Dissociation
# ==============================================================================
def build_fig3():
    W, H = 1200, 620
    df = pd.read_csv(OUT_DATA / "exp10" / "evaluation_and_judge_logs" / "operator_rank_factorial_results.csv")
    df_or = df[df["Task"] == "OverRefusal"].copy()
    df_fe = df[df["Task"] == "Epistemic"].copy()
    or_base = float(df_or[df_or["Family"] == "base"].iloc[0]["Overall_Acc (%)"])
    fe_base = float(df_fe[df_fe["Family"] == "base"].iloc[0]["Overall_Acc (%)"])

    out = [
        header_block(
            "Why No Single Method Wins Everything: Erasing a Reflex vs. Adding a Signal",
            "When we hold push strength equal, erasing activations (Projection) fixes false refusals (+23 pp), while adding a vector fixes context reading (+5 pp).",
            "FIGURE 3 · ERASING (PROJECTION) VS. NUDGING (ADDITION)",
            W,
        )
    ]

    specs = [
        ("OverRefusal", or_base, "A. Fixing False Refusals (Unsteered Accuracy: 55.3%)",
         "Erasing the refusal reflex (blue bars) unlocks +23.0 pp without hurting safety.", 36, 114, 558, 342, -10.0, 26.0),
        ("Epistemic", fe_base, "B. Reading Conflicting Contexts (Unsteered Accuracy: 61.8%)",
         "Nudging activations (terracotta) helps (+5.0 pp); erasing (red) hurts every time.", 606, 114, 558, 342, -12.0, 10.0),
    ]
    method_pairs = [
        ("Centered PCA (1D)", "unsup_pca", 1),
        ("Mean-Diff Vector (1D)", "sv_dom", 1),
        ("Mean + PCA (2D)", "anch_pca", 2),
        ("Category SVD (2D)", "sup_svd", 2),
        ("Category SVD (3D)", "sup_svd", 3),
        ("Mean + PCA (5D)", "anch_pca", 5),
        ("Category SVD (5D)", "sup_svd", 5),
        ("Centered PCA (8D)", "unsup_pca", 8),
    ]

    for task, base_acc, t_head, t_sub, px, py, pw, ph, dmin, dmax in specs:
        out.append(card(px, py, pw, ph))
        out.append(f'<text x="{px+18}" y="{py+26}" font-size="14" font-weight="700" fill="{PAL["ink"]}">{esc(t_head)}</text>')
        out.append(f'<text x="{px+18}" y="{py+44}" font-size="11.5" fill="{PAL["muted"]}">{esc(t_sub)}</text>')
        sub_t = df_or if task == "OverRefusal" else df_fe

        ax_x0, ax_x1 = px + 154, px + pw - 26
        ax_w = ax_x1 - ax_x0

        def dx(delta):
            return ax_x0 + max(0.0, min(1.0, (delta - dmin) / (dmax - dmin))) * ax_w

        x_zero = dx(0.0)
        y_top = py + 72
        row_h = 31

        for tv in range(int(dmin), int(dmax) + 1, 5):
            if dmin <= tv <= dmax:
                tx = dx(tv)
                out.append(f'<line x1="{tx:.1f}" y1="{y_top-8}" x2="{tx:.1f}" y2="{y_top+len(method_pairs)*row_h-4}" stroke="{PAL["ink"] if tv==0 else PAL["grid"]}" stroke-width="{1.5 if tv==0 else 1.0}" stroke-dasharray="{"" if tv==0 else "3,3"}"/>')
                out.append(f'<text x="{tx:.1f}" y="{y_top+len(method_pairs)*row_h+12}" text-anchor="middle" font-size="10.5" fill="{PAL["muted"]}">{"0" if tv==0 else f"{tv:+d}"} pp</text>')

        for i, (mlbl, fam, rk) in enumerate(method_pairs):
            ry = y_top + i * row_h
            out.append(f'<text x="{px+18}" y="{ry+14}" font-size="11.5" font-weight="600" fill="{PAL["ink"]}">{esc(mlbl)}</text>')
            r_add = sub_t[(sub_t["Family"] == fam) & (sub_t["Rank_k"] == rk) & (sub_t["Operator"] == "additive")]
            r_prj = sub_t[(sub_t["Family"] == fam) & (sub_t["Rank_k"] == rk) & (sub_t["Operator"] == "projection")]
            if not r_add.empty:
                d_a = float(r_add.iloc[0]["Overall_Acc (%)"]) - base_acc
                x_a = dx(d_a)
                bx0, bw_a = min(x_zero, x_a), max(2.0, abs(x_a - x_zero))
                col_a = PAL["sv"] if d_a >= 0 else PAL["warn"]
                out.append(f'<rect x="{bx0:.1f}" y="{ry+2}" width="{bw_a:.1f}" height="8.5" rx="3" fill="{col_a}" opacity="0.9"/>')
                tx_a = x_a + (4 if d_a >= 0 else -4)
                anc_a = "start" if d_a >= 0 else "end"
                out.append(f'<text x="{tx_a:.1f}" y="{ry+9}" text-anchor="{anc_a}" font-size="9.5" font-weight="700" fill="{col_a}">{d_a:+.1f}</text>')
            if not r_prj.empty:
                d_p = float(r_prj.iloc[0]["Overall_Acc (%)"]) - base_acc
                x_p = dx(d_p)
                bx0, bw_p = min(x_zero, x_p), max(2.0, abs(x_p - x_zero))
                col_p = PAL["pca"] if d_p >= 0 else PAL["warn"]
                out.append(f'<rect x="{bx0:.1f}" y="{ry+13}" width="{bw_p:.1f}" height="8.5" rx="3" fill="{col_p}"/>')
                tx_p = x_p + (4 if d_p >= 0 else -4)
                anc_p = "start" if d_p >= 0 else "end"
                out.append(f'<text x="{tx_p:.1f}" y="{ry+20}" text-anchor="{anc_p}" font-size="9.5" font-weight="700" fill="{col_p}">{d_p:+.1f}</text>')

    out.append(card(36, 470, W - 72, 128))
    out.append(f'<text x="56" y="496" font-size="14" font-weight="700" fill="{PAL["ink"]}">C. Adding Too Many Unsupervised Directions Dilutes a Fixed Steering Budget</text>')
    ks = [1, 2, 3, 5, 8]
    kx0, ky0, kw_p, kh_p = 66, 514, 340, 62
    out.append(f'<line x1="{kx0}" y1="{ky0+kh_p}" x2="{kx0+kw_p}" y2="{ky0+kh_p}" stroke="{PAL["muted"]}" stroke-width="1"/>')
    pts_or_unsup, pts_or_sup = [], []
    for idx_k, k in enumerate(ks):
        xk = kx0 + idx_k * (kw_p / 4.0)
        out.append(f'<text x="{xk:.1f}" y="{ky0+kh_p+14}" text-anchor="middle" font-size="10.5" fill="{PAL["muted"]}">{k}D</text>')
        ru = df_or[(df_or["Family"] == "unsup_pca") & (df_or["Rank_k"] == k) & (df_or["Operator"] == "projection")]
        if not ru.empty:
            v = float(ru.iloc[0]["Overall_Acc (%)"])
            yk = ky0 + kh_p - ((v - 50.0) / 32.0) * kh_p
            pts_or_unsup.append((xk, yk, v))
        rs = df_or[(df_or["Family"] == "sup_svd") & (df_or["Rank_k"] == k) & (df_or["Operator"] == "projection")]
        if not rs.empty:
            v = float(rs.iloc[0]["Overall_Acc (%)"])
            yk = ky0 + kh_p - ((v - 50.0) / 32.0) * kh_p
            pts_or_sup.append((xk, yk, v))
    out.append(f'<path d="{_line_path([(x, y) for x, y, _ in pts_or_unsup])}" fill="none" stroke="{PAL["pca"]}" stroke-width="2.2"/>')
    for idx_u, (x, y, v) in enumerate(pts_or_unsup):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PAL["pca"]}"/>')
        dy_lbl = 15 if idx_u == 3 else -8
        out.append(f'<text x="{x:.1f}" y="{y + dy_lbl:.1f}" text-anchor="middle" font-size="9.5" font-weight="700" fill="{PAL["pca"]}">{v:.1f}%</text>')
    out.append(f'<path d="{_line_path([(x, y) for x, y, _ in pts_or_sup])}" fill="none" stroke="{PAL["sup"]}" stroke-width="2.2" stroke-dasharray="4,2"/>')
    for x, y, v in pts_or_sup:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{PAL["sup"]}"/>')
    out.append(pill(424, 502, "Top bar = Nudge (Addition)", PAL["sv"], PAL["sv_soft"], 168))
    out.append(pill(600, 502, "Bottom bar = Erase (Projection: blue=gain, red=loss)", PAL["pca"], PAL["pca_soft"], 325))
    out.append(f'<text x="424" y="542" font-size="11.5" font-weight="600" fill="{PAL["pca"]}">— Unsupervised PCA peaks at 1D (78.2%), then loses focus as extra axes split the budget (60.6% at 8D)</text>')
    out.append(f'<text x="424" y="560" font-size="11.5" font-weight="600" fill="{PAL["sup"]}">-- Category-Supervised SVD keeps multi-axis steering sharp at 3D–5D (76.5% / 72.9%)</text>')
    out.append(f'<text x="424" y="578" font-size="11.5" font-weight="600" fill="{PAL["sv"]}">■ Conflicting Contexts: Nudging beats Erasing by +11.4 pp (26 prompts fixed vs. 1 broken)</text>')

    render_svg_figure("fig3_exp10_double_dissociation", "\n".join(out), W, H)


# ==============================================================================
# FIGURE 4: Regional Model Transfer & Hidden Answer Churn
# ==============================================================================
def build_fig4():
    W, H = 1200, 540
    out = [
        header_block(
            "Regional Models Need Native Vectors — And Tied Scores Can Hide Heavy Answer Flipping",
            "Left: Reusing Global vectors zero-shot on regional models (Fire & Water) stalls, while native single vectors jump ahead. Right: Item-level flips.",
            "FIGURE 4 · REGIONAL TRANSFER & HIDDEN ANSWER CHURN",
            W,
        )
    ]

    lx, ly, lw, lh = 36, 114, 620, 396
    out.append(card(lx, ly, lw, lh))
    out.append(f'<text x="{lx+20}" y="{ly+28}" font-size="14.5" font-weight="700" fill="{PAL["ink"]}">A. Global vs. Regional Models (Tiny-Aya Global, Fire, Water)</text>')
    out.append(f'<text x="{lx+20}" y="{ly+46}" font-size="11.5" fill="{PAL["muted"]}">Task accuracy (%) when transferring Global vectors zero-shot vs. extracting native vectors.</text>')

    mm_rows = [
        ("False Refusals · Global", 47.67, None, 53.67, 53.00),
        ("False Refusals · Fire", 55.00, 52.67, 70.00, 60.00),
        ("False Refusals · Water", 46.67, 47.00, 61.67, 51.00),
        ("Conflicting Context · Global", 62.86, None, 66.79, 58.57),
        ("Conflicting Context · Fire", 63.57, 64.64, 67.86, 61.79),
        ("Conflicting Context · Water", 63.57, 63.93, 68.57, 60.00),
    ]
    ax0, ax1 = lx + 218, lx + lw - 28
    aw = ax1 - ax0
    xmin, xmax = 40.0, 74.0

    def mx(v):
        return ax0 + ((v - xmin) / (xmax - xmin)) * aw

    for tv in (45, 50, 55, 60, 65, 70):
        tx = mx(tv)
        out.append(f'<line x1="{tx:.1f}" y1="{ly+72}" x2="{tx:.1f}" y2="{ly+344}" stroke="{PAL["grid"]}" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{tx:.1f}" y="{ly+362}" text-anchor="middle" font-size="10.5" fill="{PAL["muted"]}">{tv}%</text>')

    for i, (lbl, v_base, v_zs, v_nsv, v_npca) in enumerate(mm_rows):
        ry = ly + 96 + i * 43
        if i == 3:
            out.append(f'<line x1="{lx+20}" y1="{ry-22}" x2="{lx+lw-20}" y2="{ry-22}" stroke="{PAL["border"]}"/>')
        out.append(f'<text x="{lx+20}" y="{ry+4}" font-size="12" font-weight="600" fill="{PAL["ink"]}">{esc(lbl)}</text>')
        lo = min(x for x in (v_base, v_zs, v_nsv, v_npca) if x is not None)
        hi = max(x for x in (v_base, v_zs, v_nsv, v_npca) if x is not None)
        out.append(f'<line x1="{mx(lo):.1f}" y1="{ry}" x2="{mx(hi):.1f}" y2="{ry}" stroke="{PAL["grid"]}" stroke-width="5" stroke-linecap="round"/>')
        out.append(f'<circle cx="{mx(v_npca):.1f}" cy="{ry}" r="6" fill="{PAL["pca"]}"/>')
        if v_zs is not None:
            out.append(f'<circle cx="{mx(v_zs):.1f}" cy="{ry}" r="5.5" fill="#FFFFFF" stroke="{PAL["amber"]}" stroke-width="2.2"/>')
        out.append(f'<rect x="{mx(v_base)-5:.1f}" y="{ry-5}" width="10" height="10" rx="2" fill="{PAL["base"]}"/>')
        out.append(f'<circle cx="{mx(v_nsv):.1f}" cy="{ry}" r="7" fill="{PAL["sv"]}"/>')
        out.append(f'<text x="{mx(v_nsv)+11:.1f}" y="{ry+4}" font-size="11" font-weight="700" fill="{PAL["sv"]}">{v_nsv:.1f}%</text>')

    out.append(f'<rect x="{lx+36}" y="{ly+376}" width="9" height="9" rx="2" fill="{PAL["base"]}"/><text x="{lx+50}" y="{ly+385}" font-size="11" font-weight="600" fill="{PAL["ink"]}">Unsteered</text>')
    out.append(f'<circle cx="{lx+132}" cy="{ly+381}" r="5" fill="#FFFFFF" stroke="{PAL["amber"]}" stroke-width="2"/><text x="{lx+142}" y="{ly+385}" font-size="11" font-weight="600" fill="{PAL["ink"]}">Zero-Shot Global</text>')
    out.append(f'<circle cx="{lx+268}" cy="{ly+381}" r="5" fill="{PAL["pca"]}"/><text x="{lx+278}" y="{ly+385}" font-size="11" font-weight="600" fill="{PAL["ink"]}">Native 5D Subspace</text>')
    out.append(f'<circle cx="{lx+416}" cy="{ly+381}" r="6" fill="{PAL["sv"]}"/><text x="{lx+427}" y="{ly+385}" font-size="11" font-weight="600" fill="{PAL["ink"]}">Native Single Vector</text>')

    # Right Card: AuditKIT RunDiff Churn
    rx, ry, rw, rh = 674, 114, 490, 396
    out.append(card(rx, ry, rw, rh))
    out.append(f'<text x="{rx+20}" y="{ry+28}" font-size="14.5" font-weight="700" fill="{PAL["ink"]}">B. Prompts Fixed vs. Prompts Broken (AuditKIT RunDiff)</text>')
    out.append(f'<text x="{rx+20}" y="{ry+46}" font-size="11.5" fill="{PAL["muted"]}">Individual test questions flipped right (+) or wrong (−) compared to unsteered model.</text>')

    diff_rows = [
        ("False Refusals · Single Vector", 18, 0, PAL["sv"]),
        ("False Refusals · 5D Subspace", 16, 0, PAL["pca"]),
        ("Conflicting Context · Single Vector", 11, 0, PAL["sv"]),
        ("Conflicting Context · 5D Subspace", 2, 14, PAL["warn"]),
        ("Everyday Ethics · Single Vector", 3, 4, PAL["sv"]),
        ("Everyday Ethics · 5D Subspace", 12, 13, PAL["warn"]),
        ("False Refusals · 1D Centered Erase", 44, 1, PAL["pca"]),
        ("Conflicting Context · Nudge vs Erase", 26, 1, PAL["sv"]),
    ]
    zc = rx + 295
    scale = 3.8
    out.append(f'<line x1="{zc}" y1="{ry+68}" x2="{zc}" y2="{ry+348}" stroke="{PAL["ink"]}" stroke-width="1.4"/>')
    out.append(f'<text x="{zc-12}" y="{ry+364}" text-anchor="end" font-size="10.5" font-weight="600" fill="{PAL["warn"]}">← Prompts Broken (Newly Wrong)</text>')
    out.append(f'<text x="{zc+12}" y="{ry+364}" text-anchor="start" font-size="10.5" font-weight="600" fill="{PAL["sup"]}">Prompts Fixed (Newly Right) →</text>')

    for i, (lbl, n_ok, n_bad, col_ok) in enumerate(diff_rows):
        yy = ly + 84 + i * 33
        out.append(f'<text x="{rx+18}" y="{yy+4}" font-size="11.2" font-weight="600" fill="{PAL["ink"]}">{esc(lbl)}</text>')
        if n_bad > 0:
            bw = n_bad * scale
            out.append(f'<rect x="{zc - bw:.1f}" y="{yy-8}" width="{bw:.1f}" height="15" rx="3" fill="{PAL["warn"]}"/>')
            out.append(f'<text x="{zc - bw - 5:.1f}" y="{yy+4}" text-anchor="end" font-size="10.5" font-weight="700" fill="{PAL["warn"]}">-{n_bad}</text>')
        if n_ok > 0:
            gw = n_ok * scale
            c_use = PAL["sup"] if col_ok != PAL["warn"] else PAL["pca"]
            out.append(f'<rect x="{zc:.1f}" y="{yy-8}" width="{gw:.1f}" height="15" rx="3" fill="{c_use}"/>')
            out.append(f'<text x="{zc + gw + 5:.1f}" y="{yy+4}" font-size="10.5" font-weight="700" fill="{c_use}">+{n_ok}</text>')
        if n_bad == 0:
            out.append(f'<text x="{zc - 6}" y="{yy+4}" text-anchor="end" font-size="10" fill="{PAL["muted"]}">0</text>')

    out.append(f'<text x="{rx+20}" y="{ry+384}" font-size="11" fill="{PAL["amber"]}" font-weight="600">Takeaway: On Ethics, 5D Subspace looked flat (−0.4 pp) but actually flipped 25 answers (+12 / −13).</text>')
    render_svg_figure("fig4_cross_variant_transfer_and_churn", "\n".join(out), W, H)


# ==============================================================================
# FIGURE 5: Verification Pipeline & The Hidden Math Tax (GSM8K vs ARC)
# ==============================================================================
def build_fig5():
    W, H = 1200, 560
    out = [
        header_block(
            "Our 4-Stage Audit Pipeline & The Hidden Reasoning Tax on Multi-Step Math",
            "While multiple-choice science (ARC) holds steady under steering, multi-step word problems (GSM8K) often drop by 7 to 12 percentage points.",
            "FIGURE 5 · PIPELINE ARCHITECTURE & SIDE-EFFECT AUDIT",
            W,
        )
    ]

    stages = [
        ("1. CuratorKIT Hygiene", "Deduplicates prompts & removes\nany train-test overlap (0 leaks)", PAL["sup"], PAL["sup_soft"]),
        ("2. CircuitKIT Steering", "Extracts 36-layer vectors & matches\npush strength layer by layer", PAL["sv"], PAL["sv_soft"]),
        ("3. Fluency & Judge Gate", "Checks output coherence (≥0.90)\n+ calibrated 72B LLM judge", PAL["pca"], PAL["pca_soft"]),
        ("4. AuditKIT Side-Effects", "Tracks flipped items + tests\nGSM8K math & ARC science", PAL["ink"], "#F0EDE6"),
    ]
    sw = 264
    for i, (st_t, st_d, fg, bg) in enumerate(stages):
        sx_i = 36 + i * (sw + 24)
        out.append(card(sx_i, 114, sw, 82, rx=10, fill=bg, stroke=fg if i < 3 else PAL["border"]))
        out.append(f'<text x="{sx_i+16}" y="{140}" font-size="13.5" font-weight="700" fill="{fg}">{esc(st_t)}</text>')
        for li, line in enumerate(st_d.split("\n")):
            out.append(f'<text x="{sx_i+16}" y="{160 + li*16}" font-size="11.2" fill="{PAL["ink"]}">{esc(line)}</text>')
        if i < 3:
            ax = sx_i + sw + 4
            out.append(f'<path d="M {ax} 155 L {ax+14} 155 M {ax+9} 150 L {ax+15} 155 L {ax+9} 160" fill="none" stroke="{PAL["muted"]}" stroke-width="2" stroke-linecap="round"/>')

    bx, by, bw, bh = 36, 214, W - 72, 322
    out.append(card(bx, by, bw, bh))
    out.append(f'<text x="{bx+22}" y="{by+28}" font-size="14.5" font-weight="700" fill="{PAL["ink"]}">Side-Effect Check: Change in Math (GSM8K) and Science (ARC) Accuracy After Steering</text>')
    out.append(f'<text x="{bx+22}" y="{by+47}" font-size="11.5" fill="{PAL["muted"]}">Percentage-point change relative to the unsteered model across 5 experiments.</text>')

    cap_data = [
        ("False Refusals", [
            ("Best 1D", "(Erase)", +12.5, -1.7),
            ("Equal 1D", "(Erase)", +10.0, -1.7),
            ("5D PCA", "(Erase)", +7.5, 0.0),
        ], "Unsteered: Math 52.5% | Sci 73.3%"),
        ("System Hierarchy", [
            ("Best 1D", "(Erase)", -10.0, 0.0),
            ("Equal 1D", "(Erase)", -2.5, -1.7),
            ("5D PCA", "(Erase)", 0.0, -3.3),
        ], "Unsteered: Math 52.5% | Sci 73.3%"),
        ("Conflicting Context", [
            ("Best 1D", "(Nudge)", -7.5, 0.0),
            ("Equal 1D", "(Nudge)", -7.5, +1.7),
            ("5D SVD", "(Erase)", -12.5, 0.0),
        ], "Unsteered: Math 52.5% | Sci 75.0%"),
        ("Everyday Ethics", [
            ("Best 1D", "(Nudge)", 0.0, 0.0),
            ("Equal 1D", "(Nudge)", -7.5, 0.0),
            ("5D SVD", "(Erase)", -2.5, -1.7),
        ], "Unsteered: Math 52.5% | Sci 73.3%"),
        ("Factorial Suite", [
            ("Refusal 1D", "(Nudge)", -11.4, 0.0),
            ("Context 1D", "(Nudge)", -2.9, +6.0),
            ("Context 5D", "(Erase)", -8.6, +4.0),
        ], "Unsteered: Math 54.3% | Sci 74.0%"),
    ]

    col_w = (bw - 44) / 5.0
    zero_y = by + 185
    scale_y = 6.0
    out.append(f'<line x1="{bx+22}" y1="{zero_y}" x2="{bx+bw-22}" y2="{zero_y}" stroke="{PAL["ink"]}" stroke-width="1.2"/>')
    for dp in (-10, -5, 5, 10):
        yy = zero_y - dp * scale_y
        out.append(f'<line x1="{bx+22}" y1="{yy:.1f}" x2="{bx+bw-22}" y2="{yy:.1f}" stroke="{PAL["grid"]}" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{bx+18}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="{PAL["muted"]}">{dp:+d}</text>')

    for gi, (g_title, bars, bnote) in enumerate(cap_data):
        gx = bx + 32 + gi * col_w
        out.append(f'<text x="{gx + col_w/2 - 12:.1f}" y="{by+78}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{PAL["ink"]}">{esc(g_title)}</text>')
        out.append(f'<text x="{gx + col_w/2 - 12:.1f}" y="{by+94}" text-anchor="middle" font-size="10" fill="{PAL["muted"]}">{esc(bnote)}</text>')
        bw_bar = 22
        for bi, (bl1, bl2, dgsm, darc) in enumerate(bars):
            rx_b = gx + 14 + bi * 68
            gh = abs(dgsm) * scale_y
            gy_top = zero_y - gh if dgsm >= 0 else zero_y
            gcol = PAL["sup"] if dgsm >= 0 else PAL["warn"]
            if gh > 0.5:
                out.append(f'<rect x="{rx_b}" y="{gy_top:.1f}" width="{bw_bar}" height="{gh:.1f}" rx="3" fill="{gcol}"/>')
            ty_g = gy_top - 5 if dgsm >= 0 else gy_top + gh + 12
            out.append(f'<text x="{rx_b+bw_bar/2}" y="{ty_g:.1f}" text-anchor="middle" font-size="9.5" font-weight="700" fill="{gcol}">{dgsm:+.1f}</text>')
            ah = abs(darc) * scale_y
            ay_top = zero_y - ah if darc >= 0 else zero_y
            acol = PAL["pca"]
            if ah > 0.5:
                out.append(f'<rect x="{rx_b+bw_bar+3}" y="{ay_top:.1f}" width="{bw_bar-6}" height="{ah:.1f}" rx="3" fill="{acol}" opacity="0.78"/>')
                ty_a = ay_top - 5 if darc >= 0 else ay_top + ah + 11
                out.append(f'<text x="{rx_b+bw_bar+11}" y="{ty_a:.1f}" text-anchor="middle" font-size="8.8" font-weight="700" fill="{acol}">{darc:+.1f}</text>')
            out.append(f'<text x="{rx_b+bw_bar-2}" y="{by+284}" text-anchor="middle" font-size="9.8" font-weight="700" fill="{PAL["ink"]}">{esc(bl1)}</text>')
            out.append(f'<text x="{rx_b+bw_bar-2}" y="{by+297}" text-anchor="middle" font-size="9.3" fill="{PAL["muted"]}">{esc(bl2)}</text>')

    out.append(f'<rect x="{bx+bw-470}" y="{by+33}" width="11" height="11" rx="2" fill="{PAL["sup"]}"/><rect x="{bx+bw-456}" y="{by+33}" width="11" height="11" rx="2" fill="{PAL["warn"]}"/><text x="{bx+bw-439}" y="{by+43}" font-size="10.8" font-weight="600" fill="{PAL["ink"]}">GSM8K Math Change (pp)</text>')
    out.append(f'<rect x="{bx+bw-265}" y="{by+33}" width="9" height="11" rx="2" fill="{PAL["pca"]}"/><text x="{bx+bw-250}" y="{by+43}" font-size="10.8" font-weight="600" fill="{PAL["ink"]}">ARC Science Change (pp)</text>')
    render_svg_figure("fig5_collateral_and_pipeline_architecture", "\n".join(out), W, H)


if __name__ == "__main__":
    print("Rendering plain-English custom SVG + Retina PNG figures...")
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    print("All 5 figures rendered.")
