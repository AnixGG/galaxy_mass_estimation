"""Build English HTML report cards and export them to PNG with Playwright.

Run from any directory: python /path/to/repo/scripts/render_readme.py
Use --html-only to build the HTML and scientific charts without a browser.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "galaxy-mass-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_results import PROJECT_ROOT, summarize

ASSETS = PROJECT_ROOT / "docs/assets"
COLORS = {"base": "#315d91", "wide": "#17a397", "less": "#b47744"}


def scientific_charts():
    """Use actual saved arrays for all plotted values."""
    (ASSETS / "charts").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": "#20324a",
            "axes.labelcolor": "#50627a",
            "xtick.color": "#50627a",
            "ytick.color": "#50627a",
            "axes.edgecolor": "#d8e1ed",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    records = summarize()
    metrics = {(r["model"], r["variant"]): r for r in records}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.87, wspace=0.34)
    x = np.arange(3)
    for offset, variant, color, label in [
        (-0.19, "raw", "#315d91", "Raw maps"),
        (0.19, "normalized", "#17a397", "Normalized maps"),
    ]:
        values = [metrics[(f"fno_{m}", variant)]["mae_dex"] for m in COLORS]
        bars = axes[0].bar(
            x + offset, values, width=0.32, color=color, label=label, zorder=3
        )
        axes[0].bar_label(
            bars, labels=[f"{v:.4f}" for v in values], padding=6, fontsize=10
        )
    axes[0].set_xticks(x, ["FNO base", "FNO wide", "FNO less"])
    axes[0].set_ylim(0, 0.135)
    axes[0].set_yticks([0, 0.04, 0.08, 0.12])
    axes[0].set_ylabel("Mean absolute error [dex]")
    axes[0].set_title(
        "Architecture & normalization", loc="left", fontweight="bold", pad=18
    )
    axes[0].legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13)
    )
    axes[0].grid(axis="y", color="#e9eef5", zorder=0)
    targets = np.load(PROJECT_ROOT / "results/fno_less/raw/targets.npy").reshape(-1)
    preds = np.load(PROJECT_ROOT / "results/fno_less/raw/predictions.npy").reshape(-1)
    limits = [
        min(targets.min(), preds.min()) - 0.04,
        max(targets.max(), preds.max()) + 0.04,
    ]
    axes[1].scatter(
        targets, preds, s=10, alpha=0.28, color="#315d91", edgecolors="none", zorder=2
    )
    axes[1].plot(
        limits, limits, "--", color="#17a397", linewidth=1.6, label="Perfect agreement"
    )
    axes[1].set(
        xlim=limits,
        ylim=limits,
        xlabel="Catalog log10(M500cUncorr)",
        ylabel="Predicted log10(M500cUncorr)",
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("FNO less · raw maps", loc="left", fontweight="bold", pad=18)
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].grid(color="#edf1f6", zorder=0)
    fig.savefig(ASSETS / "charts/performance.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.30, top=0.86, wspace=0.30)
    for model, color in COLORS.items():
        for variant, style in [("raw", "-"), ("normalized", "--")]:
            loss = np.load(PROJECT_ROOT / f"results/fno_{model}/{variant}/losses.npy")
            if (loss <= 0).any():
                raise ValueError("Log-scale training plot requires positive losses")
            axes[0].plot(
                np.arange(1, len(loss) + 1),
                loss,
                style,
                color=color,
                linewidth=1.5,
                label=f"{model} / {variant}",
            )
    axes[0].set(
        xlabel="Epoch", ylabel="Training MSE [dex²]", yscale="log", xlim=(1, 100)
    )
    axes[0].set_title(
        "Learning curves · original split", loc="left", fontweight="bold", pad=18
    )
    axes[0].legend(
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    errors = preds - targets
    axes[1].hist(errors, bins=40, color="#315d91", alpha=0.9)
    axes[1].axvline(
        0, color="#17a397", linestyle="--", linewidth=1.7, label="Zero error"
    )
    axes[1].set(xlabel="Prediction − catalog target [dex]", ylabel="Test samples")
    axes[1].set_title(
        "Residuals · FNO less / raw", loc="left", fontweight="bold", pad=18
    )
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18, zorder=0)
    fig.savefig(ASSETS / "charts/diagnostics.png", dpi=170)
    plt.close(fig)
    (PROJECT_ROOT / "results/metrics.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    return metrics


CSS = """
*{box-sizing:border-box}body{margin:0;background:#e7edf4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#172c45}
.canvas{width:1440px;padding:60px 66px 36px;background:#fff;overflow:hidden}
.eyebrow{font-size:13px;letter-spacing:2.6px;font-weight:700;color:#61778e;text-transform:uppercase}
h1{font-size:48px;letter-spacing:-1.9px;line-height:1.12;margin:22px 0 16px;font-weight:750}
.subtitle{font-size:18px;line-height:1.6;color:#60748a;max-width:1050px;margin:0}
.rule{height:1px;background:#e3eaf2;margin:30px 0 22px}.footer{display:flex;justify-content:space-between;font-size:13px;color:#657a91;line-height:1.5}
.chart{display:block;width:100%;height:auto;margin:12px 0 0}.header-row{display:flex;justify-content:space-between;align-items:flex-end;gap:40px}
.metric{min-width:170px;border-left:2px solid #17a397;padding-left:20px;margin-bottom:3px}.metric strong{display:block;font-size:36px;letter-spacing:-1px}.metric span{display:block;font-size:13px;color:#657a91;line-height:1.5;margin-top:7px}
.dark{background:#11243a;color:#edf5ff;padding-bottom:40px;position:relative}.dark .eyebrow{color:#7cbdc2}.dark h1{font-size:64px;letter-spacing:-2.8px;max-width:930px;margin-top:30px}.dark .subtitle{color:#a4b7cb;max-width:880px;font-size:20px}.dark .rule{background:#2c4056}.dark .footer{color:#8fa8bf}
.bands{display:flex;gap:14px;margin-top:30px}.band{border:1px solid #365169;border-radius:30px;padding:10px 20px;font-size:15px;color:#bdd5e7}.band strong{color:#e8f5fd}
.pipeline{display:grid;grid-template-columns:1fr 30px 1fr 30px 1fr 30px 1fr;align-items:stretch;margin:48px 0 32px}.step{padding:21px 20px;background:#1a3048;border:1px solid #344a60;border-radius:10px;min-height:130px}.step .n{font-size:11px;letter-spacing:1.8px;color:#79c5bf}.step h2{font-size:19px;font-weight:650;letter-spacing:-.3px;margin:13px 0 8px}.step p{font-size:13px;line-height:1.5;color:#a2b8cc;margin:0}.arrow{display:flex;align-items:center;justify-content:center;color:#6e9baf;font-size:24px}.note{font-size:13px;color:#71839a;line-height:1.5;margin:2px 0 0}
"""


def write_html(name, content):
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Galaxy Cluster Mass Estimation · {name}</title><style>{CSS}</style></head><body>{content}</body></html>'
    (ASSETS / f"{name}.html").write_text(document + "\n")


def build_html(metrics):
    best = metrics[("fno_less", "raw")]["mae_dex"]
    write_html(
        "overview",
        """<main class="canvas dark">
<div class="eyebrow">Galaxy cluster mass estimation · ACT + Planck</div>
<h1>From microwave maps<br>to cluster masses.</h1>
<p class="subtitle">Fourier Neural Operators combine multi-frequency sky cutouts with cluster redshift to regress catalog log-mass.</p>
<div class="bands"><div class="band"><strong>90</strong> GHz</div><div class="band"><strong>150</strong> GHz</div><div class="band"><strong>220</strong> GHz</div><div class="band">PyTorch · FNO · CNN</div></div>
<div class="pipeline">
<div class="step"><div class="n">01 / INPUT</div><h2>Sky cutouts</h2><p>3 channels × 148 × 148 pixels<br>ACT+Planck intensity maps</p></div><div class="arrow">→</div>
<div class="step"><div class="n">02 / ENCODER</div><h2>Fourier features</h2><p>Spectral convolutions<br>+ local 1 × 1 convolutions</p></div><div class="arrow">→</div>
<div class="step"><div class="n">03 / FUSION</div><h2>Map features + z</h2><p>Flattened image features<br>concatenated with redshift</p></div><div class="arrow">→</div>
<div class="step"><div class="n">04 / REGRESSION</div><h2>Catalog log-mass</h2><p>Fully connected head<br>log10(M500cUncorr)</p></div>
</div><div class="rule"></div><footer class="footer"><span>Research project · four FNO configurations and a CNN comparison</span><span>Architecture shown: FNO branch</span></footer></main>""",
    )
    write_html(
        "results",
        f"""<main class="canvas"><div class="eyebrow">Saved experiments / original test split</div><div class="header-row"><div><h1>Architecture, normalization &amp; mass.</h1><p class="subtitle">Three FNO configurations, the same 1,994 test targets.<br>Single-run results in log-mass space; lower error is better.</p></div><div class="metric"><strong>{best:.4f} <small style="font-size:19px">dex</small></strong><span>Lowest saved MAE<br>FNO less · raw maps</span></div></div><img class="chart" src="charts/performance.png" alt="Grouped MAE comparison and paired predictions from the original test split"><div class="rule"></div><footer class="footer"><span>Source: results/fno_{{base,wide,less}}/{{raw,normalized}}/</span><span>No augmented samples in this comparison</span></footer></main>""",
    )
    write_html(
        "diagnostics",
        """<main class="canvas"><div class="eyebrow">Model diagnostics / saved artifacts</div><h1>Learning curves &amp; residuals.</h1><p class="subtitle">100 training epochs and the error distribution on the original test split.</p><img class="chart" src="charts/diagnostics.png" alt="Training MSE curves and FNO less raw residual histogram"><div class="rule"></div><footer class="footer"><span>Solid lines: raw maps · dashed lines: normalized maps</span><span>Residual = predicted log-mass − catalog log-mass</span></footer></main>""",
    )


def export_pngs():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = Path(playwright.chromium.executable_path)
        options = (
            {"executable_path": str(executable)}
            if executable.is_file()
            else {"channel": "chrome"}
        )
        browser = playwright.chromium.launch(headless=True, **options)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100}, device_scale_factor=1.5
        )
        for name in ["overview", "results", "diagnostics"]:
            page.goto((ASSETS / f"{name}.html").as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            page.locator("img").evaluate_all(
                "imgs => Promise.all(imgs.map(img => img.decode()))"
            )
            clipped = page.locator(".canvas *").evaluate_all(
                "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.tagName + ': ' + e.textContent.slice(0, 70))"
            )
            if clipped:
                raise ValueError(f"Overflow in {name}: {clipped}")
            page.locator(".canvas").screenshot(
                path=str(ASSETS / f"{name}.png"), animations="disabled"
            )
            print(f"Exported docs/assets/{name}.png")
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Build HTML and charts, skip Chromium export",
    )
    args = parser.parse_args()
    build_html(scientific_charts())
    if not args.html_only:
        export_pngs()
