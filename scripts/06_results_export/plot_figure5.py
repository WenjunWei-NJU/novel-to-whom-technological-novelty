"""Render five sensitivity panels from the supplied estimates."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = HERE / "source_data"
FORMAL = SOURCE / "formal_estimands.csv"
KEY = ["family", "model", "outcome", "contrast", "covariance"]
OUTCOMES = ["relative_y", "team_y", "pair_y"]
COLORS = ["#1677A6", "#CC7437", "#218C72"]
INK = "#283D49"
WIDTH = 17 / 2.54
HEIGHT = 6.09
BOTTOM_CLEARANCE = .14
BASE = "Figure5_sensitivity_5panels"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.3,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#A8B3BA", "axes.linewidth": .55,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def read_data():
    source = pd.read_csv(SOURCE / "estimands.csv")
    expected = pd.read_csv(SOURCE / "plotted_estimates.csv")
    expected = expected.loc[expected.family.isin(["THRESHOLDS", "HISTORY_WINDOWS", "PERIODS"])]
    data = source.merge(expected[KEY], on=KEY, validate="one_to_one")
    assert data.groupby("family").size().to_dict() == {
        "THRESHOLDS": 27, "HISTORY_WINDOWS": 12, "PERIODS": 12}
    metrics = ["estimate", "CI_low", "CI_high", "N", "p_Holm"]
    errors = {}
    for name, ref in [("original_plot", expected), ("formal", pd.read_csv(FORMAL))]:
        merged = data.merge(ref[KEY + metrics], on=KEY, suffixes=("", "_ref"), validate="one_to_one")
        assert len(merged) == len(data)
        errors[name] = {}
        for col in metrics:
            np.testing.assert_array_equal(merged[col], merged[f"{col}_ref"])
            errors[name][col] = float(np.max(np.abs(merged[col] - merged[f"{col}_ref"])))
    for w, n in [(5, 2270556), (10, 2499100)]:
        pairs = data.loc[data.model.isin([f"history_{w}_full", f"history_{w}_{w}"])]
        assert len(pairs) == 6 and (pairs.N == n).all()
        q = pairs.loc[pairs.outcome == "pair_y", metrics]
        np.testing.assert_array_equal(q.iloc[0], q.iloc[1])
    return data, errors


def axis(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, (y + BOTTOM_CLEARANCE) / HEIGHT, w / WIDTH, h / HEIGHT])


def put(fig, x, y, label, **kwargs):
    return fig.text(x / WIDTH, (y + BOTTOM_CLEARANCE) / HEIGHT, label, **kwargs)


def heading(fig, x, y, letter, title):
    put(fig, x, y, letter, fontsize=10, weight="bold", va="top")
    put(fig, x + .20, y, title, fontsize=8.3, weight="medium", va="top")


def contrast_axis(ax, limits, ticks, labels, *, fine=False):
    ax.set_xlim(limits)
    ax.set_xticks(ticks)
    if fine:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks(range(len(labels) - 1, -1, -1), labels)
    ax.set_ylim(-.52, len(labels) - .48)
    ax.axvline(0, color="#A0ADB4", lw=.65, zorder=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4, labelsize=7.2)
    ax.tick_params(axis="x", length=2.5, pad=2, labelsize=7.0)


def draw_interval(ax, row, y, color, marker, hollow=False):
    limits = ax.get_xlim()
    lo, hi = row.CI_low * 100, row.CI_high * 100
    assert limits[0] < lo <= row.estimate * 100 <= hi < limits[1]
    ax.plot([lo, hi], [y, y], color=color, lw=1.12, solid_capstyle="round", zorder=2)
    ax.plot(row.estimate * 100, y, marker, ms=3.6,
            mfc="white" if hollow else color, mec=color, mew=.9, zorder=3)


def main():
    data, errors = read_data()
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    put(fig, .26, 5.82, "Sensitivity of the prespecified composition contrast",
        fontsize=11.2, weight="bold", va="top")
    # Original 3×3 order, value formats, palette and common range.
    for j, (outcome, title) in enumerate(zip(
        OUTCOMES, ["Mean personal novelty", "Team novelty", "Corpus-first pairs"]
    )):
        x = .43 + j * 2.15
        heading(fig, x, 5.54, "ABC"[j], title + "\nThreshold stability")
        ax = axis(fig, x, 3.82, 1.38, 1.38)
        q = data[(data.family == "THRESHOLDS") & (data.outcome == outcome)].copy()
        q["early"] = q.model.str.extract(r"E(\d+)").astype(int)
        q["senior"] = q.model.str.extract(r"S(\d+)").astype(int)
        mat = q.pivot(index="early", columns="senior", values="estimate").loc[[7, 5, 3], [15, 16, 20]] * 100
        edges = np.arange(4) - .5
        im = ax.pcolormesh(edges, edges, mat, cmap="RdBu_r", vmin=-1, vmax=1,
                           shading="flat", linewidth=0)
        ax.set(ylim=(2.5, -.5), aspect="equal")
        for i, row in enumerate(mat.to_numpy()):
            for k, value in enumerate(row):
                rgb = np.asarray(im.cmap(im.norm(value))[:3])
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = linear @ [.2126, .7152, .0722]
                ax.text(k, i, f"{value:.2f}" if j < 2 else f"{value:.2g}",
                        ha="center", va="center", fontsize=7.6,
                        color="white" if luminance < .29 else INK)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=[15, 16, 20], yticklabels=[7, 5, 3])
        ax.set_xlabel("Senior threshold (years)", labelpad=2)
        ax.set_ylabel("Early threshold (years)", labelpad=3)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = axis(fig, x + 1.412, 3.82, .072, 1.38)
        cb = fig.colorbar(im, cax=bar, ticks=[-1, -.5, 0, .5, 1])
        cb.solids.set_rasterized(False)
        cb.ax.tick_params(labelsize=6.8, length=2, pad=2)
        cb.set_label("Contrast (pp)", fontsize=7.0, labelpad=0)

    heading(fig, .26, 3.17, "D", "Matched historical windows")
    heading(fig, 3.59, 3.17, "E", "Application periods")
    plots = data.set_index(["family", "model", "outcome"], verify_integrity=True)
    for col in range(2):
        x = [.60, 3.95][col]
        for j, (outcome, color, title) in enumerate(zip(
            OUTCOMES, COLORS, ["Personal", "Team", "Corpus-first pairs · Fine scale"]
        )):
            y = [2.37, 1.42, .47][j]
            ax = axis(fig, x, y, 2.57, .55)
            put(fig, x, y + .585, title, fontsize=7.5, color=color)
            fine = j == 2
            if col == 0:
                contrast_axis(ax, (-.018, .013) if fine else (-.25, 1.05),
                              [-.01, 0, .01] if fine else [0, .5, 1],
                              ["5-year", "10-year"], fine=fine)
                for i, w in enumerate([5, 10]):
                    for ref, marker, dy in [("full", "o", .15), (str(w), "s", -.15)]:
                        r = plots.loc[("HISTORY_WINDOWS", f"history_{w}_{ref}", outcome)]
                        draw_interval(ax, r, 1 - i + dy, color, marker, hollow=ref == "full")
            else:
                contrast_axis(ax, (-.055, .075) if fine else (-.75, 3),
                              [-.05, 0, .05] if fine else [0, 1, 2, 3],
                              ["2000–2009", "2010–2014", "2015–2019", "2020–2024"], fine=fine)
                for i, period in enumerate(["2000_2009", "2010_2014", "2015_2019", "2020_2024"]):
                    r = plots.loc[("PERIODS", f"period_{period}", outcome)]
                    draw_interval(ax, r, 3 - i, color, "o")
            if fine:
                ax.set_xlabel("AI − non-AI contrast (pp)", labelpad=2)
    fig.legend(handles=[
        Line2D([], [], color=INK, marker="o", mfc="white", ls="", ms=4, label="Full history"),
        Line2D([], [], color=INK, marker="s", ls="", ms=4, label="Window history"),
    ], loc="lower left", bbox_to_anchor=(.54 / WIDTH, (.006 * 5.57) / HEIGHT), ncol=2, frameon=False,
        fontsize=7.2, handletextpad=.45, columnspacing=1.1)
    fig.savefig(HERE / (BASE + ".png"), dpi=600)
    fig.savefig(HERE / (BASE + ".pdf"))
    # Inspect at nominal office screen density, retaining the physical 17 cm width.
    fig.savefig(HERE / "preview_17cm.png", dpi=96)
    plt.close(fig)
    data.to_csv(HERE / "plotted_estimates_A-E.csv", index=False)
    (HERE / "validation.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256((SOURCE / "estimands.csv").read_bytes()).hexdigest(),
        "rows": len(data), "maximum_absolute_errors": errors, "width_cm": 17,
        "height_cm": HEIGHT * 2.54, "png_dpi": 600,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
