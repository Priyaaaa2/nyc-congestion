import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
import config

PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building Figure 5 — mode shift by income quintile...")

    synth = pd.read_parquet(PROC / "integration/equity_synthesis.parquet")
    synth = synth.dropna(subset=["median_income",
                                  "yellow_pct", "mta_pct", "suppressed_pct"])

    synth["income_q"] = pd.qcut(
        synth["median_income"], q=5,
        labels=["Q1\n$64k", "Q2\n$110k", "Q3\n$136k",
                "Q4\n$155k", "Q5\n$181k"]
    )

    summary = synth.groupby("income_q", observed=True).agg(
        yellow_pct     = ("yellow_pct",     "mean"),
        mta_pct        = ("mta_pct",        "mean"),
        suppressed_pct = ("suppressed_pct", "mean"),
        effect_pct     = ("effect_pct",     "mean"),
        n              = ("zone_id",        "count"),
    ).reset_index()

    labels   = [str(q) for q in summary["income_q"]]
    yellow   = summary["yellow_pct"].values
    subway   = summary["mta_pct"].values
    suppress = summary["suppressed_pct"].values

    x   = np.arange(len(labels))
    w   = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                              gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    b1 = ax.bar(x, subway,   width=w, color="#378ADD",
                label="→ Subway",      zorder=3)
    b2 = ax.bar(x, yellow,   width=w, color="#EF9F27",
                bottom=subway,
                label="→ Yellow taxi", zorder=3)
    b3 = ax.bar(x, suppress, width=w, color="#E24B4A",
                bottom=subway + yellow,
                label="→ Suppressed demand", zorder=3)

    for i, (s, y_, sp) in enumerate(zip(subway, yellow, suppress)):
        if s > 5:
            ax.text(i, s / 2, f"{s:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="500")
        if y_ > 5:
            ax.text(i, s + y_ / 2, f"{y_:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="500")
        if sp > 5:
            ax.text(i, s + y_ + sp / 2, f"{sp:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="500")

    for i, row in summary.iterrows():
        ax.text(i, -6, f"{row['effect_pct']:.1f}%",
                ha="center", va="top", fontsize=8,
                color="#3d3d3a")
    ax.text(-0.6, -6, "Trip loss:", ha="left", va="top",
            fontsize=8, color="#3d3d3a", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(-12, 108)
    ax.set_ylabel("Share of lost HVFHV trips (%)", fontsize=10)
    ax.set_xlabel("Income quintile (avg median household income)", fontsize=10)
    ax.set_title(
        "Where did lost HVFHV trips go?\nby neighbourhood income quintile",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    ax.tick_params(labelsize=9)
    ax.axhline(0, color="#888780", linewidth=0.6)

    ax2 = axes[1]
    synth["has_sub"] = synth["has_subway"].astype(int)
    groups = synth.groupby("has_sub")["suppressed_pct"].mean()

    bars = ax2.bar(
        ["With subway\nwithin 400m", "No subway\nwithin 400m"],
        [groups.get(1, 0), groups.get(0, 0)],
        color=["#378ADD", "#E24B4A"],
        width=0.45, zorder=3
    )
    for bar, val in zip(bars,
                        [groups.get(1, 0), groups.get(0, 0)]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{val:.1f}%",
                 ha="center", va="bottom",
                 fontsize=11, fontweight="500")

    y1 = groups.get(1, 0)
    y2 = groups.get(0, 0)
    ax2.annotate("",
                 xy=(1, y2), xytext=(1, y1),
                 arrowprops=dict(arrowstyle="<->",
                                 color="#3d3d3a", lw=1.4))
    ax2.text(1.26, (y1 + y2) / 2,
             f"{y2 - y1:.0f}pp\ngap",
             ha="left", va="center",
             fontsize=9, color="#3d3d3a")

    ax2.set_ylabel("Avg suppressed demand (%)", fontsize=10)
    ax2.set_title(
        "Suppressed demand\nvs transit access",
        fontsize=11, fontweight="bold", pad=10
    )
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", alpha=0.2, zorder=0)
    ax2.tick_params(labelsize=9)

    fig.suptitle(
        "Mode Shift Decomposition by Income Quintile | CRZ Zones",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.text(0.5, -0.03,
             "Source: NYC TLC, MTA subway ridership, US Census ACS 2023. "
             "63 CRZ zones, Jan\u2013Jun 2025 vs Jan\u2013Jun 2024. "
             "Right panel: low-income zones (below median) split by subway access.",
             ha="center", fontsize=7.5, color="#666666")

    plt.tight_layout()
    out = OUT_DIR / "fig5_mode_by_income.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
