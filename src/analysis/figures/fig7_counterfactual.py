"""
src/analysis/figures/fig7_counterfactual.py
Figure 7 — Actual vs counterfactual weekly HVFHV trips
Run: python -m src.analysis.figures.fig7_counterfactual
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
import config

PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_GROWTH  = 0.034
TREATMENT_DATE   = pd.Timestamp(config.TREATMENT_DATE)
LYFT_CREDIT_END  = pd.Timestamp(config.LYFT_CREDIT_END)


def load_panel():
    panel = pd.read_parquet(PROC / "master_panel.parquet")
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    agg = (
        panel[panel["vehicle_type"] == "fhvhv"]
        .groupby("week_start")["trip_count"].sum()
        .reset_index().sort_values("week_start")
    )
    return agg[agg["week_start"] < "2025-06-30"]


def build_cf(agg):
    agg = agg.copy()
    agg["year"] = agg["week_start"].dt.year
    agg["woy"]  = agg["week_start"].dt.isocalendar().week.astype(int)

    y2024 = agg[agg["year"] == 2024].set_index("woy")
    y2025 = agg[agg["year"] == 2025].set_index("woy")

    paired = y2025.join(
        y2024[["trip_count"]].rename(columns={"trip_count": "trips_2024"}),
        how="left"
    ).dropna().reset_index()

    paired["cf"]    = paired["trips_2024"] * (1 + BASELINE_GROWTH)
    paired["cf_lo"] = paired["cf"] * 0.90
    paired["cf_hi"] = paired["cf"] * 1.10
    paired["week_start"] = pd.to_datetime(
        paired["woy"].apply(lambda w: f"2025-W{w:02d}-1"),
        format="%G-W%V-%u"
    )
    return paired.sort_values("week_start")


def main():
    print("Building Figure 7 — counterfactual plot...")
    agg    = load_panel()
    paired = build_cf(agg)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Pre-treatment actual
    pre = agg[agg["week_start"] < TREATMENT_DATE]
    ax.plot(pre["week_start"], pre["trip_count"] / 1e6,
            color="#B4B2A9", linewidth=1.4, label="Actual (pre-treatment)")

    # Post-treatment actual
    post = agg[agg["week_start"] >= TREATMENT_DATE]
    ax.plot(post["week_start"], post["trip_count"] / 1e6,
            color="#1A252F", linewidth=2.0, label="Actual (post-treatment)")

    # Counterfactual + CI band
    ax.fill_between(paired["week_start"],
                    paired["cf_lo"] / 1e6,
                    paired["cf_hi"] / 1e6,
                    alpha=0.15, color="#1D9E75")
    ax.plot(paired["week_start"], paired["cf"] / 1e6,
            color="#1D9E75", linewidth=1.8, linestyle="--",
            label="Counterfactual (2024 \u00d7 1.034 baseline)")

    # Lyft credit shading
    ax.axvspan(TREATMENT_DATE, LYFT_CREDIT_END,
               alpha=0.08, color="#FF5722", label="Lyft credit (Jan 2025)")

    # Treatment date line + label
    ax.axvline(TREATMENT_DATE, color="#3d3d3a",
               linewidth=1.4, linestyle="--", zorder=5)
    ymax = agg["trip_count"].max() / 1e6
    ax.text(TREATMENT_DATE + pd.Timedelta(days=3), ymax * 0.97,
            "CBDTP\nJan 5, 2025", fontsize=8, color="#3d3d3a", va="top")

    # Arrow annotating the gap
    mid = paired.iloc[len(paired)//3]
    actual_mid = agg[agg["week_start"] == mid["week_start"]]["trip_count"]
    if len(actual_mid):
        ya = actual_mid.values[0] / 1e6
        yc = mid["cf"] / 1e6
        ax.annotate("", xy=(mid["week_start"], ya),
                    xytext=(mid["week_start"], yc),
                    arrowprops=dict(arrowstyle="<->",
                                   color="#712B13", lw=1.4))
        ax.text(mid["week_start"] + pd.Timedelta(days=4),
                (ya + yc) / 2,
                f"\u22127.0%\ntreatment\neffect",
                fontsize=8, color="#712B13", va="center")

    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-07-01"))
    ax.set_ylabel("Weekly CRZ HVFHV trips (millions)", fontsize=10)
    ax.set_xlabel("Week", fontsize=10)
    ax.set_title(
        "Actual vs Counterfactual Weekly HVFHV Trips | Manhattan CRZ\n"
        "YoY counterfactual = same ISO week 2024 \u00d7 (1 + 3.4% baseline growth)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.text(0.5, 0.01,
             "Source: NYC TLC FHVHV trip records. "
             "Counterfactual = same ISO week in 2024 \u00d7 1.034. "
             "Shaded band = \u00b110% counterfactual uncertainty. "
             "Orange shading = Lyft $1.50 credit period.",
             ha="center", fontsize=7.5, color="#666666")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "fig7_counterfactual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
