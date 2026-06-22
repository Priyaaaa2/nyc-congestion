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

SCENARIO_LABELS = {
    "A_flat_9":  "A — \$9 flat (HVFHV \$1.50)",
    "B_rise_15": "B — Rise to \$15 in Jan 2027 (HVFHV \$2.50)",
    "C_pause":   "C — 6-month pause Jul–Dec 2025",
}
SCENARIO_COLORS = {
    "A_flat_9":  "#378ADD",
    "B_rise_15": "#E24B4A",
    "C_pause":   "#EF9F27",
}


def main():
    print("Building Figure 6 — scenario forecast lines...")

    base = pd.read_parquet(PROC / "phase5/forecast_base.parquet")
    sens = pd.read_parquet(PROC / "phase5/forecast_2x.parquet")

    base["date"] = pd.to_datetime(
        base["year"].astype(str) + "-" +
        base["month_of_year"].astype(str).str.zfill(2) + "-01"
    )
    sens["date"] = base["date"].values 

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], "monthly_trips",  "Monthly HVFHV trips (millions)", "Trips"),
        (axes[1], "mta_revenue",    "MTA toll revenue per month ($M)", "Revenue"),
    ]:
        for sc, label in SCENARIO_LABELS.items():
            df = base[base["scenario"] == sc].sort_values("date")
            y  = df[metric].values / (1e6 if metric == "monthly_trips" else 1e6)
            ax.plot(df["date"], y,
                    color=SCENARIO_COLORS[sc],
                    linewidth=2.0, label=label)

            if sc == "B_rise_15":
                df2 = sens[sens["scenario"] == sc + "_2x"].sort_values("date")
                y2  = df2[metric].values / 1e6
                ax.plot(df2["date"], y2,
                        color=SCENARIO_COLORS[sc],
                        linewidth=1.2, linestyle=":",
                        label="B (2\u00d7 elasticity)")

        toll_change = pd.Timestamp("2027-01-01")
        ax.axvline(toll_change, color="#E24B4A",
                   linewidth=1.0, linestyle="--", alpha=0.5)
        ax.text(toll_change + pd.Timedelta(days=15),
                ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
                "Scenario B\ntoll rises",
                fontsize=7.5, color="#E24B4A", va="bottom")

        ax.axvspan(pd.Timestamp("2025-07-01"),
                   pd.Timestamp("2026-01-01"),
                   alpha=0.07, color="#EF9F27")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_title(f"{title_suffix} — Three Scenarios",
                     fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%b\n%Y")
        )
        ax.xaxis.set_major_locator(
            plt.matplotlib.dates.MonthLocator(interval=6)
        )

    fig.suptitle(
        "Three-Scenario HVFHV Forecast | January 2025 – December 2027\n"
        "Elasticity: \u03b5 = \u22121.27 (base) and \u22122.53 (sensitivity). "
        "Baseline growth: +3.4%/yr.",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.02,
             "Scenario A: \$9 toll flat. "
             "Scenario B: toll rises to \$15 in Jan 2027. "
             "Scenario C: toll suspended Jul\u2013Dec 2025. "
             "Orange shading = pause window. "
             "Dotted line = 2\u00d7 elasticity sensitivity.",
             ha="center", fontsize=7.5, color="#666666")

    plt.tight_layout()
    out = OUT_DIR / "fig6_scenario_lines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
