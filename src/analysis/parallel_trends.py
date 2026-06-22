import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def load_data() -> pd.DataFrame:
    print("Loading platform weekly data...")
    path = PROC / "phase4/platform_weekly.parquet"
    df   = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df[df["platform"].isin(["lyft", "uber"])].copy()
    print(f"  Rows: {len(df):,}")
    print(f"  Weeks: {df['week_start'].nunique()}")
    return df
 
 
def make_chart(df: pd.DataFrame):
    print("Building parallel trends chart...")
 
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
 
    ax1 = axes[0]
 
    lyft = df[df["platform"] == "lyft"].sort_values("week_start")
    uber = df[df["platform"] == "uber"].sort_values("week_start")
 
    cutoff     = pd.Timestamp(config.TREATMENT_DATE)
    credit_end = pd.Timestamp(config.LYFT_CREDIT_END)
 
    ax1.plot(lyft["week_start"], lyft["trip_count"] / 1e6,
             color="#FF5722", linewidth=1.8, label="Lyft (HV0005)",
             zorder=3)
    ax1.plot(uber["week_start"], uber["trip_count"] / 1e6,
             color="#2196F3", linewidth=1.8, label="Uber (HV0003)",
             zorder=3)

    pre_start = df["week_start"].min()
    ax1.axvspan(pre_start, cutoff,
                alpha=0.05, color="#4CAF50",
                label="Pre-treatment period")

    ax1.axvspan(cutoff, credit_end,
                alpha=0.12, color="#FF5722",
                label="Lyft credit month (Jan 2025)")

    ax1.axvline(cutoff, color="#333333", linewidth=1.5,
                linestyle="--", zorder=4)
    ax1.text(cutoff, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 1.8,
             "  CBDTP launch\n  Jan 5, 2025",
             fontsize=8.5, color="#333333", va="top")
 
    credit_mid = cutoff + (credit_end - cutoff) / 2
    lyft_jan   = lyft[
        (lyft["week_start"] >= cutoff) &
        (lyft["week_start"] <= credit_end)
    ]["trip_count"].mean() / 1e6
    ax1.annotate(
        "Lyft $1.50\ncredit active",
        xy=(credit_mid, lyft_jan),
        xytext=(credit_mid, lyft_jan + 0.08),
        fontsize=8, color="#FF5722", ha="center",
        arrowprops=dict(arrowstyle="->", color="#FF5722",
                        lw=1.2)
    )
 
    ax1.set_ylabel("Weekly CRZ trips (millions)", fontsize=11)
    ax1.set_title(
        "Parallel Trends: Lyft vs Uber — Weekly CRZ Trips\n"
        "Pre-period correlation: 0.991  |  "
        "DiD τ = +0.151 (SE = 0.035, p < 0.001)  |  R² = 0.881",
        fontsize=12, fontweight="bold", pad=12
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2 = axes[1]
 
    merged = lyft.set_index("week_start")[["trip_count"]].rename(
        columns={"trip_count": "lyft"}
    ).join(
        uber.set_index("week_start")[["trip_count"]].rename(
            columns={"trip_count": "uber"}
        )
    ).dropna()
    merged["ratio"] = merged["lyft"] / merged["uber"]
 
    pre_mask  = merged.index < cutoff
    post_mask = merged.index >= cutoff
 
    ax2.plot(merged.index[pre_mask],  merged["ratio"][pre_mask],
             color="#555555", linewidth=1.5, label="Pre-treatment")
    ax2.plot(merged.index[post_mask], merged["ratio"][post_mask],
             color="#FF5722", linewidth=2.0, label="Post-treatment")

    pre_avg = merged["ratio"][pre_mask].mean()
    ax2.axhline(pre_avg, color="#555555", linewidth=0.8,
                linestyle=":", alpha=0.7,
                label=f"Pre-period avg ({pre_avg:.3f})")
 
    ax2.axvline(cutoff, color="#333333", linewidth=1.5,
                linestyle="--", zorder=4)
    ax2.axvspan(cutoff, credit_end,
                alpha=0.12, color="#FF5722")
 
    ax2.set_ylabel("Lyft / Uber\ntrip ratio", fontsize=10)
    ax2.set_xlabel("Week", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
 
    fig.text(
        0.5, 0.01,
        "Source: NYC TLC FHVHV trip records (raw Parquet). "
        "Lyft = HV0005, Uber = HV0003. "
        "CRZ = Manhattan below 60th St. "
        "Lyft credit: $1.50/trip reimbursement, Jan 1-31 2025 only.",
        ha="center", fontsize=7.5, color="#666666"
    )
 
    plt.tight_layout(rect=[0, 0.04, 1, 1])
 
    out_path = OUT_DIR / "parallel_trends.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved -> {out_path}")
    return out_path
 
 
def main():
    print("\n" + "="*55)
    print("  DELIVERABLE 3: PARALLEL TRENDS CHART")
    print("="*55)
 
    df       = load_data()
    out_path = make_chart(df)
 
    print(f"\n{'='*55}")
    print(f"  DONE: {out_path}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()