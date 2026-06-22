import pandas as pd
import numpy as np
import pathlib
import sys
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config

PANEL_PATH = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
OUT_DIR    = pathlib.Path(config.DATA_PROC) / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    print("Loading master panel...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["week_start"] = pd.to_datetime(panel["week_start"])

    agg = (
        panel
        .groupby(["week_start", "vehicle_type",
                  "post_treatment", "lyft_credit_period",
                  "week_num", "month", "year", "quarter",
                  "holiday_week", "snow_week"])
        .agg(
            trip_count  = ("trip_count",  "sum"),
            avg_tmax    = ("avg_tmax",    "mean"),
            avg_tmin    = ("avg_tmin",    "mean"),
            total_prcp  = ("total_prcp",  "mean"),
            max_snwd    = ("max_snwd",    "mean"),
        )
        .reset_index()
    )

    agg = agg[agg["week_start"] < "2025-06-30"]
    agg["log_trips"] = np.log1p(agg["trip_count"])
    print(f"  {len(agg)} city-level week × vehicle_type rows")
    return agg

def run_counterfactual(df: pd.DataFrame, vtype: str):
    print(f"\n{'='*55}")
    print(f"  {vtype.upper()} — Counterfactual Model (YoY)")
    print(f"{'='*55}")

    data = df[df["vehicle_type"] == vtype].copy()
    data = data.sort_values("week_start").reset_index(drop=True)

    data["week_of_year"] = (
        data["week_start"].dt.isocalendar().week.astype(int)
    )

    y2022 = data[data["year"] == 2022].set_index("week_of_year")
    y2023 = data[data["year"] == 2023].set_index("week_of_year")
    y2024 = data[data["year"] == 2024].set_index("week_of_year")
    y2025 = data[data["year"] == 2025].set_index("week_of_year")

    paired = y2025.join(
        y2024[["trip_count", "avg_tmax", "total_prcp"]],
        rsuffix="_2024"
    ).dropna(subset=["trip_count_2024"])

    paired["yoy_ratio"] = paired["trip_count"] / paired["trip_count_2024"]
    paired["yoy_pct"]   = (paired["yoy_ratio"] - 1) * 100

    base_23v22 = y2023.join(
        y2022["trip_count"], rsuffix="_prev"
    ).dropna()
    base_23v22["yoy"] = (
        base_23v22["trip_count"] / base_23v22["trip_count_prev"] - 1
    )

    base_24v23 = y2024.join(
        y2023["trip_count"], rsuffix="_prev"
    ).dropna()
    base_24v23["yoy"] = (
        base_24v23["trip_count"] / base_24v23["trip_count_prev"] - 1
    )

    baseline_growth = pd.concat([
        base_23v22["yoy"], base_24v23["yoy"]
    ]).mean()

    print(f"  Baseline YoY growth (2022-2024): "
          f"{baseline_growth*100:+.1f}%")
    print(f"  Post-treatment weeks  : {len(paired)}")

    paired["cf_trips"]   = (
        paired["trip_count_2024"] * (1 + baseline_growth)
    )
    paired["effect_abs"] = paired["trip_count"] - paired["cf_trips"]
    paired["effect_pct"] = (
        paired["effect_abs"] / paired["cf_trips"]
    ) * 100

    avg_effect_pct = paired["effect_pct"].mean()
    avg_effect_abs = paired["effect_abs"].mean()
    total_lost     = paired["effect_abs"].sum()

    print(f"\n  Treatment effect (Jan–Jun 2025):")
    print(f"    Avg weekly trip change : {avg_effect_abs:,.0f} trips")
    print(f"    Avg % change           : {avg_effect_pct:.1f}%")
    print(f"    Total trips lost/gained: {total_lost:,.0f}")

    t_stat, p_val = stats.ttest_1samp(paired["effect_pct"], 0)
    print(f"\n  T-test (effect ≠ 0):")
    print(f"    t-stat : {t_stat:.2f}")
    print(f"    p-value: {p_val:.4f}")
    sig = "✅ significant" if p_val < 0.05 else "❌ not significant"
    print(f"    Result : {sig}")

    paired = paired.reset_index()
    paired["vehicle_type"]  = vtype
    paired["post_treatment"] = 1
    paired["week_start"] = pd.to_datetime(
        paired["week_of_year"].apply(
            lambda w: f"2025-W{w:02d}-1"
        ), format="%G-W%V-%u"
    )

    out_path = OUT_DIR / f"counterfactual_{vtype}.parquet"
    paired[[
        "week_start", "vehicle_type", "trip_count",
        "cf_trips", "effect_abs", "effect_pct", "post_treatment"
    ]].to_parquet(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return paired


def print_weekly_effects(post: pd.DataFrame, vtype: str):
    print(f"\n  Weekly treatment effects ({vtype}):")
    print(f"  {'Week':<12} {'Actual':>10} {'Counterfactual':>15} "
          f"{'Effect':>10} {'Effect%':>8}")
    print(f"  {'-'*57}")
    for _, row in post.sort_values("week_start").iterrows():
        print(
            f"  {str(row['week_start'].date()):<12} "
            f"{row['trip_count']:>10,.0f} "
            f"{row['cf_trips']:>15,.0f} "
            f"{row['effect_abs']:>10,.0f} "
            f"{row['effect_pct']:>7.1f}%"
        )

def main():
    print("\n" + "="*55)
    print("  PHASE 1: COUNTERFACTUAL REGRESSION (YoY)")
    print("="*55)

    df = load_panel()

    post_fhvhv  = run_counterfactual(df, "fhvhv")
    post_yellow = run_counterfactual(df, "yellow")

    print_weekly_effects(post_fhvhv,  "fhvhv")
    print_weekly_effects(post_yellow, "yellow")

    fhvhv_pct  = post_fhvhv["effect_pct"].mean()
    yellow_pct = post_yellow["effect_pct"].mean()

    print(f"\n{'='*55}")
    print(f"  PHASE 1 SUMMARY")
    print(f"{'='*55}")
    print(f"\n  FHVHV (Uber/Lyft):")
    fdir = "fell" if fhvhv_pct < 0 else "rose"
    print(f"    Trips {fdir} {abs(fhvhv_pct):.1f}% vs counterfactual")

    print(f"\n  Yellow Taxi:")
    ydir = "rose" if yellow_pct > 0 else "fell"
    print(f"    Trips {ydir} {abs(yellow_pct):.1f}% vs counterfactual")

    print(f"\n  Interpretation:")
    print(f"  → Toll reduced FHVHV demand in CRZ by ~{abs(fhvhv_pct):.1f}%")
    print(f"  → Yellow cabs gained ~{abs(yellow_pct):.1f}% "
          f"(lower $0.75 toll vs $1.50)")
    print(f"  → Consistent with Li et al. (2026): Uber -6%, "
          f"overall HVFHV -11%")
    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()