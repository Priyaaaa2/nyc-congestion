import pandas as pd
import numpy as np
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PANEL_PATH   = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
PHASE1_PATH  = pathlib.Path(config.DATA_PROC) / "phase1"
OUT_DIR      = pathlib.Path(config.DATA_PROC) / "phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 

ELASTICITY_BASE  = -1.267 
ELASTICITY_2X    = -1.267 * 2  
 
BASELINE_GROWTH  = 0.034

TOLL_HVFHV_BASE  = 1.50    
TOLL_HVFHV_HIGH  = (15/9) * 1.50

WEEKS_PER_YEAR   = 52
 
 
def get_baseline_weekly_trips() -> float:
    """Average weekly CRZ HVFHV trips in 2024."""
    panel = pd.read_parquet(PANEL_PATH)
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    base = panel[
        (panel["vehicle_type"] == "fhvhv") &
        (panel["in_crz"] == 1) &
        (panel["week_start"] >= "2024-01-01") &
        (panel["week_start"] <  "2025-01-01")
    ]["trip_count"].sum() / WEEKS_PER_YEAR
    return base
 
 
def build_scenario(
    name: str,
    baseline_weekly: float,
    toll_schedule: dict,   
    elasticity: float,
    n_months: int = 36
) -> pd.DataFrame:
    """
    Build monthly forecast for one scenario.
 
    Logic:
    - Start from 2024 annual average weekly trips
    - Apply organic growth trend each month
    - Apply elasticity-based demand response to toll level
    - Compute MTA revenue = trips × toll per trip
    """
    rows = []
    base_toll = TOLL_HVFHV_BASE  
 
    for month in range(1, n_months + 1):
        
        year   = 2025 + (month - 1) // 12
        month_of_year = ((month - 1) % 12) + 1

        toll = toll_schedule.get(month, TOLL_HVFHV_BASE)
 
        monthly_growth = (1 + BASELINE_GROWTH) ** (month / 12)
        organic_trips_weekly = baseline_weekly * monthly_growth

        pct_price_change = (toll - base_toll) / base_toll
        pct_demand_change = elasticity * pct_price_change
        adjusted_trips_weekly = organic_trips_weekly * \
                                (1 + pct_demand_change)

        monthly_trips = adjusted_trips_weekly * (WEEKS_PER_YEAR / 12)
 
        mta_revenue = monthly_trips * toll
 
        rows.append({
            "scenario":      name,
            "month":         month,
            "year":          year,
            "month_of_year": month_of_year,
            "toll":          toll,
            "weekly_trips":  adjusted_trips_weekly,
            "monthly_trips": monthly_trips,
            "mta_revenue":   mta_revenue,
            "elasticity":    elasticity,
        })
 
    return pd.DataFrame(rows)
 
 
def define_toll_schedules() -> dict:
    """
    Three scenarios:
    A: $9 toll stays flat (HVFHV pays $1.50)
    B: Toll rises to $15 in month 25 (Jan 2027)
       HVFHV share scales proportionally → $2.50
    C: 6-month pause starting month 7 (Jul 2025),
       reinstated at $1.50 in month 13 (Jan 2026)
    """
    sched_a = {m: 1.50 for m in range(1, 37)}

    sched_b = {m: 1.50 for m in range(1, 25)}
    sched_b.update({m: TOLL_HVFHV_HIGH for m in range(25, 37)})
    sched_c = {m: 1.50 for m in range(1, 7)}
    sched_c.update({m: 0.00 for m in range(7, 13)})
    sched_c.update({m: 1.50 for m in range(13, 37)})
 
    return {"A_flat_9": sched_a,
            "B_rise_15": sched_b,
            "C_pause": sched_c}
 
 
def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly forecast to annual totals."""
    annual = (df
        .groupby(["scenario", "year"])
        .agg(
            hvfhv_trips    = ("monthly_trips", "sum"),
            mta_revenue    = ("mta_revenue",   "sum"),
            avg_toll       = ("toll",           "mean"),
        )
        .reset_index()
    )
    annual["mta_revenue_M"] = annual["mta_revenue"] / 1e6
    annual["hvfhv_trips_M"] = annual["hvfhv_trips"] / 1e6
    return annual
 
 
def run_sensitivity(baseline_weekly: float,
                    schedules: dict) -> pd.DataFrame:
    """Run all scenarios with 2× elasticity."""
    frames = []
    for name, sched in schedules.items():
        df = build_scenario(
            name + "_2x", baseline_weekly,
            sched, ELASTICITY_2X
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
 
 
def print_annual_table(annual: pd.DataFrame, label: str):
    print(f"\n  {label}")
    print(f"  {'Scenario':<15} {'Year':>6} "
          f"{'HVFHV trips (M)':>16} "
          f"{'MTA revenue ($M)':>17} "
          f"{'Avg toll':>9}")
    print(f"  {'-'*67}")
    for _, row in annual.iterrows():
        print(f"  {row['scenario']:<15} {int(row['year']):>6} "
              f"{row['hvfhv_trips_M']:>16.2f} "
              f"{row['mta_revenue_M']:>17.1f} "
              f"${row['avg_toll']:>7.2f}")
 
 
def print_scenario_comparison(base: pd.DataFrame,
                               sens: pd.DataFrame):
    """Show revenue impact of 2× elasticity."""
    print("\n" + "="*55)
    print("  SENSITIVITY: 2× ELASTICITY IMPACT ON REVENUE")
    print("="*55)
 
    base_ann = annual_summary(base)
    sens_ann = annual_summary(sens)
 
    sens_ann["scenario"] = sens_ann["scenario"].str.replace("_2x","")
 
    merged = base_ann.merge(
        sens_ann[["scenario","year","mta_revenue_M"]],
        on=["scenario","year"],
        suffixes=("_base","_2x")
    )
    merged["revenue_diff_M"] = (
        merged["mta_revenue_M_2x"] - merged["mta_revenue_M_base"]
    )
 
    print(f"\n  {'Scenario':<15} {'Year':>6} "
          f"{'Base rev ($M)':>14} "
          f"{'2x elas ($M)':>13} "
          f"{'Diff ($M)':>10}")
    print(f"  {'-'*60}")
    for _, row in merged.iterrows():
        print(f"  {row['scenario']:<15} {int(row['year']):>6} "
              f"{row['mta_revenue_M_base']:>14.1f} "
              f"{row['mta_revenue_M_2x']:>13.1f} "
              f"{row['revenue_diff_M']:>10.1f}")
 
 
def main():
    print("\n" + "="*55)
    print("  PHASE 5: THREE-SCENARIO REVENUE FORECAST")
    print("="*55)
 
    baseline_weekly = get_baseline_weekly_trips()
    print(f"\n  2024 baseline weekly CRZ HVFHV trips: "
          f"{baseline_weekly:,.0f}")
    print(f"  Elasticity (Phase 1 implied)         : "
          f"{ELASTICITY_BASE}")
    print(f"  Organic growth rate                  : "
          f"{BASELINE_GROWTH*100:.1f}% YoY")
    schedules = define_toll_schedules()
    print(f"\n  Scenarios:")
    print(f"    A — $9 flat (HVFHV $1.50) for 36 months")
    print(f"    B — Rise to $15 in Jan 2027 (HVFHV $2.50)")
    print(f"    C — 6-month pause Jul-Dec 2025, then $9 restored")

    print(f"\n  Running base case forecasts...")
    base_frames = []
    for name, sched in schedules.items():
        df = build_scenario(
            name, baseline_weekly, sched, ELASTICITY_BASE
        )
        base_frames.append(df)
    base_all = pd.concat(base_frames, ignore_index=True)
 
    print(f"  Running 2× elasticity sensitivity...")
    sens_all = run_sensitivity(baseline_weekly, schedules)

    base_annual = annual_summary(base_all)
    sens_annual = annual_summary(sens_all)
 
    print("\n" + "="*55)
    print("  BASE CASE RESULTS")
    print("="*55)
    print_annual_table(base_annual, "Annual HVFHV trips and MTA revenue")
 
    print_scenario_comparison(base_all, sens_all)

    print("\n" + "="*55)
    print("  KEY FINDINGS")
    print("="*55)
 
    a3 = base_annual[
        (base_annual["scenario"]=="A_flat_9") &
        (base_annual["year"]==2027)
    ].iloc[0]

    b3 = base_annual[
        (base_annual["scenario"]=="B_rise_15") &
        (base_annual["year"]==2027)
    ].iloc[0]

    c1 = base_annual[
        (base_annual["scenario"]=="C_pause") &
        (base_annual["year"]==2025)
    ].iloc[0]
 
    print(f"\n  Year 3 (2027) comparison:")
    print(f"    A ($9 flat)  : {a3['hvfhv_trips_M']:.1f}M trips, "
          f"${a3['mta_revenue_M']:.0f}M MTA revenue")
    print(f"    B ($15 rise) : {b3['hvfhv_trips_M']:.1f}M trips, "
          f"${b3['mta_revenue_M']:.0f}M MTA revenue")
    print(f"    B vs A revenue delta: "
          f"${b3['mta_revenue_M']-a3['mta_revenue_M']:+.0f}M")
 
    print(f"\n  Scenario C (pause) year 1 (2025):")
    print(f"    {c1['hvfhv_trips_M']:.1f}M trips, "
          f"${c1['mta_revenue_M']:.0f}M MTA revenue")
    print(f"    Revenue loss vs Scenario A: "
          f"${c1['mta_revenue_M'] - base_annual[(base_annual['scenario']=='A_flat_9')&(base_annual['year']==2025)].iloc[0]['mta_revenue_M']:+.0f}M")
    
    print(f"\n  Conservative elasticity check (ε = -0.48):")
    conservative_frames = []
    for name, sched in schedules.items():
        df = build_scenario(name+"_cons", baseline_weekly,
                            sched, -0.48)
        conservative_frames.append(df)
    cons_all = pd.concat(conservative_frames, ignore_index=True)
    cons_annual = annual_summary(cons_all)
    b3_cons = cons_annual[
        (cons_annual["scenario"]=="B_rise_15_cons") &
        (cons_annual["year"]==2027)
    ].iloc[0]
    print(f"    B ($15) 2027 with ε=-0.48: "
        f"{b3_cons['hvfhv_trips_M']:.1f}M trips, "
        f"${b3_cons['mta_revenue_M']:.0f}M revenue")
 
    base_all.to_parquet(OUT_DIR / "forecast_base.parquet", index=False)
    sens_all.to_parquet(OUT_DIR / "forecast_2x.parquet",   index=False)
    base_annual.to_csv(OUT_DIR / "forecast_annual.csv",    index=False)
 
    print(f"\n{'='*55}")
    print(f"  PHASE 5 SUMMARY")
    print(f"{'='*55}")
    print(f"\n  Scenario A ($9 flat):")
    for _, row in base_annual[
        base_annual["scenario"]=="A_flat_9"
    ].iterrows():
        print(f"    {int(row['year'])}: "
              f"{row['hvfhv_trips_M']:.1f}M trips, "
              f"${row['mta_revenue_M']:.0f}M revenue")
 
    print(f"\n  Scenario B (rise to $15 in 2027):")
    for _, row in base_annual[
        base_annual["scenario"]=="B_rise_15"
    ].iterrows():
        print(f"    {int(row['year'])}: "
              f"{row['hvfhv_trips_M']:.1f}M trips, "
              f"${row['mta_revenue_M']:.0f}M revenue")
 
    print(f"\n  Scenario C (6-month pause):")
    for _, row in base_annual[
        base_annual["scenario"]=="C_pause"
    ].iterrows():
        print(f"    {int(row['year'])}: "
              f"{row['hvfhv_trips_M']:.1f}M trips, "
              f"${row['mta_revenue_M']:.0f}M revenue")
 
    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()