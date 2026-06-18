import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PANEL_PATH  = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
INCOME_PATH = pathlib.Path(config.DATA_PROC) / "zone_income.csv"
SUBWAY_PATH = pathlib.Path(config.DATA_PROC) / "zone_subway.csv"
OUT_DIR     = pathlib.Path(config.DATA_PROC) / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def load_zone_panel() -> pd.DataFrame:

    print("Loading master panel...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["week_start"] = pd.to_datetime(panel["week_start"])
 
    # Filter to FHVHV only (main treatment vehicle)
    # CRZ pickup zones only
    fhvhv = panel[
        (panel["vehicle_type"] == "fhvhv") &
        (panel["in_crz"] == 1)
    ].copy()
 
    # Jan-Jun 2024 (pre-treatment comparison period)
    pre = fhvhv[
        (fhvhv["week_start"] >= "2024-01-01") &
        (fhvhv["week_start"] <  "2024-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    pre.columns = ["zone_id", "trips_2024"]
 
    # Jan-Jun 2025 (post-treatment)
    post = fhvhv[
        (fhvhv["week_start"] >= "2025-01-01") &
        (fhvhv["week_start"] <  "2025-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    post.columns = ["zone_id", "trips_2025"]
 
    # Merge
    zone_df = pre.merge(post, on="zone_id", how="inner")
    zone_df = zone_df[zone_df["zone_id"] != 194]
 
    # YoY baseline from Phase 1: FHVHV grew +3.4% pre-toll
    BASELINE_GROWTH = 0.034
    zone_df["cf_trips"] = zone_df["trips_2024"] * (1 + BASELINE_GROWTH)
    zone_df["effect_abs"] = zone_df["trips_2025"] - zone_df["cf_trips"]
    zone_df["effect_pct"] = (
        zone_df["effect_abs"] / zone_df["cf_trips"]
    ) * 100

    EXCLUDE_ZONES = [194, 105]  # Randalls Island, Governor's Island
    zone_df = zone_df[~zone_df["zone_id"].isin(EXCLUDE_ZONES)]
    print(f"  Excluded {len(EXCLUDE_ZONES)} non-residential zones")
    print(f"  Final zones: {len(zone_df)}")
    return zone_df
 
 
def merge_covariates(zone_df: pd.DataFrame) -> pd.DataFrame:
    print("Merging income and subway data...")
 
    income = pd.read_csv(INCOME_PATH)
    subway = pd.read_csv(SUBWAY_PATH)
 
    zone_df = zone_df.merge(
        income[["zone_id", "median_income", "Borough", "in_crz"]],
        on="zone_id", how="left"
    )
    zone_df = zone_df.merge(
        subway[["zone_id", "has_subway", "dist_m"]],
        on="zone_id", how="left"
    )
 
    zone_df["log_income"] = np.log(
        zone_df["median_income"].clip(lower=1)
    )
    zone_df["has_subway"] = zone_df["has_subway"].fillna(0).astype(int)
 
    print(f"  Missing income : {zone_df['median_income'].isna().sum()}")
    print(f"  Has subway     : {zone_df['has_subway'].sum()}/{len(zone_df)}")
    return zone_df
 
 
def run_equity_regression(zone_df: pd.DataFrame):
    print("\n" + "="*55)
    print("  EQUITY REGRESSION")
    print("  effect_pct ~ log(income) + has_subway + borough")
    print("="*55)
 
    df = zone_df.dropna(subset=["median_income", "effect_pct"]).copy()
 
    # Borough fixed effects
    formula = (
        "effect_pct ~ log_income + has_subway + C(Borough)"
    )
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
 
    print(f"\n  N zones  : {int(model.nobs)}")
    print(f"  R²       : {model.rsquared:.3f}")
    print(f"\n  Key coefficients:")
    print(f"  {'Variable':<25} {'Coef':>8} {'p-val':>8} {'Sig':>5}")
    print(f"  {'-'*50}")
 
    for var in model.params.index:
        coef  = model.params[var]
        pval  = model.pvalues[var]
        sig   = ("***" if pval < 0.001 else
                 "**"  if pval < 0.01  else
                 "*"   if pval < 0.05  else "")
        label = var.replace("C(Borough)[T.", "Borough=").replace("]","")
        print(f"  {label:<25} {coef:>8.3f} {pval:>8.4f} {sig:>5}")
 
    print(f"\n  Interpretation:")
    income_coef = model.params.get("log_income", 0)
    subway_coef = model.params.get("has_subway", 0)
    income_pval = model.pvalues.get("log_income", 1)
    subway_pval = model.pvalues.get("has_subway", 1)
 
    direction = "higher" if income_coef > 0 else "lower"
    print(f"  → log_income coef = {income_coef:.3f} "
          f"(p={income_pval:.4f})")
    if income_pval < 0.05:
        print(f"    Higher income zones had {direction} trip losses")
        if income_coef < 0:
            print(f"    ✅ LOW-INCOME zones lost MORE trips — equity concern")
        else:
            print(f"    ℹ️  HIGH-INCOME zones lost more trips")
    else:
        print(f"    No significant income gradient in trip losses")
 
    print(f"\n  → has_subway coef = {subway_coef:.3f} "
          f"(p={subway_pval:.4f})")
    if subway_pval < 0.05:
        sub_dir = "smaller" if subway_coef > 0 else "larger"
        print(f"    Zones WITH subway had {sub_dir} trip losses")
 
    return model
 
 
def income_quintile_summary(zone_df: pd.DataFrame):
    print("\n" + "="*55)
    print("  INCOME QUINTILE BREAKDOWN")
    print("="*55)
 
    df = zone_df.dropna(subset=["median_income"]).copy()
    df["income_quintile"] = pd.qcut(
        df["median_income"], q=5,
        labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
    )
 
    summary = df.groupby("income_quintile", observed=True).agg(
        n_zones        = ("zone_id",     "count"),
        avg_income     = ("median_income","mean"),
        avg_effect_pct = ("effect_pct",  "mean"),
        total_trips_lost = ("effect_abs","sum"),
    ).reset_index()
 
    summary["avg_income"]     = summary["avg_income"].round(0)
    summary["avg_effect_pct"] = summary["avg_effect_pct"].round(1)
    summary["total_trips_lost"]= summary["total_trips_lost"].round(0)
 
    print(f"\n  {'Quintile':<12} {'N zones':>8} "
          f"{'Avg income':>12} {'Avg effect%':>12} "
          f"{'Total lost':>12}")
    print(f"  {'-'*58}")
    for _, row in summary.iterrows():
        q = str(row["income_quintile"]).replace("\n"," ")
        print(f"  {q:<12} {int(row['n_zones']):>8} "
              f"${row['avg_income']:>10,.0f} "
              f"{row['avg_effect_pct']:>11.1f}% "
              f"{row['total_trips_lost']:>12,.0f}")
 
    return summary
 
 
def save_choropleth_data(zone_df: pd.DataFrame):
    out = zone_df[[
        "zone_id", "median_income", "Borough",
        "has_subway", "dist_m",
        "trips_2024", "trips_2025",
        "cf_trips", "effect_abs", "effect_pct"
    ]].copy()
    out_path = OUT_DIR / "zone_equity.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Choropleth data saved → {out_path}")
    return out
 
 
def main():
    print("\n" + "="*55)
    print("  PHASE 2: INCOME EQUITY DECOMPOSITION")
    print("="*55)
 
    zone_df  = load_zone_panel()
    zone_df  = merge_covariates(zone_df)
    model    = run_equity_regression(zone_df)
    quintile = income_quintile_summary(zone_df)
    choro    = save_choropleth_data(zone_df)
 
    print(f"\n{'='*55}")
    print(f"  PHASE 2 SUMMARY")
    print(f"{'='*55}")
    print(f"\n  Overall CRZ FHVHV effect : "
          f"{zone_df['effect_pct'].mean():.1f}%")
    print(f"  Zones with income data   : "
          f"{zone_df['median_income'].notna().sum()}")
    print(f"\n  Top 5 most affected zones:")
    top5 = (zone_df
            .dropna(subset=["median_income"])
            .nsmallest(5, "effect_pct")
            [["zone_id","Borough","median_income",
              "effect_pct","has_subway"]])
    print(top5.to_string(index=False))
 
    print(f"\n  Bottom 5 least affected zones:")
    bot5 = (zone_df
            .dropna(subset=["median_income"])
            .nlargest(5, "effect_pct")
            [["zone_id","Borough","median_income",
              "effect_pct","has_subway"]])
    print(bot5.to_string(index=False))
    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()