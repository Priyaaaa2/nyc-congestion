import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import pathlib
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
FHVHV_DIR  = pathlib.Path(config.TLC_FHVHV_DIR)
PANEL_PATH = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
OUT_DIR    = pathlib.Path(config.DATA_PROC) / "phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
LYFT_LIC = "HV0005"
UBER_LIC = "HV0003"
 
CRZ_ZONES = tuple(config.CRZ_ZONES)
 
 
def load_platform_weekly() -> pd.DataFrame:
    """
    Query raw FHVHV Parquet files to get weekly CRZ trip counts
    split by platform (Lyft vs Uber).
    Uses 2023, 2024, and Jan-Jun 2025 files.
    """
    print("Loading platform-level weekly trips from raw Parquet...")
    import duckdb
    con = duckdb.connect()
 
    files = sorted(FHVHV_DIR.glob("*.parquet"))

    files = [f for f in files if any(
        f.name.startswith(f"fhvhv_tripdata_{y}")
        for y in ["2023","2024","2025"]
    )]
    print(f"  Files: {len(files)}")
 
    file_list = str([str(f) for f in files]).replace("'",'"')
 
    query = f"""
        SELECT
            DATE_TRUNC('week', pickup_datetime::TIMESTAMP) AS week_start,
            CASE
                WHEN hvfhs_license_num = '{LYFT_LIC}' THEN 'lyft'
                WHEN hvfhs_license_num = '{UBER_LIC}' THEN 'uber'
                ELSE 'other'
            END AS platform,
            COUNT(*) AS trip_count
        FROM read_parquet({file_list}, union_by_name=true)
        WHERE PULocationID IN {CRZ_ZONES}
          AND pickup_datetime >= '2023-01-01'
          AND pickup_datetime <  '2025-07-01'
          AND base_passenger_fare > 0
          AND hvfhs_license_num IN ('{LYFT_LIC}', '{UBER_LIC}')
        GROUP BY
            DATE_TRUNC('week', pickup_datetime::TIMESTAMP),
            CASE
                WHEN hvfhs_license_num = '{LYFT_LIC}' THEN 'lyft'
                WHEN hvfhs_license_num = '{UBER_LIC}' THEN 'uber'
                ELSE 'other'
            END
        ORDER BY week_start, platform
    """
    df = con.execute(query).df()
    con.close()
 
    df["week_start"] = pd.to_datetime(df["week_start"])

    df = df[df["week_start"] < "2025-06-30"]

    df = df[df["platform"].isin(["lyft","uber"])]
 
    print(f"  Rows: {len(df):,}")
    print(f"  Weeks: {df['week_start'].nunique()}")
 
    jan25 = df[
        (df["week_start"] >= "2025-01-01") &
        (df["week_start"] <  "2025-02-01")
    ].groupby("platform")["trip_count"].sum()
    print(f"\n  Jan 2025 CRZ trips:")
    print(f"    Lyft (HV0005): {jan25.get('lyft',0):>10,.0f}")
    print(f"    Uber (HV0003): {jan25.get('uber',0):>10,.0f}")
 
    return df
 
 
def test_parallel_trends(df: pd.DataFrame) -> float:
    print("\n" + "="*55)
    print("  PARALLEL TRENDS TEST (2023-2024 pre-period)")
    print("="*55)
 
    pre = df[df["week_start"] < "2025-01-01"].copy()
 
    wide = pre.pivot_table(
        index="week_start",
        columns="platform",
        values="trip_count"
    ).reset_index().dropna()
 
    if len(wide) < 10:
        print("  Not enough pre-period data")
        return np.nan
 
    wide["lyft_chg"] = wide["lyft"].pct_change()
    wide["uber_chg"] = wide["uber"].pct_change()
    wide = wide.dropna()
 
    corr = wide["lyft_chg"].corr(wide["uber_chg"])
    print(f"\n  WoW correlation (Lyft vs Uber): {corr:.3f}")
 
    if corr > 0.7:
        print(f"  ✅ Strong parallel trends — DiD assumption holds")
    elif corr > 0.5:
        print(f"  ⚠️  Moderate parallel trends — DiD reasonable")
    else:
        print(f"  ❌ Weak parallel trends — interpret with caution")
 
    wide["week_num"] = range(len(wide))
    lyft_trend = np.polyfit(wide["week_num"], wide["lyft"], 1)[0]
    uber_trend = np.polyfit(wide["week_num"], wide["uber"], 1)[0]
    print(f"\n  Pre-period weekly trend:")
    print(f"    Lyft: {lyft_trend:+.0f} trips/week")
    print(f"    Uber: {uber_trend:+.0f} trips/week")

    print(f"\n  {'Week':<12} {'Lyft':>10} {'Uber':>10} {'Ratio':>8}")
    print(f"  {'-'*42}")
    for _, row in wide.sort_values("week_start").tail(12).iterrows():
        ratio = row["lyft"] / row["uber"] if row["uber"] > 0 else 0
        print(f"  {str(row['week_start'].date()):<12} "
              f"{row['lyft']:>10,.0f} "
              f"{row['uber']:>10,.0f} "
              f"{ratio:>8.3f}")
 
    return corr
 
 
def run_did(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  DIFFERENCE-IN-DIFFERENCES")
    print("  Treated = Lyft (got $1.50 credit Jan 2025)")
    print("  Control = Uber (no credit)")
    print("="*55)
 
    df = df.copy()
    df["post"]     = (df["week_start"] >= "2025-01-05").astype(int)
    df["is_lyft"]  = (df["platform"] == "lyft").astype(int)
    df["did"]      = df["post"] * df["is_lyft"]
    df["log_trips"] = np.log1p(df["trip_count"])
    df["month"]    = df["week_start"].dt.month
    df["year"]     = df["week_start"].dt.year
    df["week_num"] = (
        (df["week_start"] - df["week_start"].min()).dt.days // 7
    )

    df["lyft_credit"] = (
        (df["week_start"] >= config.LYFT_CREDIT_START) &
        (df["week_start"] <= config.LYFT_CREDIT_END) &
        (df["is_lyft"] == 1)
    ).astype(int)
 
    print(f"\n  Sample sizes:")
    print(df.groupby(["platform","post"])["trip_count"].agg(
        weeks=("count"), avg_trips=("mean")
    ).round(0).to_string())
 
    formula = "log_trips ~ did + is_lyft + post + C(month)"
    model   = smf.ols(formula, data=df).fit(cov_type="HC3")
 
    tau    = model.params["did"]
    tau_se = model.bse["did"]
    tau_p  = model.pvalues["did"]
    tau_ci = model.conf_int().loc["did"]
 
    print(f"\n  DiD estimate (τ):")
    print(f"    Coefficient : {tau:+.4f}")
    print(f"    Std error   : {tau_se:.4f}")
    print(f"    95% CI      : [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]")
    print(f"    p-value     : {tau_p:.4f}")
    print(f"    R²          : {model.rsquared:.3f}")
    sig = "✅ significant" if tau_p < 0.05 else "❌ not significant"
    print(f"    Result      : {sig}")
 
    print(f"\n  Interpretation:")
    direction = "MORE" if tau > 0 else "FEWER"
    print(f"    Lyft had {direction} trips than Uber in Jan 2025")
    print(f"    relative to their pre-period difference")
    print(f"    consistent with Lyft credit boosting demand")
 
    return model, tau, tau_se
 
 
def compute_elasticity(tau: float, tau_se: float) -> tuple:
    print("\n" + "="*55)
    print("  PRICE ELASTICITY OF DEMAND")
    print("="*55)
 
    panel = pd.read_parquet(PANEL_PATH)
    pre   = panel[
        (panel["vehicle_type"] == "fhvhv") &
        (panel["in_crz"] == 1) &
        (panel["week_start"] < "2025-01-01")
    ]
    avg_fare = pre["avg_fare"].mean()
 
    subsidy       = 1.50
    price_control = avg_fare             
    price_treated = avg_fare - subsidy   
 
    delta_log_p   = np.log(price_control / price_treated)
 
    print(f"\n  Average pre-treatment CRZ fare : ${avg_fare:.2f}")
    print(f"  Lyft effective price           : ${price_treated:.2f}")
    print(f"  Uber price                     : ${price_control:.2f}")
    print(f"  Δlog(P)                        : {delta_log_p:.4f}")
 
    elasticity = tau / delta_log_p
    elas_lo    = (tau - 1.96 * tau_se) / delta_log_p
    elas_hi    = (tau + 1.96 * tau_se) / delta_log_p
 
    print(f"\n  Price elasticity ε̂:")
    print(f"    Point estimate : {elasticity:.3f}")
    print(f"    95% CI         : [{elas_lo:.3f}, {elas_hi:.3f}]")
    print(f"\n  Benchmark comparison:")
    print(f"    Chicago GTT (Zheng et al.)  : -0.48")
    print(f"    NYC 2019 surcharge          : ~-0.30")
    print(f"    Your estimate               : {elasticity:.3f}")
 
    if abs(elasticity) < 1:
        print(f"\n  → Demand is INELASTIC")
        print(f"    10% price increase → "
              f"{abs(elasticity)*10:.1f}% trip decrease")
    else:
        print(f"\n  → Demand is ELASTIC")
        print(f"    10% price increase → "
              f"{abs(elasticity)*10:.1f}% trip decrease")
        
    print(f"\n  ⚠️  NOTE ON INTERPRETATION:")
    print(f"  This elasticity measures platform substitution")
    print(f"  (riders switching Lyft↔Uber), not aggregate demand.")
    print(f"  For Phase 5 scenario model, use Phase 1 implied ε:")
    phase1_effect     = -0.070   # FHVHV fell 7.0% vs counterfactual
    price_change_pct  = subsidy / avg_fare  # 1.50 / 27.15
    implied_elas      = phase1_effect / price_change_pct
    print(f"    Phase 1 effect   : {phase1_effect*100:.1f}%")
    print(f"    Price change     : +{price_change_pct*100:.1f}%")
    print(f"    Implied ε        : {implied_elas:.3f}")
    print(f"    Chicago benchmark: -0.48")
    print(f"    → Using {implied_elas:.3f} for Phase 5 forecast")
 
    return elasticity, elas_lo, elas_hi
 
 
def main():
    print("\n" + "="*55)
    print("  PHASE 4: LYFT DiD + PRICE ELASTICITY (v2)")
    print("  Platform ID via hvfhs_license_num")
    print("="*55)
 
    df             = load_platform_weekly()
    corr           = test_parallel_trends(df)
    model, tau, se = run_did(df)
    elas, lo, hi   = compute_elasticity(tau, se)
 
    results = pd.DataFrame([{
        "tau_did":          tau,
        "tau_se":           se,
        "elasticity_did":   elas,      # platform substitution
        "elasticity_lo":    lo,
        "elasticity_hi":    hi,
        "elasticity_phase1": -0.070 / (1.50 / 27.15),  # for Phase 5
        "parallel_corr":    corr,
    }])
    results.to_parquet(OUT_DIR / "did_results.parquet", index=False)

    df.to_parquet(OUT_DIR / "platform_weekly.parquet", index=False)
 
    print(f"\n{'='*55}")
    print(f"  PHASE 4 SUMMARY")
    print(f"{'='*55}")
    print(f"\n  DiD τ̂        : {tau:+.4f} (SE: {se:.4f})")
    print(f"  Elasticity ε̂ : {elas:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"\n  This elasticity feeds Phase 5 scenario model.")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()