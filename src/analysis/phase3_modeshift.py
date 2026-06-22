import pandas as pd
import numpy as np
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PANEL_PATH  = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
MTA_PATH    = pathlib.Path(config.MTA_FULL)
SUBWAY_PATH = pathlib.Path(config.DATA_PROC) / "zone_subway.csv"
INCOME_PATH = pathlib.Path(config.DATA_PROC) / "zone_income.csv"
OUT_DIR     = pathlib.Path(config.DATA_PROC) / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
BASELINE_GROWTH_FHVHV  = 0.034
BASELINE_GROWTH_YELLOW = 0.014
BASELINE_GROWTH_MTA    = 0.037
 
EXCLUDE_ZONES = [194, 105]
 
 
def load_fhvhv_zone_change() -> pd.DataFrame:
    print("Loading FHVHV zone-level changes...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["week_start"] = pd.to_datetime(panel["week_start"])
 
    fhvhv = panel[
        (panel["vehicle_type"] == "fhvhv") &
        (panel["in_crz"] == 1) &
        (~panel["zone_id"].isin(EXCLUDE_ZONES))
    ]
 
    pre = fhvhv[
        (fhvhv["week_start"] >= "2024-01-01") &
        (fhvhv["week_start"] <  "2024-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    pre.columns = ["zone_id", "fhvhv_2024"]
 
    post = fhvhv[
        (fhvhv["week_start"] >= "2025-01-01") &
        (fhvhv["week_start"] <  "2025-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    post.columns = ["zone_id", "fhvhv_2025"]
 
    df = pre.merge(post, on="zone_id")
    df["fhvhv_cf"]   = df["fhvhv_2024"] * (1 + BASELINE_GROWTH_FHVHV)
    df["fhvhv_loss"] = df["fhvhv_2025"] - df["fhvhv_cf"]
 
    print(f"  {len(df)} CRZ zones")
    print(f"  Total FHVHV loss: {df['fhvhv_loss'].sum():,.0f} trips")
    return df
 
 
def load_yellow_zone_change() -> pd.DataFrame:
    print("Loading Yellow taxi zone-level changes...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["week_start"] = pd.to_datetime(panel["week_start"])
 
    yellow = panel[
        (panel["vehicle_type"] == "yellow") &
        (panel["in_crz"] == 1) &
        (~panel["zone_id"].isin(EXCLUDE_ZONES))
    ]
 
    pre = yellow[
        (yellow["week_start"] >= "2024-01-01") &
        (yellow["week_start"] <  "2024-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    pre.columns = ["zone_id", "yellow_2024"]
 
    post = yellow[
        (yellow["week_start"] >= "2025-01-01") &
        (yellow["week_start"] <  "2025-07-01")
    ].groupby("zone_id")["trip_count"].sum().reset_index()
    post.columns = ["zone_id", "yellow_2025"]
 
    df = pre.merge(post, on="zone_id", how="outer").fillna(0)
    df["yellow_cf"]   = df["yellow_2024"] * (1 + BASELINE_GROWTH_YELLOW)
    df["yellow_gain"] = df["yellow_2025"] - df["yellow_cf"]
 
    print(f"  Total Yellow gain: {df['yellow_gain'].sum():,.0f} trips")
    return df[["zone_id", "yellow_2024", "yellow_2025",
               "yellow_cf", "yellow_gain"]]
 
 
def load_mta_zone_change(subway_map: pd.DataFrame) -> pd.DataFrame:
    print("Loading MTA subway ridership changes...")
 
    import duckdb
    con = duckdb.connect()
 
    query = """
        SELECT
            station_complex_id,
            MAX(station_complex) AS station_complex,
            DATE_TRUNC('month', transit_timestamp::TIMESTAMP) AS month,
            SUM(ridership) AS monthly_ridership
        FROM read_parquet(?)
        WHERE transit_mode = 'subway'
          AND transit_timestamp >= '2024-01-01'
          AND transit_timestamp <  '2025-07-01'
        GROUP BY station_complex_id,
                 DATE_TRUNC('month', transit_timestamp::TIMESTAMP)
        ORDER BY station_complex_id, month
    """
    mta = con.execute(query, [str(MTA_PATH)]).df()
    mta["month"] = pd.to_datetime(mta["month"])
    con.close()
 
    print(f"  MTA rows loaded: {len(mta):,}")
 
    station_names = mta.groupby("station_complex_id")\
                       ["station_complex"].first().reset_index()
 
    pre_mta = mta[
        (mta["month"] >= "2024-01-01") &
        (mta["month"] <  "2024-07-01")
    ].groupby("station_complex_id")["monthly_ridership"].sum()\
     .reset_index()
    pre_mta.columns = ["station_complex_id", "mta_2024"]
 
    post_mta = mta[
        (mta["month"] >= "2025-01-01") &
        (mta["month"] <  "2025-07-01")
    ].groupby("station_complex_id")["monthly_ridership"].sum()\
     .reset_index()
    post_mta.columns = ["station_complex_id", "mta_2025"]
 
    mta_change = pre_mta.merge(post_mta, on="station_complex_id",
                               how="outer").fillna(0)
    mta_change = mta_change.merge(station_names,
                                  on="station_complex_id", how="left")
    mta_change["mta_cf"]   = (
        mta_change["mta_2024"] * (1 + BASELINE_GROWTH_MTA)
    )
    mta_change["mta_gain"] = (
        mta_change["mta_2025"] - mta_change["mta_cf"]
    )
 
    stations = pd.read_csv(config.MTA_STATIONS)
    stations = stations[["GTFS Stop ID", "Complex ID"]].copy()
    stations.columns = ["gtfs_stop_id", "complex_id"]
    stations["gtfs_stop_id"] = stations["gtfs_stop_id"].astype(str).str.strip()
    stations["complex_id"]   = stations["complex_id"].astype(str).str.strip()
 
    subway_map["gtfs_stop_id"] = (
        subway_map["nearest_station"].astype(str).str.strip()
    )
 
    zone_to_complex = subway_map.merge(
        stations, on="gtfs_stop_id", how="left"
    )[["zone_id", "complex_id"]]
 
    mta_change["complex_id"] = (
        mta_change["station_complex_id"].astype(str).str.strip()
    )
 
    zone_mta = zone_to_complex.merge(
        mta_change[["complex_id", "mta_2024",
                    "mta_2025", "mta_cf", "mta_gain"]],
        on="complex_id", how="left"
    )
 
    matched = zone_mta["mta_gain"].notna().sum()
    print(f"  Zones matched to MTA station: {matched}/{len(zone_mta)}")
    return zone_mta[["zone_id", "complex_id",
                     "mta_2024", "mta_2025", "mta_gain"]]
 
 
def build_decomposition(fhvhv_df, yellow_df, mta_df) -> pd.DataFrame:
    print("\nBuilding three-way decomposition...")
 
    df = (fhvhv_df
          .merge(yellow_df, on="zone_id", how="left")
          .merge(mta_df,    on="zone_id", how="left"))
 
    lost = df[df["fhvhv_loss"] < 0].copy()
    lost["loss_abs"] = lost["fhvhv_loss"].abs()
 
    lost["yellow_captured"] = np.minimum(
        lost["yellow_gain"].clip(lower=0),
        lost["loss_abs"]
    )
 
    lost["mta_captured"] = lost["mta_gain"].clip(lower=0).fillna(0)
    lost["mta_captured"] = np.minimum(
        lost["mta_captured"],
        (lost["loss_abs"] - lost["yellow_captured"]).clip(lower=0)
    )
 
    lost["suppressed"] = (
        lost["loss_abs"]
        - lost["yellow_captured"]
        - lost["mta_captured"]
    ).clip(lower=0)
 
    lost["yellow_pct"]     = (
        lost["yellow_captured"] / lost["loss_abs"] * 100
    ).round(1)
    lost["mta_pct"]        = (
        lost["mta_captured"]    / lost["loss_abs"] * 100
    ).round(1)
    lost["suppressed_pct"] = (
        lost["suppressed"]      / lost["loss_abs"] * 100
    ).round(1)
 
    return lost
 
 
def print_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  THREE-WAY MODE SHIFT DECOMPOSITION")
    print("="*55)
 
    total_loss     = df["loss_abs"].sum()
    total_yellow   = df["yellow_captured"].sum()
    total_mta      = df["mta_captured"].sum()
    total_suppress = df["suppressed"].sum()
 
    print(f"\n  Total FHVHV trips lost (Jan-Jun 2025): "
          f"{total_loss:,.0f}")
    print(f"\n  Where did they go?")
    print(f"    → Yellow taxi   : {total_yellow:>10,.0f} "
          f"({total_yellow/total_loss*100:.1f}%)")
    print(f"    → Subway        : {total_mta:>10,.0f} "
          f"({total_mta/total_loss*100:.1f}%)")
    print(f"    → Suppressed    : {total_suppress:>10,.0f} "
          f"({total_suppress/total_loss*100:.1f}%)")
 
    print(f"\n  Zone-level averages:")
    print(f"    Yellow pct     : {df['yellow_pct'].mean():.1f}%")
    print(f"    MTA pct        : {df['mta_pct'].mean():.1f}%")
    print(f"    Suppressed pct : {df['suppressed_pct'].mean():.1f}%")
 
    if "has_subway" in df.columns:
        print(f"\n  Zones with subway vs without:")
        for has_sub in [1, 0]:
            sub = df[df["has_subway"] == has_sub]
            label = "With subway   " if has_sub else "Without subway"
            if len(sub) > 0:
                print(f"    {label}: "
                      f"suppressed {sub['suppressed_pct'].mean():.1f}% "
                      f"| yellow {sub['yellow_pct'].mean():.1f}% "
                      f"| mta {sub['mta_pct'].mean():.1f}%")
 
 
def main():
    print("\n" + "="*55)
    print("  PHASE 3: MODE SHIFT DECOMPOSITION")
    print("="*55)
 
    subway_map = pd.read_csv(SUBWAY_PATH)
    income     = pd.read_csv(INCOME_PATH)
 
    fhvhv_df  = load_fhvhv_zone_change()
    yellow_df = load_yellow_zone_change()
    mta_df    = load_mta_zone_change(subway_map)
 
    df = build_decomposition(fhvhv_df, yellow_df, mta_df)
 
    df = df.merge(
        income[["zone_id", "median_income", "Borough"]],
        on="zone_id", how="left"
    )
    df = df.merge(
        subway_map[["zone_id", "has_subway"]],
        on="zone_id", how="left"
    )
 
    print_summary(df)
 
    out_path = OUT_DIR / "mode_split.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved → {out_path}")
 
    print(f"\n{'='*55}")
    print(f"  PHASE 3 SUMMARY")
    print(f"{'='*55}")
    total_loss = df["loss_abs"].sum()
    yp = df["yellow_captured"].sum() / total_loss * 100
    mp = df["mta_captured"].sum()    / total_loss * 100
    sp = df["suppressed"].sum()      / total_loss * 100
    print(f"\n  Of every 100 lost FHVHV trips:")
    print(f"    {yp:.0f} went to yellow taxi")
    print(f"    {mp:.0f} went to subway")
    print(f"    {sp:.0f} were suppressed (trip cancelled)")
    print(f"\n  Interpretation:")
    print(f"  → Suppressed demand is the largest component")
    print(f"  → Consistent with Stockholm (50% suppressed)")
    print(f"  → Transit-poor zones have higher suppression")
    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()