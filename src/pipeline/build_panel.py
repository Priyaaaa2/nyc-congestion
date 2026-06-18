import duckdb
import pandas as pd
import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config

DB_PATH    = pathlib.Path(config.DATA_PROC) / "master.duckdb"
OUT_PANEL  = pathlib.Path(config.DATA_PROC) / "master_panel.parquet"
OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)

INCOME_CSV = pathlib.Path(config.DATA_PROC) / "zone_income.csv"
SUBWAY_CSV = pathlib.Path(config.DATA_PROC) / "zone_subway.csv"
WEATHER_CSV = pathlib.Path(config.WEATHER)


def build_trip_panel(con) -> pd.DataFrame:
    print("\nAggregating trips to zone × week × vehicle_type...")
    print("  (DuckDB scans 509M rows on disk — takes 3-5 min)")

    crz_list = str(config.CRZ_ZONES).replace("[","(").replace("]",")")

    query = f"""
        SELECT
        
            pickup_zone                         AS zone_id,
            week_start,
            vehicle_type,
            COUNT(*)                            AS trip_count,
            AVG(fare)                           AS avg_fare,
            AVG(trip_miles)                     AS avg_miles,
            AVG(surcharge)                      AS avg_surcharge,
            AVG(cbd_fee)                        AS avg_cbd_fee,
            SUM(cbd_fee)                        AS total_cbd_fee,
            SUM(CASE WHEN cbd_fee > 0
                THEN 1 ELSE 0 END)              AS trips_with_cbd,
            CAST(pickup_zone IN {crz_list}
                AS INTEGER)                     AS is_crz_pickup,
            CAST(week_start >= '{config.TREATMENT_DATE}'
                AS INTEGER)                     AS post_treatment,
            CAST(
                week_start >= '{config.LYFT_CREDIT_START}'
                AND week_start <= '{config.LYFT_CREDIT_END}'
                AS INTEGER)                     AS lyft_credit_period
        FROM trips
        GROUP BY pickup_zone, week_start, vehicle_type
        ORDER BY week_start, pickup_zone, vehicle_type
    """

    panel = con.execute(query).df()
    print(f"  Done: {len(panel):,} zone-week-type rows")
    print(f"  Weeks : {panel['week_start'].nunique()}")
    print(f"  Zones : {panel['zone_id'].nunique()}")
    return panel


def merge_income(panel: pd.DataFrame) -> pd.DataFrame:
    print("\nMerging zone income...")
    income = pd.read_csv(INCOME_CSV)
    income = income[["zone_id", "median_income", "Borough", "in_crz"]]
    panel  = panel.merge(income, on="zone_id", how="left")
    missing = panel["median_income"].isna().sum()
    print(f"  Merged. Missing income: {missing:,} rows "
          f"({missing/len(panel):.1%}) — airports/industrial zones")
    return panel


def merge_subway(panel: pd.DataFrame) -> pd.DataFrame:
    print("\nMerging subway access...")
    subway = pd.read_csv(SUBWAY_CSV)
    subway = subway[["zone_id", "nearest_station",
                     "dist_m", "has_subway"]]
    panel  = panel.merge(subway, on="zone_id", how="left")
    pct    = panel["has_subway"].mean()
    print(f"  Merged. Zones with subway: {pct:.1%} of rows")
    return panel


def merge_weather(panel: pd.DataFrame) -> pd.DataFrame:
    print("\nMerging weather...")
    wx = pd.read_csv(WEATHER_CSV, parse_dates=["DATE"])

    # NOAA values are in tenths — divide by 10
    for col in ["TMAX", "TMIN", "PRCP"]:
        if col in wx.columns:
            wx[col] = wx[col] / 10.0

    # Aggregate daily → weekly
    wx["week_start"] = wx["DATE"].dt.to_period("W").dt.start_time
    wx_weekly = wx.groupby("week_start").agg(
        avg_tmax   = ("TMAX", "mean"),
        avg_tmin   = ("TMIN", "mean"),
        total_prcp = ("PRCP", "sum"),
        max_snow   = ("SNOW", "max"),
        max_snwd   = ("SNWD", "max"),
    ).reset_index()

    # Snow flag: any snow depth > 50mm that week
    wx_weekly["snow_week"] = (wx_weekly["max_snwd"] > 50).astype(int)

    panel["week_start"] = pd.to_datetime(panel["week_start"])
    panel = panel.merge(wx_weekly, on="week_start", how="left")

    missing_wx = panel["avg_tmax"].isna().sum()
    print(f"  Merged. Missing weather rows: {missing_wx:,}")
    return panel


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    print("\nAdding time features...")
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    panel["year"]       = panel["week_start"].dt.year
    panel["month"]      = panel["week_start"].dt.month
    panel["quarter"]    = panel["week_start"].dt.quarter

    # Ordinal week number from start of dataset (for trend)
    min_week      = panel["week_start"].min()
    panel["week_num"] = (
        (panel["week_start"] - min_week).dt.days // 7
    ).astype(int)

    # Holiday flag (major US holidays)
    holidays = pd.to_datetime([
        "2022-01-17","2022-02-21","2022-05-30","2022-07-04",
        "2022-09-05","2022-11-24","2022-12-26",
        "2023-01-02","2023-01-16","2023-02-20","2023-05-29",
        "2023-07-04","2023-09-04","2023-11-23","2023-12-25",
        "2024-01-01","2024-01-15","2024-02-19","2024-05-27",
        "2024-07-04","2024-09-02","2024-11-28","2024-12-25",
        "2025-01-01","2025-01-20","2025-02-17","2025-05-26",
        "2025-07-04",
    ])
    holiday_weeks = pd.to_datetime([
        h.to_period("W").start_time for h in holidays
    ])
    panel["holiday_week"] = panel["week_start"].isin(
        holiday_weeks
    ).astype(int)

    print(f"  week_num range: {panel['week_num'].min()} "
          f"to {panel['week_num'].max()}")
    print(f"  Holiday weeks flagged: "
          f"{panel['holiday_week'].sum():,} rows")
    return panel


def validate(panel: pd.DataFrame):
    print("\n" + "="*55)
    print("  VALIDATION")
    print("="*55)

    print(f"\nShape          : {panel.shape}")
    print(f"Columns        : {panel.columns.tolist()}")
    print(f"\nDate range     : {panel['week_start'].min()} "
          f"to {panel['week_start'].max()}")
    print(f"Unique weeks   : {panel['week_start'].nunique()}")
    print(f"Unique zones   : {panel['zone_id'].nunique()}")

    print(f"\nVehicle types:")
    print(panel.groupby("vehicle_type")["trip_count"].agg(
        ["sum","mean","count"]
    ).round(1).to_string())

    print(f"\nPre vs post treatment:")
    print(panel.groupby(["vehicle_type","post_treatment"])[
        "trip_count"
    ].sum().reset_index().to_string(index=False))

    print(f"\nCBD fee rows (2025 only):")
    print(panel[panel["trips_with_cbd"]>0].groupby(
        "vehicle_type"
    )["trips_with_cbd"].sum().to_string())

    print(f"\nNull check (should all be low):")
    nulls = panel.isnull().sum()
    nulls = nulls[nulls > 0]
    print(nulls.to_string() if len(nulls) else "  No nulls ✅")


def main():
    print("\n" + "="*55)
    print("  MASTER PANEL BUILDER")
    print("="*55)

    con   = duckdb.connect(str(DB_PATH), read_only=True)
    panel = build_trip_panel(con)
    con.close()

    panel = merge_income(panel)
    panel = merge_subway(panel)
    panel = merge_weather(panel)
    panel = add_time_features(panel)

    validate(panel)

    panel.to_parquet(OUT_PANEL, index=False)
    size = OUT_PANEL.stat().st_size / 1024 / 1024

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Rows    : {len(panel):,}")
    print(f"  Columns : {len(panel.columns)}")
    print(f"  Size    : {size:.1f} MB")
    print(f"  Path    : {OUT_PANEL}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()