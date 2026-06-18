import geopandas as gpd
import pandas as pd
import numpy as np
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config

# ── output paths ─────────────────────────────────────────
OUT_INCOME = pathlib.Path(config.DATA_PROC) / "zone_income.csv"
OUT_SUBWAY = pathlib.Path(config.DATA_PROC) / "zone_subway.csv"
OUT_INCOME.parent.mkdir(parents=True, exist_ok=True)

# ── NY ZCTA prefixes ──────────────────────────────────────
NY_PREFIXES = (
    "100","101","102","103","104","105",
    "106","107","108","109","110","111",
    "112","113","114","116"
)


# ============================================================
# PART 1 — Zone income via spatial overlay
# ============================================================

def build_zone_income():
    print("\n" + "="*55)
    print("  PART 1: TLC zones → Census income")
    print("="*55)

    # ── Load TLC zones ────────────────────────────────────
    print("\nLoading TLC zone shapefile...")
    zones = gpd.read_file(config.TLC_ZONES_SHP)
    zones = zones.to_crs("EPSG:4326")
    print(f"  {len(zones)} zones loaded")

    # ── Load Census ZCTA shapefile ────────────────────────
    print("Loading Census ZCTA shapefile...")
    zcta = gpd.read_file(config.CENSUS_ZCTA)
    zcta = zcta[zcta["ZCTA5CE20"].str.startswith(NY_PREFIXES)].copy()
    zcta = zcta.to_crs("EPSG:4326")
    print(f"  {len(zcta)} NY ZCTAs loaded")

    # ── Load ACS income ───────────────────────────────────
    print("Loading Census income data...")
    income = pd.read_csv(
        config.CENSUS_INCOME,
        dtype={"zcta": str}
    )
    income["median_income"] = pd.to_numeric(
        income["median_income"], errors="coerce"
    )
    # Remove -666666666 (Census code for missing)
    income = income[income["median_income"] > 0]
    print(f"  {len(income)} ZCTAs with valid income")

    # ── Merge income onto ZCTA geometry ──────────────────
    zcta = zcta.merge(
        income, left_on="ZCTA5CE20", right_on="zcta", how="left"
    )
    zcta_with_income = zcta[zcta["median_income"].notna()].copy()
    print(f"  {len(zcta_with_income)} ZCTAs with income + geometry")

    # ── Area-weighted overlay ─────────────────────────────
    print("\nRunning spatial overlay (TLC zones × ZCTAs)...")
    print("  This takes 1-2 minutes...")

    # Project to metres for accurate area calculation
    zones_m = zones.to_crs("EPSG:32618")
    zcta_m  = zcta_with_income.to_crs("EPSG:32618")

    overlay = gpd.overlay(
        zones_m[["LocationID", "geometry"]],
        zcta_m[["ZCTA5CE20", "median_income", "geometry"]],
        how="intersection"
    )
    overlay["piece_area"] = overlay.geometry.area

    # Weighted average income per TLC zone
    overlay["income_x_area"] = (
        overlay["median_income"] * overlay["piece_area"]
    )
    zone_income = (
        overlay
        .groupby("LocationID")
        .apply(lambda g: g["income_x_area"].sum() /
                         g["piece_area"].sum())
        .reset_index()
        .rename(columns={"LocationID": "zone_id", 0: "median_income"})
    )

    # Add borough info
    zone_lookup = pd.read_csv(config.TLC_ZONE_CSV)
    zone_income = zone_income.merge(
        zone_lookup[["LocationID", "Borough", "Zone"]],
        left_on="zone_id", right_on="LocationID", how="left"
    ).drop(columns=["LocationID"])
    zone_income["median_income"] = zone_income["median_income"].round(0)

    # Flag CRZ zones
    zone_income["in_crz"] = zone_income["zone_id"].isin(
        config.CRZ_ZONES
    ).astype(int)

    zone_income.to_csv(OUT_INCOME, index=False)
    print(f"\n  Saved {len(zone_income)} zones → {OUT_INCOME}")

    # Quick sense check
    crz = zone_income[zone_income["in_crz"] == 1]
    print(f"\n  CRZ zone income stats:")
    print(f"    Zones with income : {crz['median_income'].notna().sum()}")
    print(f"    Min income        : ${crz['median_income'].min():,.0f}")
    print(f"    Median income     : ${crz['median_income'].median():,.0f}")
    print(f"    Max income        : ${crz['median_income'].max():,.0f}")

    return zone_income


# ============================================================
# PART 2 — Zone subway access via nearest-neighbour join
# ============================================================

def build_zone_subway():
    print("\n" + "="*55)
    print("  PART 2: TLC zones → nearest subway station")
    print("="*55)

    # ── Load TLC zones ────────────────────────────────────
    print("\nLoading TLC zones...")
    zones = gpd.read_file(config.TLC_ZONES_SHP)

    # ── Load MTA stations ─────────────────────────────────
    print("Loading MTA station data...")
    stations = pd.read_csv(config.MTA_STATIONS)
    print(f"  Columns: {stations.columns.tolist()}")

    # Find lat/lon columns — they vary by dataset version
    lat_col = next((c for c in stations.columns
                    if "lat" in c.lower()), None)
    lon_col = next((c for c in stations.columns
                    if "lon" in c.lower()), None)

    if not lat_col or not lon_col:
        print(f"  ERROR: Cannot find lat/lon columns")
        print(f"  Available: {stations.columns.tolist()}")
        return None

    print(f"  Using lat={lat_col}, lon={lon_col}")

    stations = stations.dropna(subset=[lat_col, lon_col])
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations[lon_col], stations[lat_col]
        ),
        crs="EPSG:4326"
    )
    print(f"  {len(stations_gdf)} stations loaded")

    # ── Project to metres ─────────────────────────────────
    zones_m    = zones.to_crs("EPSG:32618")
    stations_m = stations_gdf.to_crs("EPSG:32618")

    # Use zone centroids for distance calculation
    zone_centroids = zones_m.copy()
    zone_centroids["geometry"] = zones_m.centroid

    # ── Nearest station within 400m ───────────────────────
    print("\nFinding nearest subway station per zone (400m max)...")
    joined = gpd.sjoin_nearest(
        zone_centroids[["LocationID", "geometry"]],
        stations_m[["geometry",
                     stations.columns[0],   # station name col
                     lat_col, lon_col]],
        how="left",
        max_distance=400,
        distance_col="dist_to_station_m"
    )

    # Keep only closest match per zone
    joined = (
        joined
        .sort_values("dist_to_station_m")
        .groupby("LocationID")
        .first()
        .reset_index()
    )

    joined["has_subway"] = joined["dist_to_station_m"].notna().astype(int)
    joined["dist_to_station_m"] = joined["dist_to_station_m"].round(1)

    # Rename station name column
    name_col = stations.columns[0]
    joined = joined.rename(columns={
        "LocationID":         "zone_id",
        name_col:             "nearest_station",
        "dist_to_station_m":  "dist_m"
    })

    out_cols = ["zone_id", "nearest_station", "dist_m", "has_subway"]
    out_cols = [c for c in out_cols if c in joined.columns]
    zone_subway = joined[out_cols].copy()

    zone_subway.to_csv(OUT_SUBWAY, index=False)
    print(f"  Saved {len(zone_subway)} zones → {OUT_SUBWAY}")

    # Summary
    total      = len(zone_subway)
    with_sub   = zone_subway["has_subway"].sum()
    without    = total - with_sub
    crz_sub    = zone_subway[
        zone_subway["zone_id"].isin(config.CRZ_ZONES)
    ]["has_subway"].sum()

    print(f"\n  Subway access summary:")
    print(f"    All zones with subway   : {with_sub}/{total}")
    print(f"    All zones without       : {without}/{total}")
    print(f"    CRZ zones with subway   : {crz_sub}/{len(config.CRZ_ZONES)}")
    print(f"    Unmatched fraction      : "
          f"{without/total:.1%} (transit-poor zones)")

    return zone_subway


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*55)
    print("  SPATIAL JOIN PIPELINE")
    print("="*55)

    zone_income = build_zone_income()
    zone_subway = build_zone_subway()

    print("\n" + "="*55)
    print("  DONE")
    print(f"  zone_income.csv : {OUT_INCOME}")
    print(f"  zone_subway.csv : {OUT_SUBWAY}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()