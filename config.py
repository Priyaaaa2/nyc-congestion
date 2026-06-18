DATA_RAW = "data/raw/"
DATA_PROC = "data/processed/"
OUTPUTS = "outputs/"

CRZ_ZONES = [
    4, 12, 13, 24, 41, 42, 43, 45, 48,
    50, 68, 79, 87, 88, 90, 100, 103,
    107, 113, 114, 116, 120, 125, 127,
    128, 137, 140, 141, 142, 143, 144,
    148, 151, 152, 153, 158, 161, 162,
    163, 164, 166, 170, 186, 194, 202,
    209, 211, 224, 229, 230, 231, 232,
    233, 234, 236, 237, 238, 239, 243,
    244, 246, 249, 261, 262, 263,
]

TREATMENT_DATE = "2025-01-05"
LYFT_CREDIT_START = "2025-01-01"
LYFT_CREDIT_END = "2025-01-31"

# --- paths ---
TLC_FHVHV_DIR  = DATA_RAW + "tlc/fhvhv/"
TLC_YELLOW_DIR = DATA_RAW + "tlc/yellow/"
TLC_ZONES_SHP  = DATA_RAW + "tlc/taxi_zones/taxi_zones/taxi_zones.shp"
TLC_ZONE_CSV   = DATA_RAW + "tlc/zone_lookup.csv"
MTA_FULL       = DATA_RAW + "mta/subway_full.parquet"
MTA_STATIONS   = DATA_RAW + "mta/stations.csv"
CENSUS_INCOME  = DATA_RAW + "census/income_zcta_clean.csv"
CENSUS_COMMUTE = DATA_RAW + "census/commute_zcta_clean.csv"
CENSUS_ZCTA    = DATA_RAW + "census/zcta_shapefile/tl_2023_us_zcta520.shp"
WEATHER        = DATA_RAW + "weather/central_park.csv"
DOT_2024       = DATA_RAW + "dot/speeds_jan2024.parquet"
DOT_2025       = DATA_RAW + "dot/speeds_jan2025.parquet"
