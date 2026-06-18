import duckdb
import pathlib
import sys
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
DB_PATH      = pathlib.Path(config.DATA_PROC) / "master.duckdb"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
 
FHVHV_FILES  = sorted(pathlib.Path(config.TLC_FHVHV_DIR).glob("*.parquet"))
YELLOW_FILES = sorted(pathlib.Path(config.TLC_YELLOW_DIR).glob("*.parquet"))
 
# CRZ zone list as SQL array literal
CRZ_SQL = str(config.CRZ_ZONES).replace("[", "[").replace("]", "]")
 
print(f"FHVHV files  : {len(FHVHV_FILES)}")
print(f"Yellow files : {len(YELLOW_FILES)}")
print(f"CRZ zones    : {len(config.CRZ_ZONES)}")
 
 
def has_column(con, path: str, col: str) -> bool:
    try:
        cols = con.execute(
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 1)"
        ).df()["column_name"].tolist()
        return col in cols
    except Exception:
        return False
 
 
def build_fhvhv_query(path: str, has_cbd: bool) -> str:
    cbd_expr = ("COALESCE(cbd_congestion_fee, 0)::DOUBLE"
                if has_cbd else "0.0::DOUBLE")
    return f"""
        SELECT
            'fhvhv'                                   AS vehicle_type,
            pickup_datetime::TIMESTAMP                AS pickup_ts,
            dropoff_datetime::TIMESTAMP               AS dropoff_ts,
            PULocationID::INTEGER                     AS pickup_zone,
            DOLocationID::INTEGER                     AS dropoff_zone,
            base_passenger_fare::DOUBLE               AS fare,
            COALESCE(congestion_surcharge, 0)::DOUBLE AS surcharge,
            {cbd_expr}                                AS cbd_fee,
            trip_miles::DOUBLE                        AS trip_miles,
            DATE_TRUNC('week', pickup_datetime)       AS week_start
        FROM read_parquet('{path}', union_by_name=true)
        WHERE PULocationID IS NOT NULL
          AND DOLocationID IS NOT NULL
          AND base_passenger_fare > 0
          AND pickup_datetime >= '2022-01-01'
          AND pickup_datetime <  '2025-07-01'
          AND EPOCH(dropoff_datetime) - EPOCH(pickup_datetime) > 30
          AND (
            PULocationID IN {tuple(config.CRZ_ZONES)}
            OR
            DOLocationID IN {tuple(config.CRZ_ZONES)}
          )
    """
 
 
def build_yellow_query(path: str, has_cbd: bool) -> str:
    cbd_expr = ("COALESCE(cbd_congestion_fee, 0)::DOUBLE"
                if has_cbd else "0.0::DOUBLE")
    return f"""
        SELECT
            'yellow'                                        AS vehicle_type,
            tpep_pickup_datetime::TIMESTAMP                 AS pickup_ts,
            tpep_dropoff_datetime::TIMESTAMP                AS dropoff_ts,
            PULocationID::INTEGER                           AS pickup_zone,
            DOLocationID::INTEGER                           AS dropoff_zone,
            fare_amount::DOUBLE                             AS fare,
            COALESCE(congestion_surcharge, 0)::DOUBLE       AS surcharge,
            {cbd_expr}                                      AS cbd_fee,
            trip_distance::DOUBLE                           AS trip_miles,
            DATE_TRUNC('week', tpep_pickup_datetime)        AS week_start
        FROM read_parquet('{path}', union_by_name=true)
        WHERE PULocationID IS NOT NULL
          AND DOLocationID IS NOT NULL
          AND fare_amount > 0
          AND tpep_pickup_datetime >= '2022-01-01'
          AND tpep_pickup_datetime <  '2025-07-01'
          AND EPOCH(tpep_dropoff_datetime) -
              EPOCH(tpep_pickup_datetime) > 30
          AND (
            PULocationID IN {tuple(config.CRZ_ZONES)}
            OR
            DOLocationID IN {tuple(config.CRZ_ZONES)}
          )
    """
 
 
def ingest_files(con, files, query_fn, label):
    total_rows  = 0
    table_ready = False
    skipped     = []
 
    for i, f in enumerate(files, 1):
        path    = str(f)
        has_cbd = has_column(con, path, "cbd_congestion_fee")
        q       = query_fn(path, has_cbd)
 
        try:
            if not table_ready:
                con.execute(f"CREATE TABLE IF NOT EXISTS trips AS {q}")
                table_ready = True
            else:
                con.execute(f"INSERT INTO trips {q}")
 
            n = con.execute(
                f"SELECT COUNT(*) FROM ({q})"
            ).fetchone()[0]
            total_rows += n
            tag = " [+cbd]" if has_cbd else ""
            print(f"  [{i:02d}/{len(files)}] {f.name}{tag} — "
                  f"{n:>8,} rows  (total: {total_rows:,})")
 
        except Exception as e:
            print(f"  [{i:02d}/{len(files)}] ERROR {f.name}: {e}")
            skipped.append(f.name)
 
    if skipped:
        print(f"\n  Skipped {len(skipped)} files: {skipped}")
    return total_rows
 
 
def main():
    print("\n" + "="*60)
    print("  TLC INGESTION PIPELINE v3 — CRZ trips only")
    print("="*60)
 
    # Delete old DB to free space first
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Deleted old DB — freed space")
 
    con = duckdb.connect(str(DB_PATH))
 
    # Limit DuckDB temp space usage
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    con.execute("SET memory_limit='4GB'")
 
    print(f"\nIngesting FHVHV (Uber/Lyft) — CRZ trips only...")
    fhvhv_rows = ingest_files(con, FHVHV_FILES, build_fhvhv_query, "fhvhv")
 
    print(f"\nIngesting Yellow Taxi — CRZ trips only...")
    yellow_rows = ingest_files(con, YELLOW_FILES, build_yellow_query, "yellow")
 
    print("\nValidation:")
    summary = con.execute("""
        SELECT
            vehicle_type,
            COUNT(*)               AS total_rows,
            MIN(pickup_ts)         AS earliest,
            MAX(pickup_ts)         AS latest,
            ROUND(AVG(fare), 2)    AS avg_fare,
            SUM(CASE WHEN cbd_fee > 0
                THEN 1 ELSE 0 END) AS rows_with_cbd_fee
        FROM trips
        GROUP BY vehicle_type
        ORDER BY vehicle_type
    """).df()
    print(summary.to_string(index=False))
 
    total = fhvhv_rows + yellow_rows
    size  = DB_PATH.stat().st_size / 1024 / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"Total rows : {total:,}")
    print(f"DB size    : {size:.1f} GB")
    print(f"DB path    : {DB_PATH}")
    print(f"{'='*60}\n")
 
    con.close()
 
 
if __name__ == "__main__":
    main()