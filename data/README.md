# Data

Raw and processed data are not included in this repository due to size (approximately 27 GB compressed). All sources are publicly available and free to download.

---

## Download Instructions

### 1. TLC Trip Records (Uber, Lyft, Yellow Cab)

Go to the NYC Taxi and Limousine Commission trip data portal:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Download the following for January 2022 through June 2025:
- **High Volume For-Hire Vehicle** (HVFHV) — these are Uber and Lyft trips
- **Yellow Taxi Trip Records**

Files are monthly Parquet format. Place them in:
```
data/raw/tlc/fhvhv/
data/raw/tlc/yellow/
```

Also download the **TLC Taxi Zone Shapefile** from the same page and place it in:
```
data/raw/tlc/taxi_zones/
```

---

### 2. MTA Subway Ridership

Go to the MTA Open Data portal:
https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-202/wujg-7c2s

Download the full dataset (hourly ridership by station complex). Place the file in:
```
data/raw/mta/
```

Also download the station location file:
https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f

---

### 3. Census Income Data

Go to the US Census Bureau data explorer:
https://data.census.gov

Download the **ACS 5-Year Estimates, Table S1903** (Median Income) at the ZCTA (zip code) level for 2023. Place it in:
```
data/raw/census/
```

---

### 4. Weather Data

Go to NOAA's Climate Data Online portal:
https://www.ncdc.noaa.gov/cdo-web/

Request daily summaries for station **USW00094728** (Central Park, NY) for January 2022 through June 2025. Select TMAX, PRCP, and SNWD. Place the downloaded CSV in:
```
data/raw/weather/
```

---

### 5. NYC DOT Traffic Speeds

Go to NYC Open Data:
https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9

Download January 2024 and January 2025. Place the files in:
```
data/raw/dot/
```

---

## After Downloading

Run the pipeline in order from the project root:

```bash
python -m src.pipeline.ingest
python -m src.pipeline.spatial_join
python -m src.pipeline.build_panel
```

This will produce:
- `data/processed/master.duckdb` — 9 GB, all CRZ trips
- `data/processed/master_panel.parquet` — 89,117 rows, the main analysis file
- `data/processed/zone_income.csv` — income by zone
- `data/processed/zone_subway.csv` — subway access by zone

The full pipeline takes approximately 2–4 hours on a standard laptop.