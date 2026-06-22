# The $9 Shock: What NYC's Congestion Pricing Did to Ride-Hailing

New York City launched the first congestion pricing scheme in the United States on January 5, 2025. Every Uber and Lyft trip in lower Manhattan now carries a $1.50 surcharge. This project uses half a billion trip records to find out what actually happened — who lost access, where those trips went, and what it means for the MTA's plans to raise the toll further.

---

## What We Found

**Ride-hailing trips fell 7%.** Uber and Lyft trips in the tolled zone dropped about 157,000 per week compared to what we would have expected without the toll. Yellow cab trips rose 9.5% over the same period — because yellow cabs only face a $0.75 surcharge, they became relatively cheaper.

**Lower-income neighbourhoods were hit harder.** After accounting for location and subway access, zones with lower median incomes lost proportionally more ride-hail trips. But the story is more nuanced than it first appears — see below.

**Transit access matters more than income.** This is the central finding. Of the roughly 4.8 million lost trips in the first half of 2025:
- 40% shifted to the subway
- 24% switched to yellow cabs
- 36% simply did not happen

In neighbourhoods with a subway station nearby, only 17% of lost trips disappeared entirely. In neighbourhoods without nearby subway access, that figure was 75%. A **58 percentage-point gap** — driven entirely by whether or not people had a transit alternative.

**Raising the toll to $15 in 2027 is a gamble.** Under our demand estimates, it would cost the MTA $110 million a year in surcharge revenue. Under a more conservative estimate from Chicago, it would earn $20 million. The uncertainty alone spans $130 million. The MTA should commission a proper elasticity study before deciding.

---

## Why It Matters

The obvious policy response to equity concerns is to discount or exempt low-income riders from the surcharge. Our findings suggest that misses the point. A low-income neighbourhood with a nearby subway station behaves almost identically to a wealthy one — riders find an alternative. The problem is not the toll level. It is the absence of transit in underserved areas. Subway investment is the more powerful lever.

---

## The Data

| Source | What it covers | Size |
|---|---|---|
| NYC TLC trip records | Every Uber, Lyft, and yellow cab trip, Jan 2022–Jun 2025 | 509M trips |
| MTA subway ridership | Station entries across the whole system | 93M records |
| US Census ACS 2023 | Median household income by neighbourhood | 1,610 zip codes |
| NOAA Central Park | Daily weather (temperature, rain, snow) | 1,278 days |
| NYC DOT speeds | Traffic speeds before and after the toll | Jan 2024 + Jan 2025 |

Raw data is not included in this repository because of its size (approximately 27 GB). See `data/README.md` for download instructions and sources.

---

## The Paper

A full write-up of the analysis is in the `paper/` folder and available as a preprint on arXiv.

> **Equity, Elasticity, and Mode Shift: Zone-Level Effects of NYC's Congestion Relief Zone on High-Volume For-Hire Vehicle Demand**
> Priyanka Mysore Krishna, 2026

---

## How to Run It

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw data (see data/README.md for instructions)

# 3. Run the pipeline in order
python -m src.pipeline.ingest
python -m src.pipeline.spatial_join
python -m src.pipeline.build_panel

# 4. Run the five analyses
python -m src.analysis.phase1_counterfactual
python -m src.analysis.phase2_equity
python -m src.analysis.phase3_modeshift
python -m src.analysis.phase4_did
python -m src.analysis.phase5_forecast

# 5. Generate figures and tables
python -m src.analysis.tables
python -m src.analysis.figures.fig1_study_area
python -m src.analysis.figures.fig2_timeline
python -m src.analysis.figures.fig5_mode_by_income
python -m src.analysis.figures.fig6_scenario_lines
python -m src.analysis.figures.fig7_counterfactual
python -m src.analysis.choropleth
python -m src.analysis.parallel_trends
```

All outputs go to `outputs/`.

---

## Project Structure

```
nyc-congestion/
├── config.py               # Paths, dates, zone IDs
├── src/
│   ├── pipeline/           # Data ingestion and panel construction
│   └── analysis/           # Five analysis phases + figures + tables
├── outputs/                # Figures (PNG) and tables (LaTeX)
├── paper/                  # Full LaTeX paper
└── data/
    └── README.md           # Download instructions for raw data
```

---

## Citation

If you use this work, please cite:

```text
Mysore Krishna, P. (2026) “Equity, Elasticity, and Mode Shift: Zone-Level Effects of NYC's Congestion Relief Zone on High-Volume For-Hire Vehicle Demand”. Self-published. doi:10.5281/zenodo.20802038.
```

## Paper

* DOI: https://doi.org/10.5281/zenodo.20802038
* Zenodo Record: https://zenodo.org/records/20802038

## Author

Priyanka Mysore Krishna

## License

Code in this repository is licensed under the MIT License.

The paper is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
