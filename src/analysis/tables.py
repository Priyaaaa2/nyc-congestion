import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config

PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    return (str(s)
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#"))


def booktabs_wrap(title: str, label: str,
                  header: str, body: str,
                  footnote: str = "") -> str:
    """Wrap content in a booktabs-style LaTeX table."""
    fn = (f"\\multicolumn{{999}}{{l}}{{"
          f"\\footnotesize {tex_escape(footnote)}}}\\\\\n"
          if footnote else "")
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{title}}}
\\label{{{label}}}
\\small
\\begin{{tabular}}{{{header}}}
\\toprule
{body}
\\bottomrule
{fn}\\end{{tabular}}
\\end{{table}}
"""

def make_table1():
    print("Building Table 1 — Data sources...")

    rows = [
        ("NYC TLC HVFHV",
         "NYC Taxi \\& Limousine Commission",
         "Jan 2022 -- Jun 2025",
         "Trip",
         "378M",
         "Platform ID, pickup zone, fare, timestamp"),
        ("NYC TLC Yellow Taxi",
         "NYC Taxi \\& Limousine Commission",
         "Jan 2022 -- Jun 2025",
         "Trip",
         "131M",
         "Pickup zone, fare, timestamp"),
        ("MTA Subway Ridership",
         "MTA Open Data",
         "Jan 2022 -- Jun 2025",
         "Station-hour",
         "93M",
         "Station complex ID, ridership, transit mode"),
        ("Census ACS 5-Year",
         "US Census Bureau",
         "2023 (2019--2023)",
         "ZCTA",
         "1,610",
         "Median household income, commute mode shares"),
        ("NOAA GHCND",
         "NOAA Climate Data",
         "Jan 2022 -- Jun 2025",
         "Day",
         "1,278",
         "Max temp, precipitation, snow depth (Central Park)"),
        ("NYC DOT Speeds",
         "NYC Open Data",
         "Jan 2024, Jan 2025",
         "Segment-5min",
         "$\\sim$2M",
         "Link speed, travel time, corridor ID"),
    ]

    col_spec = "llllrl"
    header_row = (
        "\\textbf{Source} & \\textbf{Provider} & "
        "\\textbf{Period} & \\textbf{Unit} & "
        "\\textbf{N} & \\textbf{Key variables} \\\\\n\\midrule"
    )

    body = header_row + "\n"
    for src, prov, period, unit, n, vars_ in rows:
        body += (
            f"{src} & {prov} & {period} & "
            f"{unit} & {n} & {vars_} \\\\\n"
        )

    tex = booktabs_wrap(
        title="Data Sources",
        label="tab:data",
        header=col_spec,
        body=body,
        footnote=(
            "HVFHV = High-Volume For-Hire Vehicle (Uber, Lyft, Via). "
            "TLC zone shapefile (263 zones) used for spatial joins. "
            "ACS = American Community Survey."
        )
    )

    out = OUT_DIR / "table1_data_sources.tex"
    out.write_text(tex)
    print(f"  Saved -> {out}")
    return tex

def make_table2():
    print("Building Table 2 — Descriptive statistics...")

    panel = pd.read_parquet(PROC / "master_panel.parquet")
    panel["week_start"] = pd.to_datetime(panel["week_start"])

    fhvhv = panel[
        (panel["vehicle_type"] == "fhvhv") &
        (panel["in_crz"] == 1)
    ].copy()

    pre  = fhvhv[fhvhv["post_treatment"] == 0]
    post = fhvhv[fhvhv["post_treatment"] == 1]

    variables = [
        ("trip_count",      "Weekly HVFHV trips (CRZ pickup)",  "{:,.0f}"),
        ("avg_fare",        "Average fare (\\$)",                "{:.2f}"),
        ("avg_cbd_fee",     "Average CBD congestion fee (\\$)",  "{:.2f}"),
        ("median_income",   "Zone median income (\\$000s)",      "{:.1f}",
         lambda x: x / 1000),
        ("has_subway",      "Has subway within 400m (0/1)",      "{:.3f}"),
        ("avg_tmax",        "Max temperature (°C)",              "{:.1f}"),
        ("total_prcp",      "Weekly precipitation (mm)",         "{:.1f}"),
        ("snow_week",       "Snow week indicator (0/1)",         "{:.3f}"),
    ]

    def stats(df, col, transform=None):
        s = df[col].dropna()
        if transform:
            s = s.apply(transform)
        return s.mean(), s.std()

    col_spec = "lrrrr"
    header_row = (
        "\\textbf{Variable} & "
        "\\multicolumn{2}{c}{\\textbf{Pre-treatment}} & "
        "\\multicolumn{2}{c}{\\textbf{Post-treatment}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        " & Mean & SD & Mean & SD \\\\\n\\midrule"
    )

    body = header_row + "\n"
    for row in variables:
        col     = row[0]
        label   = row[1]
        fmt     = row[2]
        tfm     = row[4] if len(row) > 4 else None

        if len(row) == 4 and callable(row[3]):
            tfm = row[3]

        try:
            pre_m,  pre_s  = stats(pre,  col, tfm)
            post_m, post_s = stats(post, col, tfm)
            body += (
                f"{label} & "
                f"{fmt.format(pre_m)} & {fmt.format(pre_s)} & "
                f"{fmt.format(post_m)} & {fmt.format(post_s)} \\\\\n"
            )
        except Exception as e:
            print(f"  Warning: {col} — {e}")

    body += "\\midrule\n"
    body += (
        "Zone-week observations & "
        f"\\multicolumn{{2}}{{c}}{{{len(pre):,}}} & "
        f"\\multicolumn{{2}}{{c}}{{{len(post):,}}} \\\\\n"
    )
    body += (
        "Unique zones & "
        f"\\multicolumn{{2}}{{c}}{{{pre['zone_id'].nunique()}}} & "
        f"\\multicolumn{{2}}{{c}}{{{post['zone_id'].nunique()}}} \\\\\n"
    )
    body += (
        "Weeks & "
        f"\\multicolumn{{2}}{{c}}{{{pre['week_start'].nunique()}}} & "
        f"\\multicolumn{{2}}{{c}}{{{post['week_start'].nunique()}}} \\\\\n"
    )

    tex = booktabs_wrap(
        title="Descriptive Statistics — CRZ HVFHV Zone-Week Panel",
        label="tab:descriptive",
        header=col_spec,
        body=body,
        footnote=(
            "Pre-treatment: Jan 2022 -- Dec 2024 (156 weeks). "
            "Post-treatment: Jan -- Jun 2025 (25 weeks). "
            "CRZ = Congestion Relief Zone (65 zones, excl. Randalls Island "
            "and Governor's Island). "
            "CBD fee = 0 in pre-period (column added Jan 2025)."
        )
    )

    out = OUT_DIR / "table2_descriptive.tex"
    out.write_text(tex)
    print(f"  Saved -> {out}")
    return tex


def make_table3():
    print("Building Table 3 — Regression results...")

    panel  = pd.read_parquet(PROC / "master_panel.parquet")
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    income = pd.read_csv(PROC / "zone_income.csv")
    subway = pd.read_csv(PROC / "zone_subway.csv")
    equity = pd.read_parquet(PROC / "phase2/zone_equity.parquet")

    equity = equity[~equity["zone_id"].isin([194, 105])]
    equity["log_income"] = np.log(equity["median_income"].clip(lower=1))

    m2 = smf.ols(
        "effect_pct ~ log_income + has_subway + C(Borough)",
        data=equity.dropna(subset=["median_income", "effect_pct"])
    ).fit(cov_type="HC3")

    did_path = PROC / "phase4/platform_weekly.parquet"
    did_df   = pd.read_parquet(did_path)
    did_df["week_start"] = pd.to_datetime(did_df["week_start"])
    did_df["post"]       = (did_df["week_start"] >= config.TREATMENT_DATE).astype(int)
    did_df["is_lyft"]    = (did_df["platform"] == "lyft").astype(int)
    did_df["did"]        = did_df["post"] * did_df["is_lyft"]
    did_df["log_trips"]  = np.log1p(did_df["trip_count"])
    did_df["month"]      = did_df["week_start"].dt.month

    m4 = smf.ols(
        "log_trips ~ did + is_lyft + post + C(month)",
        data=did_df
    ).fit(cov_type="HC3")

    def stars(p):
        if p < 0.001: return "^{***}"
        if p < 0.01:  return "^{**}"
        if p < 0.05:  return "^{*}"
        return ""

    def fmt_coef(coef, se, pval):
        s = stars(pval)
        return (
            f"${coef:.3f}{s}$ \\\\\n"
            f" & $({se:.3f})$ & \\\\\n"
        )

    col_spec = "lcc"
    header_row = (
        "\\textbf{Variable} & "
        "\\textbf{(1) Equity OLS} & "
        "\\textbf{(2) Platform DiD} \\\\\n"
        " & \\textit{Dep: effect\\_pct} & "
        "\\textit{Dep: log(trips)} \\\\\n"
        "\\midrule"
    )

    all_vars = [

        ("log(Median income)",               "log_income", None),
        ("Has subway (0/1)",                 "has_subway",  None),
        ("DiD ($\\tau$: Lyft $\\times$ Post)", None,       "did"),
        ("Lyft indicator",                   None,         "is_lyft"),
        ("Post-treatment",                   None,         "post"),
        ("Intercept",                        "Intercept",  "Intercept"),
    ]

    body = header_row + "\n"

    for label, k2, k4 in all_vars:
        c1 = ""   
        c2 = "" 

        if k2 and k2 in m2.params:
            s  = stars(m2.pvalues[k2])
            c1 = f"${m2.params[k2]:.3f}{s}$"
        if k4 and k4 in m4.params:
            s  = stars(m4.pvalues[k4])
            c2 = f"${m4.params[k4]:.3f}{s}$"

        body += f"{label} & {c1} & {c2} \\\\\n"

        se1 = f"$({m2.bse[k2]:.3f})$" if k2 and k2 in m2.bse else ""
        se2 = f"$({m4.bse[k4]:.3f})$" if k4 and k4 in m4.bse else ""
        body += f" & {se1} & {se2} \\\\\n"

    body += "\\midrule\n"
    body += (
        f"Borough FE & Yes & No \\\\\n"
        f"Month FE & No & Yes \\\\\n"
        f"HC3 standard errors & Yes & Yes \\\\\n"
        f"$N$ & {int(m2.nobs)} zones & "
        f"{int(m4.nobs)} week-platform \\\\\n"
        f"$R^2$ & {m2.rsquared:.3f} & "
        f"{m4.rsquared:.3f} \\\\\n"
    )

    tex = booktabs_wrap(
        title="Regression Results: Equity OLS (1) and Platform DiD (2)",
        label="tab:regressions",
        header=col_spec,
        body=body,
        footnote=(
            "HC3 heteroskedasticity-robust standard errors in parentheses. "
            "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
            "(1) Dep. var: HVFHV trip change vs counterfactual (\\%). "
            "63 CRZ zones, Jan--Jun 2025 vs Jan--Jun 2024. "
            "(2) Dep. var: log weekly CRZ trips. "
            "Treated = Lyft (\\$1.50 credit, Jan 2025); "
            "Control = Uber. Pre-period: Jan 2023 -- Dec 2024."
        )
    )

    out = OUT_DIR / "table3_regressions.tex"
    out.write_text(tex)
    print(f"  Saved -> {out}")
    return tex

def main():
    print("\n" + "="*55)
    print("  PHASE B: LATEX TABLES")
    print("="*55)

    t1 = make_table1()
    t2 = make_table2()
    t3 = make_table3()

    print(f"\n{'='*55}")
    print(f"  DONE — 3 tables written to {OUT_DIR}")
    print(f"  table1_data_sources.tex")
    print(f"  table2_descriptive.tex")
    print(f"  table3_regressions.tex")
    print(f"{'='*55}\n")

    for name, tex in [("Table 1", t1), ("Table 2", t2), ("Table 3", t3)]:
        print(f"\n{'─'*50}")
        print(f"  {name} preview (first 8 lines):")
        print(f"{'─'*50}")
        for line in tex.split("\n")[:8]:
            print(f"  {line}")


if __name__ == "__main__":
    main()