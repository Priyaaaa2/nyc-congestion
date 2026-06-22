import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def build_table():
    base = pd.read_parquet(PROC / "phase5/forecast_base.parquet")
    sens = pd.read_parquet(PROC / "phase5/forecast_2x.parquet")
 
    annual_base = (base
        .groupby(["scenario", "year"])
        .agg(trips_M   =("monthly_trips", "sum"),
             revenue_M =("mta_revenue",   "sum"),
             avg_toll  =("toll",          "mean"))
        .reset_index()
    )
    annual_base["trips_M"]   /= 1e6
    annual_base["revenue_M"] /= 1e6
 
    annual_sens = (sens
        .groupby(["scenario", "year"])
        .agg(revenue_M_2x=("mta_revenue", "sum"))
        .reset_index()
    )
    annual_sens["revenue_M_2x"] /= 1e6
    annual_sens["scenario"] = (
        annual_sens["scenario"].str.replace("_2x", "")
    )
 
    df = annual_base.merge(
        annual_sens, on=["scenario", "year"], how="left"
    )
 
    label_map = {
        "A_flat_9":  r"A — \$9 flat (HVFHV \$1.50)",
        "B_rise_15": r"B — Rise to \$15 in 2027 (HVFHV \$2.50)",
        "C_pause":    "C — 6-month pause Jul-Dec 2025",
    }
    df["label"] = df["scenario"].map(label_map)
    return df
 
 
def make_table_image(df):
    scenarios = ["A_flat_9", "B_rise_15", "C_pause"]
    years     = [2025, 2026, 2027]
 
    col_labels = [
        "Scenario",
        "2025\nTrips (M)", "2025\nRevenue",
        "2026\nTrips (M)", "2026\nRevenue",
        "2027\nTrips (M)", "2027\nRevenue",
        "2027 Revenue\n(2x elasticity)",
    ]
 
    rows = []
    for sc in scenarios:
        label = df[df["scenario"] == sc]["label"].iloc[0]
        row   = [label]
        for yr in years:
            sub = df[(df["scenario"] == sc) & (df["year"] == yr)]
            if len(sub):
                row.append(f"{sub['trips_M'].iloc[0]:.1f}M")
                row.append(f"${sub['revenue_M'].iloc[0]:.0f}M")
            else:
                row.extend(["--", "--"])
        sub27 = df[(df["scenario"] == sc) & (df["year"] == 2027)]
        if len(sub27):
            val = sub27["revenue_M_2x"].iloc[0]
            row.append(f"${val:.0f}M")
        else:
            row.append("--")
        rows.append(row)
 
    n_cols = len(col_labels)
    n_rows = len(rows) + 1
 
    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.set_axis_off()
 
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    col_widths = [0.30, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12]
    for j, w in enumerate(col_widths):
        for i in range(n_rows):
            table[i, j].set_width(w)
 

    header_color = "#1A252F"
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold",
                            fontsize=9)
        cell.set_height(0.28)
 

    row_colors = ["#F5F6F7", "#FFFFFF", "#F5F6F7"]
    warn_color = "#FFF3CD"
    warn_red   = "#FDECEA"
 
    for i, sc in enumerate(scenarios):
        for j in range(n_cols):
            cell = table[i + 1, j]
            if sc == "B_rise_15" and j == 6:
                cell.set_facecolor(warn_color)
            elif sc == "B_rise_15" and j == 7:
                val = df[
                    (df["scenario"] == sc) & (df["year"] == 2027)
                ]["revenue_M_2x"].iloc[0]
                cell.set_facecolor(warn_red if val < 0 else warn_color)
            else:
                cell.set_facecolor(row_colors[i])
            if j == 0:
                cell.set_text_props(fontweight="bold", fontsize=9)
            else:
                cell.set_text_props(fontsize=10)
 
    ax.set_title(
        "Phase 5 — Three-Scenario Revenue Forecast\n"
        "HVFHV trips (M) and MTA toll revenue "
        "(HVFHV surcharge only, $1.50/trip base)",
        fontsize=11, fontweight="bold", pad=10, loc="center"
    )
 
    fig.text(
        0.04, 0.0,
        "Elasticity: e = -1.27 (Phase 1 implied, base) and e = -2.53 (2x sensitivity).  "
        "Baseline growth: +3.4%/yr.  "
        "HVFHV surcharge: $1.50 (Scenarios A/C), $2.50 (Scenario B from Jan 2027).  "
        "Yellow = below Scenario A.  Red = negative revenue.",
        fontsize=7.5, color="#555555"
    )
 
    plt.subplots_adjust(top=0.82, bottom=0.12, left=0.02, right=0.98)
 
    out_path = OUT_DIR / "scenario_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Table image -> {out_path}")
    return out_path
 
 
def save_csv(df):
    out = OUT_DIR / "scenario_table.csv"
    export = df[["label", "year", "trips_M", "revenue_M",
                 "revenue_M_2x", "avg_toll"]].copy()
    export.columns = ["Scenario", "Year", "Trips_M",
                      "Revenue_M", "Revenue_M_2x_elast", "Avg_toll"]
    export = export.sort_values(["Scenario", "Year"])
    export.to_csv(out, index=False, float_format="%.1f")
    print(f"  CSV         -> {out}")
 
 
def print_summary(df):
    print("\n  Scenario table:")
    print(f"  {'Scenario':<42} {'Year':>5} "
          f"{'Trips(M)':>9} {'Rev($M)':>9} {'2xelas($M)':>11}")
    print(f"  {'-'*79}")
    for _, row in df.sort_values(["scenario", "year"]).iterrows():
        print(f"  {row['label']:<42} {int(row['year']):>5} "
              f"{row['trips_M']:>9.1f} "
              f"{row['revenue_M']:>9.1f} "
              f"{row['revenue_M_2x']:>11.1f}")
 
    a27  = df[(df["scenario"]=="A_flat_9")  & (df["year"]==2027)]["revenue_M"].iloc[0]
    b27  = df[(df["scenario"]=="B_rise_15") & (df["year"]==2027)]["revenue_M"].iloc[0]
    b27s = df[(df["scenario"]=="B_rise_15") & (df["year"]==2027)]["revenue_M_2x"].iloc[0]
    c25  = df[(df["scenario"]=="C_pause")   & (df["year"]==2025)]["revenue_M"].iloc[0]
    a25  = df[(df["scenario"]=="A_flat_9")  & (df["year"]==2025)]["revenue_M"].iloc[0]
 
    print(f"\n  Key takeaways:")
    print(f"  -> Keep $9   (2027)  : ${a27:.0f}M revenue")
    print(f"  -> Rise $15  (2027)  : ${b27:.0f}M (base e) / ${b27s:.0f}M (2x e)")
    print(f"  -> Pause cost (2025) : ${c25-a25:.0f}M vs keeping toll")
    print(f"  -> $15 rise destroys revenue under aggressive elasticity")
 
 
def main():
    print("\n" + "="*55)
    print("  DELIVERABLE 2: SCENARIO TABLE")
    print("="*55)
 
    df = build_table()
    print_summary(df)
    make_table_image(df)
    save_csv(df)
 
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()