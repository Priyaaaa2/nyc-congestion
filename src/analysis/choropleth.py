import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
 
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import config
 
PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def load_data():
    print("Loading data...")
 
    equity = pd.read_parquet(PROC / "phase2/zone_equity.parquet")
    mode   = pd.read_parquet(PROC / "phase3/mode_split.parquet")
    zones  = gpd.read_file(config.TLC_ZONES_SHP).to_crs("EPSG:4326")
 
    df = equity.merge(
        mode[["zone_id", "suppressed_pct", "mta_pct",
              "yellow_pct", "loss_abs"]],
        on="zone_id", how="left"
    )
 
    map_df   = zones.merge(df, left_on="LocationID",
                           right_on="zone_id", how="left")
    manhattan = map_df[map_df["borough"] == "Manhattan"].copy()
 
    print(f"  Manhattan zones          : {len(manhattan)}")
    print(f"  With equity data         : "
          f"{manhattan['effect_pct'].notna().sum()}")
    print(f"  Without subway (CRZ)     : "
          f"{((manhattan['has_subway']==0) & manhattan['effect_pct'].notna()).sum()}")
 
    return manhattan, map_df
 
 
def make_choropleth(manhattan, map_df):
    print("Building choropleth...")
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 14))

    ax1 = axes[0]
 
    map_df.plot(ax=ax1, color="#F0F0F0", edgecolor="#CCCCCC",
                linewidth=0.3)
 
    vmin, vmax = -25, 10
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
 
    manhattan.plot(
        column="effect_pct", ax=ax1, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=0.5,
        missing_kwds={"color": "#D8D8D8", "edgecolor": "white"}
    )
 
    crz = manhattan[manhattan["effect_pct"].notna()]
    crz.plot(ax=ax1, facecolor="none", edgecolor="#333333",
             linewidth=1.2)
 
    no_sub = manhattan[
        manhattan["effect_pct"].notna() &
        (manhattan["has_subway"] == 0)
    ]
    if len(no_sub) > 0:
        no_sub.plot(ax=ax1, facecolor="none",
                    edgecolor="#222222", linewidth=0.8, hatch="///")
 
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.55, pad=0.02)
    cbar.set_label("Trip change vs counterfactual (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
 
    ax1.set_xlim(-74.03, -73.90)
    ax1.set_ylim(40.68, 40.88)
    ax1.set_axis_off()
    ax1.set_title(
        "FHVHV Trip Change vs Counterfactual\n"
        "Jan–Jun 2025 | Manhattan CRZ zones",
        fontsize=12, fontweight="bold", pad=12
    )
 
    handles = [
        mpatches.Patch(facecolor="none", edgecolor="#333333",
                       linewidth=1.5, label="CRZ zone boundary"),
        mpatches.Patch(facecolor="none", edgecolor="#222222",
                       linewidth=0.8, hatch="///",
                       label="No subway within 400m"),
        mpatches.Patch(color="#D8D8D8",
                       label="Outside CRZ / no data"),
    ]
    ax1.legend(handles=handles, loc="lower left",
               fontsize=8.5, framealpha=0.9)
 
    ax2 = axes[1]
 
    crz_data = manhattan[
        manhattan["median_income"].notna() &
        manhattan["effect_pct"].notna()
    ].copy()
 
    colors = crz_data["has_subway"].map(
        {1.0: "#2196F3", 0.0: "#FF5722"}
    ).fillna("#999999")
 
    if ("loss_abs" in crz_data.columns and
            crz_data["loss_abs"].notna().any()):
        max_loss = crz_data["loss_abs"].max()
        sizes = crz_data["loss_abs"].fillna(0) / max_loss * 200 + 40
    else:
        sizes = 70
 
    x = crz_data["median_income"] / 1000
    y = crz_data["effect_pct"]
 
    ax2.scatter(x, y, c=colors, s=sizes, alpha=0.75,
                edgecolors="white", linewidth=0.6, zorder=3)
 
    mask = x.notna() & y.notna()
    if mask.sum() > 5:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x_line, p(x_line), "k--", linewidth=1.5,
                 alpha=0.7, label=f"OLS β = {z[0]:.2f}", zorder=2)
 
    ax2.axhline(0, color="#999999", linewidth=0.8,
                linestyle=":", zorder=1)
 
    ax2.set_xlim(35, 240)
    ax2.set_ylim(y.min() - 2, y.max() + 4)
 
    quintiles = x.quantile([0.2, 0.4, 0.6, 0.8])
    for q in quintiles:
        ax2.axvline(q, color="#CCCCCC", linewidth=0.6,
                    linestyle="--", zorder=1)
 
    y_top = y.max() + 2.0
    labels    = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    positions = [55,   88,   118,  148,  200]
    for label, pos in zip(labels, positions):
        ax2.text(pos, y_top, label, ha="center", va="bottom",
                 fontsize=8, color="#888888", style="italic",
                 transform=ax2.transData, clip_on=False)
    ax2.set_xlabel("Median household income ($000s)", fontsize=11)
    ax2.set_ylabel("Trip change vs counterfactual (%)", fontsize=11)
    ax2.set_title(
        "Income vs Trip Loss — CRZ Zones\n"
        "β = −3.06  (p = 0.026)  |  n = 63 zones",
        fontsize=12, fontweight="bold", pad=12
    )
 
    h = [
        mpatches.Patch(color="#2196F3",
                       label="Subway within 400m"),
        mpatches.Patch(color="#FF5722",
                       label="No subway within 400m"),
        plt.Line2D([0], [0], linestyle="--", color="black",
                   linewidth=1.5, label="OLS trend"),
    ]
    ax2.legend(handles=h, fontsize=9, loc="upper right",
               framealpha=0.9)
    ax2.grid(True, alpha=0.25, zorder=0)
    ax2.tick_params(labelsize=9)
 
    fig.suptitle(
        "NYC Congestion Pricing: Equity Impact on Ride-Hailing\n"
        "Congestion Relief Zone  |  January–June 2025",
        fontsize=14, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.01,
        "Source: NYC TLC trip records · US Census ACS 2023 · "
        "MTA subway ridership. "
        "Counterfactual = Jan–Jun 2024 × (1 + 3.4% baseline growth). "
        "Hatching = no subway station within 400m of zone centroid.",
        ha="center", fontsize=7.5, color="#666666"
    )
 
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
 
    out_path = OUT_DIR / "equity_choropleth.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path
 
 
def main():
    print("\n" + "="*55)
    print("  DELIVERABLE 1: EQUITY CHOROPLETH")
    print("="*55)
 
    manhattan, map_df = load_data()
    out_path = make_choropleth(manhattan, map_df)
 
    print(f"\n{'='*55}")
    print(f"  DONE: {out_path}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()