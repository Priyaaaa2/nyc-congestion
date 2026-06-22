import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
import config

PROC    = pathlib.Path(config.DATA_PROC)
OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building Figure 1 — study area map...")

    zones  = gpd.read_file(config.TLC_ZONES_SHP).to_crs("EPSG:4326")
    income = pd.read_csv(PROC / "zone_income.csv")
    zones  = zones.merge(income[["zone_id","Borough"]],
                         left_on="LocationID", right_on="zone_id",
                         how="left")

    crz_ids = set(config.CRZ_ZONES)
    zones["is_crz"] = zones["LocationID"].isin(crz_ids)

    fig, axes = plt.subplots(1, 2, figsize=(13, 9),
                              gridspec_kw={"width_ratios": [1, 1.4]})

    ax1 = axes[0]
    zones.plot(ax=ax1, color="#E8E6DE", edgecolor="#B4B2A9",
               linewidth=0.3)
    zones[zones["is_crz"]].plot(ax=ax1, color="#378ADD",
                                edgecolor="white", linewidth=0.4,
                                alpha=0.85)
    borough_centroids = {
        "Manhattan": (-73.971, 40.783),
        "Brooklyn":  (-73.944, 40.650),
        "Queens":    (-73.820, 40.710),
        "Bronx":     (-73.865, 40.845),
        "Staten Island": (-74.150, 40.580),
    }
    for b, (lon, lat) in borough_centroids.items():
        ax1.text(lon, lat, b, fontsize=7.5, ha="center",
                 color="#3d3d3a", fontweight="500")

    ax1.set_xlim(-74.26, -73.70)
    ax1.set_ylim(40.49, 40.93)
    ax1.set_axis_off()
    ax1.set_title("NYC context — CRZ in blue",
                  fontsize=10, pad=6)

    from matplotlib.patches import FancyArrowPatch
    ax1.add_patch(plt.Rectangle(
        (-74.03, 40.68), 0.13, 0.20,
        fill=False, edgecolor="#E24B4A",
        linewidth=1.5, linestyle="--"
    ))

    ax2 = axes[1]
    manhattan = zones[zones["borough"] == "Manhattan"]
    crz       = zones[zones["is_crz"]]

    manhattan.plot(ax=ax2, color="#F0EEE6",
                   edgecolor="#C8C6BE", linewidth=0.4)

    crz.plot(ax=ax2, color="#378ADD",
             edgecolor="white", linewidth=0.5, alpha=0.75)

    crz_union = crz.union_all()
    import geopandas as gpd2
    gpd.GeoSeries([crz_union]).plot(
        ax=ax2, facecolor="none",
        edgecolor="#1A252F", linewidth=2.0
    )

    ax2.axhline(40.768, color="#3d3d3a",
                linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.text(-73.903, 40.769, "60th St (northern boundary)",
             fontsize=7.5, color="#3d3d3a", va="bottom")

    ax2.annotate("N", xy=(-73.902, 40.876),
                 fontsize=11, ha="center", fontweight="bold",
                 color="#3d3d3a")
    ax2.annotate("", xy=(-73.902, 40.882),
                 xytext=(-73.902, 40.870),
                 arrowprops=dict(arrowstyle="-|>",
                                 color="#3d3d3a", lw=1.5))

    ax2.set_xlim(-74.03, -73.90)
    ax2.set_ylim(40.68, 40.89)
    ax2.set_axis_off()
    ax2.set_title(
        f"Manhattan CRZ — {len(crz)} taxi zones\n"
        "South of 60th St",
        fontsize=10, pad=6
    )

    ax2.text(-73.998, 40.685,
             f"{len(crz)} CRZ zones\n"
             f"63 zones with equity data",
             fontsize=8, color="#3d3d3a",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="white",
                       edgecolor="#C8C6BE",
                       alpha=0.9))

    handles = [
        mpatches.Patch(color="#378ADD", alpha=0.75,
                       label="CRZ zones (65 total)"),
        mpatches.Patch(color="#F0EEE6",
                       edgecolor="#C8C6BE",
                       label="Non-CRZ Manhattan zones"),
    ]
    ax2.legend(handles=handles, loc="lower right",
               fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "Study Area: NYC Congestion Relief Zone\n"
        "Central Business District Tolling Program (CBDTP), January 2025",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.text(0.5, -0.01,
             "Source: NYC TLC taxi zone shapefile. "
             "CRZ = Manhattan south of 60th Street. "
             "Red dashed box indicates the detail area shown at right.",
             ha="center", fontsize=7.5, color="#666666")

    plt.tight_layout()
    out = OUT_DIR / "fig1_study_area.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
