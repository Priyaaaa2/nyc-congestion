import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
import config

OUT_DIR = pathlib.Path(config.OUTPUTS)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE     = "#0072B2"
ORANGE   = "#E69F00"
GREEN    = "#009E73"
PURPLE   = "#CC79A7"
YELLOW   = "#F0E442"
RED      = "#D55E00"
BLACK    = "#000000"


def main():
    print("Building Figure 2 — data timeline...")

    START = pd.Timestamp("2021-10-01")
    END   = pd.Timestamp("2025-10-01")
    TOTAL = (END - START).days

    def xd(dt):
        return (pd.Timestamp(dt) - START).days

    sources = [
        ("TLC HVFHV (Uber/Lyft)",  "2022-01-01", "2025-06-30", BLUE,   6),
        ("TLC Yellow Taxi",         "2022-01-01", "2025-06-30", ORANGE, 5),
        ("MTA Subway Ridership",    "2022-01-01", "2025-06-30", GREEN,  4),
        ("Census ACS Income",       "2023-01-01", "2023-12-31", PURPLE, 3),
        ("NOAA Weather",            "2022-01-01", "2025-06-30", YELLOW, 2),
        ("DOT Speeds (Jan 2024)",   "2024-01-01", "2024-02-01", RED,    1),
        ("DOT Speeds (Jan 2025)",   "2025-01-01", "2025-02-01", RED,    1),
    ]

    n_labels = {
        "TLC HVFHV (Uber/Lyft)": "378M trips",
        "TLC Yellow Taxi":         "131M trips",
        "MTA Subway Ridership":    "93M rows",
        "Census ACS Income":       "1,610 ZCTAs",
        "NOAA Weather":            "1,278 days",
        "DOT Speeds (Jan 2024)":   "",
        "DOT Speeds (Jan 2025)":   "Jan only \u00d72",
    }

    left_labels = {
        "TLC HVFHV (Uber/Lyft)": "TLC HVFHV\n(Uber/Lyft)",
        "TLC Yellow Taxi":         "TLC Yellow Taxi",
        "MTA Subway Ridership":    "MTA Subway\nRidership",
        "Census ACS Income":       "Census ACS\nIncome",
        "NOAA Weather":            "NOAA Weather",
        "DOT Speeds (Jan 2024)":   "",
        "DOT Speeds (Jan 2025)":   "DOT Speeds",
    }

    n_rows   = 6
    BAR_H    = 0.55
    Y_MARGIN = 0.5
    PAD      = 0.07

    fig, ax = plt.subplots(figsize=(14, n_rows * 1.1 + 2.5))
    ax.set_xlim(xd("2021-10-01"), xd("2025-10-01"))
    ax.set_ylim(0, n_rows + Y_MARGIN + 1)
    ax.set_axis_off()

    drawn_right = set()
    for name, s, e, color, row in sources:
        x0 = xd(s)
        x1 = xd(e)
        rect = mpatches.FancyBboxPatch(
            (x0, row - BAR_H / 2), x1 - x0, BAR_H,
            boxstyle=f"round,pad={PAD}",
            facecolor=color, edgecolor="white",
            alpha=1.0, linewidth=0.8,
            zorder=3
        )
        ax.add_patch(rect)

        lbl = left_labels.get(name, name)
        if lbl:
            ax.text(xd("2021-10-01") - 15, row, lbl,
                    ha="right", va="center", fontsize=9,
                    color="#3d3d3a")

        nr = n_labels.get(name, "")
        if nr and name not in drawn_right:
            ax.text(xd("2025-10-01") + 15, row, nr,
                    ha="left", va="center", fontsize=8.5,
                    color="#6E6E6E")
            drawn_right.add(name)

    y_top = n_rows + 0.3
    y_bot = 0.6

    events = [
        ("2022-01-01", "Data window\nstarts",       PURPLE,  ":",  False, 0),
        ("2025-01-05", "CBDTP launch\nJan 5, 2025", BLACK,   "--", True,  0),
        ("2025-01-31", "Lyft credit\nends",          ORANGE,  ":",  False, 1),
        ("2025-06-30", "Analysis\nwindow ends",      BLUE,    ":",  False, 0),
    ]

    for date_str, label, color, ls, bold, stagger in events:
        xv = xd(date_str)
        ax.plot([xv, xv], [y_bot, y_top],
                color=color,
                linewidth=1.8 if bold else 1.0,
                linestyle=ls, zorder=5)
        y_text = y_top + 0.05 + stagger * 0.55
        ax.text(xv + 8, y_text, label,
                ha="left", va="bottom",
                fontsize=8 if bold else 7.5,
                color=color,
                fontweight="bold" if bold else "normal")

    ax.axvspan(
        xd("2025-01-05"), xd("2025-01-31"),
        ymin=y_bot / (n_rows + Y_MARGIN + 1),
        ymax=y_top / (n_rows + Y_MARGIN + 1),
        alpha=0.10, color=ORANGE, zorder=1
    )

    tick_dates = pd.date_range("2022-01-01", "2025-09-01", freq="6MS")
    for td in tick_dates:
        xv = xd(td)
        ax.plot([xv, xv], [y_bot - 0.05, y_bot - 0.2],
                color="#999999", linewidth=0.8, zorder=2)
        ax.text(xv, y_bot - 0.35,
                td.strftime("%b\n%Y"),
                ha="center", va="top",
                fontsize=8, color="#333333")

    ax.plot(
        [xd("2022-01-01"), xd("2025-07-01")],
        [y_bot - 0.05, y_bot - 0.05],
        color="#999999", linewidth=0.8, zorder=2
    )

    ax.set_title(
        "Data Sources and Event Timeline\n"
        "NYC Congestion Relief Zone Analysis  |  January 2022 \u2013 June 2025",
        fontsize=12, fontweight="bold", pad=14
    )

    out = OUT_DIR / "fig2_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()