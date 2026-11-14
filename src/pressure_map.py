"""Relative acoustic activity per site and month, for the research write-up.

The challenge asks for a pressure map but warns that absolute calibration of the
South African exports has not been verified. So this produces a *relative
activity* index: the median level in the low-frequency bands where vessel noise
dominates (31.6-400 Hz), expressed as a percentile rank across all sites and
months. It is a proxy, not a pressure in dB re 1 uPa.

Outputs
-------
data/site_monthly.csv          one row per site and month
entrega/figures/activity_map.png
entrega/figures/activity_heatmap.png

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.pressure_map
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.layout import BAND_CENTRES, ordered_band_columns  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
FIGURES = PROJECT_DIR / "entrega" / "figures"
ACOUSTICS = DATA_DIR / "acoustics"
STATIONS = DATA_DIR / "stations.csv"

# Vessel noise dominates the low end; 31.6-400 Hz is the first 13 bands.
LOW_BANDS = slice(0, 13)
MIN_MINUTES_PER_MONTH = 360  # ~6 hours of data before a monthly value is trusted

# Rough South African coastline for a schematic frame (lon, lat).
COASTLINE = [
    (17.6, -32.6), (17.9, -33.1), (18.4, -33.9), (18.5, -34.35),
    (19.4, -34.6), (20.0, -34.8), (21.0, -34.4), (22.1, -34.2),
    (23.5, -34.1), (25.6, -34.0), (27.9, -33.0), (29.2, -31.5),
    (31.1, -29.9), (32.1, -28.8), (32.6, -27.5),
]

NAME_MAP = {
    "Richards Bay": "Richards Bay", "Durban": "Durban", "Cape Recife": "Cape Recife",
    "Bird Island": "Bird Island", "St Croix Island": "St Croix", "Riy Banks": "Riy Banks",
    "St Francis": "St Francis", "Fish Hoek": "Fish Hoek", "Sea Point": "Sea Point",
    "Saldhana Bay": "Saldanha Bay",
}


def site_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    bands = ordered_band_columns(list(frame.columns))

    levels = frame[bands].to_numpy(dtype=np.float32)
    energy = 10.0 ** (levels / 10.0)
    low_energy = np.nansum(energy[:, LOW_BANDS], axis=1)
    low_db = 10.0 * np.log10(np.maximum(low_energy, 1e-30))

    stamps = pd.to_datetime(frame["t"], errors="coerce")
    monthly = pd.DataFrame({"month": stamps.dt.to_period("M").astype(str), "low_db": low_db})
    grouped = monthly.groupby("month")["low_db"].agg(["median", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= MIN_MINUTES_PER_MONTH]

    name = path.stem.split("_", 1)[1].replace("_", " ")
    grouped["station_name"] = NAME_MAP.get(name, name)
    return grouped


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(STATIONS)

    rows = []
    for path in sorted(ACOUSTICS.glob("*.parquet")):
        monthly = site_monthly(path)
        rows.append(monthly)
        print(f"{path.stem:<24} {len(monthly):>3} meses con datos suficientes")
    monthly = pd.concat(rows, ignore_index=True)

    monthly = monthly.merge(
        stations[["site_name", "latitude_candidate", "longitude_candidate"]],
        left_on="station_name", right_on="site_name", how="left",
    )

    # Relative activity: percentile rank of the monthly median across all site-months.
    monthly["activity_rank"] = monthly["median"].rank(pct=True)

    # Within-site z-score: comparable over time without trusting the absolute
    # calibration, which differs between recorders.
    grouped = monthly.groupby("station_name")["median"]
    monthly["activity_z"] = (monthly["median"] - grouped.transform("median")) / (
        grouped.transform("std").replace(0, np.nan).fillna(1.0)
    )

    output = DATA_DIR / "site_monthly.csv"
    monthly.to_csv(output, index=False)
    print(f"\nwrote {output.relative_to(PROJECT_DIR)} ({len(monthly)} filas)")

    per_site = (
        monthly.groupby(["station_name", "latitude_candidate", "longitude_candidate"])
        .agg(activity=("activity_rank", "mean"), months=("month", "count"), median_db=("median", "median"))
        .reset_index()
        .sort_values("activity", ascending=False)
    )
    print("\nActividad relativa por sitio (1 = mas activo relativo)")
    print(per_site.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_map(per_site, FIGURES / "activity_map.png")
    plot_heatmap(monthly, FIGURES / "activity_heatmap.png")
    print(f"\nfiguras en {FIGURES.relative_to(PROJECT_DIR)}")


def plot_map(per_site: pd.DataFrame, path: Path) -> None:
    """Overview map with numbered sites, a legend, and a zoom on Algoa Bay."""
    sites = per_site.reset_index(drop=True)
    colours = plt.get_cmap("viridis")(sites["activity"].to_numpy())

    figure, axes = plt.subplots(figsize=(10, 7), dpi=170)
    coast = np.array(COASTLINE)
    axes.plot(coast[:, 0], coast[:, 1], color="#8a8a8a", linewidth=1.2, zorder=1)

    axes.scatter(
        sites["longitude_candidate"], sites["latitude_candidate"],
        c=sites["activity"], cmap="viridis", vmin=sites["activity"].min(), vmax=sites["activity"].max(),
        s=180, edgecolor="black", linewidth=0.7, zorder=3,
    )
    for number, row in sites.iterrows():
        axes.annotate(
            str(number + 1), (row["longitude_candidate"], row["latitude_candidate"]),
            color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=4,
        )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=colours[i], markeredgecolor="black",
                   label=f"{i + 1}. {row['station_name']} ({row['activity']:.2f})")
        for i, row in sites.iterrows()
    ]
    axes.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.92,
                title="Site (relative activity)", title_fontsize=8)

    scatter = axes.scatter([], [], c=[], cmap="viridis")
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.72)
    colorbar.set_label("Relative activity (percentile rank, 31.6-400 Hz)")

    axes.set_xlabel("Longitude (deg E)")
    axes.set_ylabel("Latitude (deg N)")
    axes.set_title("Relative low-frequency acoustic activity by monitoring site\n"
                   "Schematic coastline; cross-site levels partly reflect unverified recorder calibration")
    axes.grid(alpha=0.25, linestyle=":")
    axes.set_aspect("equal", adjustable="box")

    # Zoom on the Algoa Bay cluster, where four sites sit within 0.6 degrees.
    inset = figure.add_axes([0.55, 0.17, 0.30, 0.30])
    inset.plot(coast[:, 0], coast[:, 1], color="#8a8a8a", linewidth=1.0, zorder=1)
    inset.scatter(
        sites["longitude_candidate"], sites["latitude_candidate"],
        c=sites["activity"], cmap="viridis", vmin=sites["activity"].min(), vmax=sites["activity"].max(),
        s=120, edgecolor="black", linewidth=0.6, zorder=3,
    )
    for number, row in sites.iterrows():
        inset.annotate(str(number + 1), (row["longitude_candidate"], row["latitude_candidate"]),
                       color="white", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=4)
    inset.set_xlim(24.4, 26.9)
    inset.set_ylim(-34.4, -33.6)
    inset.set_title("Algoa Bay cluster", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.3, linestyle=":")

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_heatmap(monthly: pd.DataFrame, path: Path) -> None:
    """Within-site monthly anomaly, which does not depend on absolute calibration.

    The x axis is a continuous monthly grid: months without enough data stay
    blank, so gaps are visible instead of being silently closed up.
    """
    frame = monthly.copy()
    frame["period"] = pd.PeriodIndex(frame["month"], freq="M")
    months = pd.period_range(frame["period"].min(), frame["period"].max(), freq="M")

    pivot = frame.pivot_table(index="station_name", columns="period", values="activity_z")
    pivot = pivot.reindex(columns=months).dropna(how="all").sort_index()

    values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    colourmap = plt.get_cmap("RdBu_r").copy()
    colourmap.set_bad("#f2f2f2")

    figure, axes = plt.subplots(figsize=(12, 4.4), dpi=160)
    image = axes.imshow(values, aspect="auto", cmap=colourmap, vmin=-2, vmax=2, interpolation="nearest")
    axes.set_yticks(range(len(pivot.index)))
    axes.set_yticklabels(pivot.index, fontsize=8)

    ticks = [i for i, period in enumerate(months) if period.month in (1, 7)]
    axes.set_xticks(ticks)
    axes.set_xticklabels([str(months[i]) for i in ticks], rotation=90, fontsize=7)

    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("Within-site anomaly (z-score of monthly median)")
    axes.set_title("Low-frequency activity anomaly by site and month\n"
                   "Continuous monthly axis; blank = fewer than 360 usable minutes")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
