"""Join the acoustic activity index to AIS vessel density from GMTDS.

For every site-month in ``data/site_monthly.csv`` we query the Global Maritime
Traffic Density Service (GMTDS) WMS at the station's coordinates and read the
time-density value for that month (hours per 1 km2 cell, all vessel categories).
The result is an independent, external covariate for the acoustic map.

Caveats carried into the write-up: station coordinates are working site-level
candidates, the raster cell is 1 km, and the acoustic export clock has not been
validated against UTC.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.ais_density
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
FIGURES = PROJECT_DIR / "entrega" / "figures"

WMS = "https://gmtds.maplarge.com/ogc/wms"
LAYER = "ais:density"
BOX_DEGREES = 0.01
PAUSE_SECONDS = 0.15


def query_density(session: requests.Session, lat: float, lon: float, month: str) -> float:
    """Vessel time-density (hours per km2) at one point and month."""
    bbox = f"{lon - BOX_DEGREES},{lat - BOX_DEGREES},{lon + BOX_DEGREES},{lat + BOX_DEGREES}"
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetFeatureInfo",
        "layers": LAYER,
        "styles": "default",
        "SRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": 1,
        "HEIGHT": 1,
        "X": 0,
        "Y": 0,
        "INFO_FORMAT": "text/plain",
        "time": f"{month}-01T00:00:00Z",
    }
    response = session.get(WMS, params=params, timeout=60)
    response.raise_for_status()
    lines = [line for line in response.text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"unexpected response: {response.text[:200]!r}")
    return float(lines[-1].split(",")[-1])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(DATA_DIR / "site_monthly.csv")
    output = DATA_DIR / "ais_density.csv"

    if output.exists():
        cache = pd.read_csv(output)
        done = set(zip(cache["station_name"], cache["month"]))
    else:
        cache = pd.DataFrame(columns=["station_name", "month", "ais_density"])
        done = set()

    session = requests.Session()
    session.headers["User-Agent"] = "underwater-vessel-noise/1.0 (Zindi challenge research)"
    rows = []
    for _, row in monthly.iterrows():
        key = (row["station_name"], row["month"])
        if key in done:
            continue
        try:
            value = query_density(session, row["latitude_candidate"], row["longitude_candidate"], row["month"])
        except Exception as error:  # noqa: BLE001 - one bad query must not stop the run
            print(f"  {key[0]} {key[1]}: FAILED ({error})")
            value = np.nan
        rows.append({"station_name": key[0], "month": key[1], "ais_density": value})
        time.sleep(PAUSE_SECONDS)

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(output, index=False)
    print(f"{output.name}: {len(cache)} filas, {cache['ais_density'].notna().sum()} con densidad")

    merged = monthly.merge(cache, on=["station_name", "month"], how="left")
    merged = merged[merged["ais_density"].notna() & (merged["ais_density"] > 0)]
    if len(merged) < 10:
        raise SystemExit("not enough matched site-months to analyse")

    overall = stats.spearmanr(merged["median"], merged["ais_density"])
    print(f"\nSpearman global (nivel dB vs densidad AIS): rho={overall.statistic:.3f}, p={overall.pvalue:.3g}")

    per_site = []
    for name, group in merged.groupby("station_name"):
        if len(group) >= 5 and group["ais_density"].nunique() > 1:
            result = stats.spearmanr(group["median"], group["ais_density"])
            per_site.append({"station_name": name, "months": len(group), "rho": result.statistic, "p": result.pvalue})
    per_site = pd.DataFrame(per_site).sort_values("rho", ascending=False)

    # Within-site anomalies versus within-site AIS anomalies: the calibration-free view.
    merged["ais_z"] = merged.groupby("station_name")["ais_density"].transform(
        lambda values: (values - values.mean()) / (values.std() or np.nan)
    )
    paired = merged.dropna(subset=["ais_z", "activity_z"])
    anomaly = stats.spearmanr(paired["activity_z"], paired["ais_z"])
    print(f"Spearman de anomalias intra-sitio: rho={anomaly.statistic:.3f}, p={anomaly.pvalue:.3g} (n={len(paired)})")

    if not per_site.empty:
        print("\nPor sitio:")
        print(per_site.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot(merged, per_site, overall, anomaly)
    merged.to_csv(DATA_DIR / "ais_density_merged.csv", index=False)


def plot(merged: pd.DataFrame, per_site: pd.DataFrame, overall, anomaly) -> None:
    """Three panels: the confounded view, the honest one, and the per-site split."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=160)
    sites = merged["station_name"].unique()
    colourmap = plt.get_cmap("tab10")

    # Panel 1: raw levels. Sites sit in separate blobs, so this view mixes the
    # traffic relation with the recorder-calibration differences between sites.
    for index, name in enumerate(sites):
        group = merged[merged["station_name"] == name]
        axes[0].scatter(group["ais_density"], group["median"], s=20, alpha=0.85,
                        color=colourmap(index % 10), label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("AIS time-density (hours per km2, log)")
    axes[0].set_ylabel("Median low-frequency level (dB)")
    axes[0].set_title(f"Between sites: clustered, confounded\n(Spearman rho={overall.statistic:.2f})",
                      fontsize=10)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    axes[0].grid(alpha=0.25, linestyle=":")

    # Panel 2: within-site anomalies. Removing each site's own mean and spread
    # takes the calibration out, which is the comparison that means something.
    paired = merged.dropna(subset=["ais_z", "activity_z"])
    for index, name in enumerate(sites):
        group = paired[paired["station_name"] == name]
        if group.empty:
            continue
        axes[1].scatter(group["ais_z"], group["activity_z"], s=26, alpha=0.9,
                        color=colourmap(index % 10))
    if len(paired) > 2:
        slope, intercept = np.polyfit(paired["ais_z"], paired["activity_z"], 1)
        grid = np.linspace(paired["ais_z"].min(), paired["ais_z"].max(), 50)
        axes[1].plot(grid, slope * grid + intercept, color="#333", linewidth=1.4, linestyle="--")
    axes[1].axhline(0, color="#999", linewidth=0.7)
    axes[1].axvline(0, color="#999", linewidth=0.7)
    axes[1].set_xlabel("AIS anomaly within site (z-score)")
    axes[1].set_ylabel("Acoustic anomaly within site (z-score)")
    axes[1].set_title(f"Within sites: the honest test\n(Spearman rho={anomaly.statistic:.2f})", fontsize=10)
    axes[1].grid(alpha=0.25, linestyle=":")

    if not per_site.empty:
        ordered = per_site.sort_values("rho")
        colours = ["#c0392b" if value < 0 else "#1b6ca8" for value in ordered["rho"]]
        axes[2].barh(ordered["station_name"], ordered["rho"], color=colours)
        axes[2].axvline(0, color="#444", linewidth=0.8)
        axes[2].set_xlabel("Spearman rho")
        axes[2].set_title("Per site (months with data)", fontsize=10)
        axes[2].tick_params(labelsize=8)
        axes[2].grid(axis="x", alpha=0.25, linestyle=":")

    figure.tight_layout()
    figure.savefig(FIGURES / "ais_vs_activity.png")
    plt.close(figure)
    print("\nwrote entrega/figures/ais_vs_activity.png")


if __name__ == "__main__":
    main()
