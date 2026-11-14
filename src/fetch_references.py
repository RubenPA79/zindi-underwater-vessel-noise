"""Collect vessel reference spectra from HearMyShip.

The catalogue pages embed a JSON array of vessels, each with ``thirdOctaveData``
holding already-computed band levels in dB. That is exactly what the challenge
needs, so nothing is downloaded per vessel: one request per catalogue page.

Only the bands of the challenge grid are kept; all 25 must be present.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.fetch_references --pages 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.layout import BAND_CENTRES, N_BANDS

BASE_URL = "https://hearmyship.fer.hr/"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT = DATA_DIR / "vessel_levels.csv"
POLITE_DELAY_SECONDS = 0.3
BAND_TOLERANCE = 0.02
META_COLUMNS = ["name", "category", "speed", "length", "pressure"]
BAND_COLUMNS = [f"b{band:02d}" for band in range(N_BANDS)]
COLUMNS = META_COLUMNS + BAND_COLUMNS


def page_url(page: int) -> str:
    return BASE_URL if page == 1 else f"{BASE_URL}?page={page}"


def extract_vessel_array(html: str) -> list[dict]:
    """Parse the embedded JSON array of vessels by matching brackets outside strings."""
    start = html.find('[{"id"')
    if start < 0:
        raise ValueError("vessel array not found on the page")

    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(html)):
        character = html[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return json.loads(html[start : position + 1])
    raise ValueError("unterminated vessel array")


def levels_from_third_octave(data: dict[str, float]) -> np.ndarray | None:
    """Map nominal third-octave names onto the challenge grid."""
    levels = np.full(N_BANDS, np.nan)
    for name, value in data.items():
        frequency = float(name)
        band = int(np.argmin(np.abs(BAND_CENTRES - frequency)))
        if abs(BAND_CENTRES[band] - frequency) <= BAND_CENTRES[band] * BAND_TOLERANCE:
            levels[band] = float(value)
    return None if np.isnan(levels).any() else levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=int, default=64)
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "underwater-vessel-noise/0.1 (Zindi challenge research)"

    rows: list[list] = []
    skipped = 0
    for page in range(1, args.pages + 1):
        try:
            response = session.get(page_url(page), timeout=60)
            response.raise_for_status()
            vessels = extract_vessel_array(response.text)
        except (requests.RequestException, ValueError) as error:
            print(f"page {page}: {error}")
            continue

        added = 0
        for vessel in vessels:
            levels = levels_from_third_octave(vessel.get("thirdOctaveData") or {})
            if levels is None:
                skipped += 1
                continue
            rows.append(
                [
                    vessel.get("image", "").replace("\\", "/").split("/")[-1].replace("_Camera.JPG", ""),
                    vessel.get("category"),
                    vessel.get("speed"),
                    vessel.get("length"),
                    vessel.get("pressure"),
                    *levels.tolist(),
                ]
            )
            added += 1
        print(f"page {page:>2}: {len(vessels)} vessels, {added} usable")
        time.sleep(POLITE_DELAY_SECONDS)

    frame = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(subset="name")
    frame.to_csv(OUTPUT, index=False)
    print(f"\nwrote {len(frame)} vessels to {OUTPUT.name} (skipped {skipped} incomplete)")
    print(frame["category"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
