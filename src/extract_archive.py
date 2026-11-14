"""Extract the acoustic archive into ``data/``.

``Expand-Archive`` is unreliable under the Windows sandbox, so extraction goes
through :mod:`zipfile`. Members are checked to stay inside the target directory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARCHIVE_NAME = "South_African_Real_Acoustics.zip"


def main() -> None:
    archive_path = DATA_DIR / ARCHIVE_NAME
    target = DATA_DIR.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise RuntimeError(f"archive member escapes the data directory: {member.filename}")
        archive.extractall(target)

    parquet_files = sorted((target / "acoustics").glob("*.parquet"))
    print(f"extracted {len(parquet_files)} parquet files into data/acoustics/")
    total = sum(path.stat().st_size for path in parquet_files)
    print(f"total parquet size: {total / 1e6:.1f} MB")
    for path in parquet_files:
        print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
