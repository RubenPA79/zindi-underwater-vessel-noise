#!/usr/bin/env python3
"""Download the Zindi challenge files with a session token.

Zindi exposes no CLI and the file endpoints return 401 without a signed-in
session. The token is read from the ``ZINDI_TOKEN`` environment variable and is
never written to disk or printed.

Examples
--------
Check that the token works (fetches only the 10 KB notebook)::

    ZINDI_TOKEN=... python src/download_data.py --check

Download everything into ``data/``::

    ZINDI_TOKEN=... python src/download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

COMPETITION_ID = "french-embassy-underwater-vessel-noise-challenge"
API_ROOT = "https://api.zindi.world/v1/competitions"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"
CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 120

# name -> minimum plausible size in bytes. Guards against an error document
# being saved as if it were the dataset (a 200 with a JSON body looks like a
# download until you check the size).
FILES: dict[str, int] = {
    "South_African_Real_Acoustics.zip": 80_000_000,
    "Test.csv": 2_000_000,
    "sample_submission.csv": 40_000,
    "maritime_starter_notebook.ipynb": 5_000,
}

# Cheapest file to probe the token with.
PROBE_FILE = "maritime_starter_notebook.ipynb"


class DownloadError(RuntimeError):
    """Raised when a file cannot be fetched or looks corrupt."""


def build_url(name: str, token: str) -> str:
    return f"{API_ROOT}/{COMPETITION_ID}/files/{name}?auth_token={token}"


def _raise_for_auth(response: requests.Response, name: str) -> None:
    if response.status_code in (401, 403):
        raise DownloadError(
            f"{name}: HTTP {response.status_code}. The session token is missing, "
            "invalid or expired. Sign in at https://zindi.world and copy a fresh one."
        )
    response.raise_for_status()


def probe(token: str) -> None:
    """Verify the token without downloading a full dataset."""
    with requests.get(build_url(PROBE_FILE, token), stream=True, timeout=TIMEOUT_SECONDS) as response:
        _raise_for_auth(response, PROBE_FILE)
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            raise DownloadError(f"{PROBE_FILE}: server returned JSON instead of a file: {response.text[:200]}")
        print(f"Token OK: {PROBE_FILE} is reachable ({response.headers.get('content-length', '?')} bytes).")


def download(name: str, token: str, out_dir: Path, min_bytes: int) -> Path:
    target = out_dir / name
    partial = target.with_name(target.name + ".part")
    with requests.get(build_url(name, token), stream=True, timeout=TIMEOUT_SECONDS) as response:
        _raise_for_auth(response, name)
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            raise DownloadError(f"{name}: server returned JSON instead of a file: {response.text[:200]}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)

    size = partial.stat().st_size
    if size < min_bytes:
        partial.unlink()
        raise DownloadError(f"{name}: got {size} bytes, expected at least {min_bytes}. File discarded.")
    partial.replace(target)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="destination directory")
    parser.add_argument("--check", action="store_true", help="only verify the token, do not download")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--only", nargs="+", choices=sorted(FILES), help="download just these files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("ZINDI_TOKEN")
    if not token:
        print("ZINDI_TOKEN is not set. Export the session token first; it is never stored.", file=sys.stderr)
        return 2

    try:
        if args.check:
            probe(token)
            return 0

        args.out_dir.mkdir(parents=True, exist_ok=True)
        wanted = args.only or list(FILES)
        for name in wanted:
            target = args.out_dir / name
            if target.exists() and not args.force:
                print(f"skip {name} (already present, {target.stat().st_size} bytes)")
                continue
            print(f"downloading {name} ...")
            saved = download(name, token, args.out_dir, FILES[name])
            print(f"  saved {saved.relative_to(PROJECT_ROOT)} ({saved.stat().st_size} bytes)")
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except requests.RequestException as error:
        print(f"network error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
