"""Download a large file over several parallel HTTP range requests.

Zenodo throttles a single connection to roughly 140 KB/s after a couple of
gigabytes, which turns a 1.6 GB archive into a three-hour download. Range
requests are supported, so the file is split into chunks fetched concurrently
and stitched together afterwards.

The result is verified against the expected length before it replaces the target.

Usage::

    python -m src.fetch_chunked --url <url> --out data/vtuad/x.zip --chunks 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CHUNK_SIZE = 8 * 1024 * 1024  # read granularity inside a chunk
TIMEOUT = 120


def remote_size(session: requests.Session, url: str) -> int:
    response = session.get(url, headers={"Range": "bytes=0-0"}, timeout=TIMEOUT, stream=True)
    response.raise_for_status()
    content_range = response.headers.get("content-range", "")
    response.close()
    if "/" not in content_range:
        raise RuntimeError(f"server did not report a total size: {content_range!r}")
    return int(content_range.split("/")[-1])


def fetch_chunk(session: requests.Session, url: str, start: int, end: int, path: Path) -> int:
    """Download bytes [start, end] inclusive into ``path``."""
    written = 0
    with session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=TIMEOUT, stream=True) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code} for range {start}-{end}")
        with path.open("wb") as handle:
            for block in response.iter_content(chunk_size=CHUNK_SIZE):
                if block:
                    handle.write(block)
                    written += len(block)
    expected = end - start + 1
    if written != expected:
        raise RuntimeError(f"chunk {start}-{end}: got {written} bytes, expected {expected}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunks", type=int, default=8)
    args = parser.parse_args()

    session = requests.Session()
    size = remote_size(session, args.url)
    print(f"remote size: {size / 1e6:.1f} MB, {args.chunks} chunks")

    step = size // args.chunks
    ranges = []
    for index in range(args.chunks):
        start = index * step
        end = size - 1 if index == args.chunks - 1 else (start + step - 1)
        ranges.append((start, end))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    parts = [args.out.with_name(f"{args.out.name}.part{index:02d}") for index in range(args.chunks)]

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.chunks) as pool:
            futures = {
                pool.submit(fetch_chunk, session, args.url, start, end, parts[index]): index
                for index, (start, end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                completed += future.result()
                print(f"  progress: {completed / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
    except Exception:
        for part in parts:
            part.unlink(missing_ok=True)
        raise

    with args.out.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(CHUNK_SIZE):
                    target.write(block)
            part.unlink()

    final = args.out.stat().st_size
    if final != size:
        args.out.unlink()
        raise RuntimeError(f"assembled {final} bytes, expected {size}")
    print(f"ok: {args.out} ({final / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
