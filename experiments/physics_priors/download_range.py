"""Download a large HTTP object with parallel byte-range requests.

This is intentionally dependency-free so it can run in the native conda Python
on remote training hosts before the project environment is fully prepared.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import os
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def _head_size(url: str, timeout: float) -> int:
    req = Request(url, method="HEAD", headers={"User-Agent": "VARC-range-downloader"})
    with urlopen(req, timeout=timeout) as resp:
        size = resp.headers.get("Content-Length")
        if size is None:
            raise RuntimeError("HEAD response did not include Content-Length")
        return int(size)


def _download_part(
    *,
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: float,
) -> int:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return expected

    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp_path.unlink(missing_ok=True)
            req = Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "VARC-range-downloader",
                },
            )
            with urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out:
                status = getattr(resp, "status", None)
                if status != 206:
                    raise RuntimeError(f"expected HTTP 206 for range, got {status}")
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            actual = tmp_path.stat().st_size
            if actual != expected:
                raise RuntimeError(f"part size mismatch: expected {expected}, got {actual}")
            tmp_path.replace(part_path)
            return actual
        except (OSError, RuntimeError, URLError) as exc:
            last_error = exc
            sleep_s = min(2 ** attempt, 30)
            print(
                f"part {part_path.name} failed attempt {attempt}/{retries}: {exc}; "
                f"sleeping {sleep_s}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"failed to download {part_path.name}") from last_error


def _merge_parts(part_paths: list[Path], output: Path, expected_size: int) -> None:
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.unlink(missing_ok=True)
    with tmp_output.open("wb") as out:
        for part_path in part_paths:
            with part_path.open("rb") as part:
                while True:
                    chunk = part.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    actual = tmp_output.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"merged size mismatch: expected {expected_size}, got {actual}")
    tmp_output.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--expected-size", type=int, default=None)
    parser.add_argument("--temp-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.parts < 1:
        raise SystemExit("--parts must be >= 1")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_dir or output.parent / f".{output.name}.parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    expected_size = args.expected_size or _head_size(args.url, args.timeout)
    if output.exists() and output.stat().st_size == expected_size:
        print(f"already complete: {output} ({expected_size} bytes)", flush=True)
        return

    chunk_size = math.ceil(expected_size / args.parts)
    ranges: list[tuple[int, int, Path]] = []
    for idx in range(args.parts):
        start = idx * chunk_size
        if start >= expected_size:
            break
        end = min(start + chunk_size - 1, expected_size - 1)
        ranges.append((start, end, temp_dir / f"part_{idx:03d}"))

    print(
        f"downloading {expected_size} bytes into {len(ranges)} parts -> {output}",
        flush=True,
    )
    start_time = time.time()
    with futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        future_to_range = {
            executor.submit(
                _download_part,
                url=args.url,
                part_path=part_path,
                start=start,
                end=end,
                retries=args.retries,
                timeout=args.timeout,
            ): (idx, start, end, part_path)
            for idx, (start, end, part_path) in enumerate(ranges)
        }
        completed_bytes = 0
        for future in futures.as_completed(future_to_range):
            idx, start, end, part_path = future_to_range[future]
            part_bytes = future.result()
            completed_bytes += part_bytes
            elapsed = max(time.time() - start_time, 1e-6)
            mbps = completed_bytes / elapsed / (1024 * 1024)
            print(
                f"part {idx:03d} complete bytes={part_bytes} "
                f"range={start}-{end} aggregate={mbps:.2f} MiB/s",
                flush=True,
            )

    part_paths = [part_path for _, _, part_path in ranges]
    _merge_parts(part_paths, output, expected_size)
    print(
        f"complete: {output} size={output.stat().st_size} "
        f"elapsed={time.time() - start_time:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
