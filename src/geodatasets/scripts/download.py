"""Download and extract files from a Zenodo record, with checksum validation and progress bars."""

import argparse
import hashlib
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
CHUNK = 1024 * 1024  # 1 MB
# Optionally add files to skip
SKIP_FILES: set[str] = set()


def md5_of(path: Path) -> str:
    """Calculate the MD5 checksum of a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_matches(path: Path, expected: str) -> bool:
    """Check if the file's MD5 checksum matches the expected value."""
    actual = md5_of(path)
    if actual != expected:
        log.error(
            "Checksum mismatch for %s (expected %s, got %s)",
            path.name,
            expected,
            actual,
        )
    return actual == expected


def strip_md5_prefix(raw: str) -> str | None:
    """If the checksum string has an MD5 prefix, strip it and return the rest, otherwise None."""
    return raw.removeprefix("md5:") if raw.startswith("md5:") else None


def fetch_record_files(record_id: str) -> list[dict]:
    """Fetch the list of files associated with a zenodo record."""
    resp = requests.get(f"{ZENODO_API}/{record_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def is_fresh(path: Path, expected_md5: str | None) -> bool:
    """Return True if the file exists and its checksum is valid."""
    if not path.exists():
        return False
    if expected_md5 and not checksum_matches(path, expected_md5):
        return False
    log.info("Already verified — skipping %s", path.name)
    return True


def download(url: str, dest: Path) -> None:
    """Download a file from the given URL to the destination path, showing a progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with (
        dest.open("wb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK):
            fh.write(chunk)
            bar.update(len(chunk))


def download_file(url: str, dest: Path, expected_md5: str | None) -> None:
    """Download a file unless it doesn't exist or fails checksum validation."""
    if is_fresh(dest, expected_md5):
        return

    log.info("Downloading %s", dest.name)
    download(url, dest)

    if expected_md5 and not checksum_matches(dest, expected_md5):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum failed for {dest.name} — file removed.")


def extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip archive to the destination directory."""
    log.info("Extracting %s → %s", archive.name, dest)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def download_record(record_id: str, output_dir: Path, *, extract: bool = True) -> None:
    """Download all files from a Zenodo record, optionally extracting zip files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = fetch_record_files(record_id)
    log.info("Found %d file(s) in record %s", len(files), record_id)

    for entry in files:
        if entry["key"] in SKIP_FILES:
            log.info("Skipping %s", entry["key"])
            continue

        dest = output_dir / entry["key"]
        download_file(entry["links"]["self"], dest, strip_md5_prefix(entry.get("checksum", "")))

        if extract and dest.suffix == ".zip":
            extract_zip(dest, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record-id", default="4172871", help="Zenodo record ID")
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Download destination")
    p.add_argument("--no-extract", dest="extract", action="store_false", help="Skip ZIP extraction")
    return p.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    download_record(args.record_id, args.output_dir, extract=args.extract)


if __name__ == "__main__":
    main()
