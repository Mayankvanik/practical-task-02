"""
ingestion/pdf_downloader.py — Downloads the NASA SE Handbook PDF or accepts a user-uploaded file.
"""
import logging
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_pdf(url: str, dest: Path, force: bool = False) -> Path:
    """
    Download PDF from url to dest.
    Skips download if file already exists unless force=True.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        logger.info(f"PDF already exists at {dest}, skipping download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading PDF from {url} → {dest}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"PDF saved to {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")
    return dest


def save_uploaded_pdf(uploaded_bytes: bytes, dest: Path) -> Path:
    """Save bytes uploaded via Streamlit to dest path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(uploaded_bytes)
    logger.info(f"Saved uploaded PDF to {dest} ({len(uploaded_bytes) / 1_000_000:.1f} MB)")
    return dest
