#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent

LATEST_DIR = BASE / "downloads" / "latest"
LATEST_PDF = LATEST_DIR / "latest.pdf"
LATEST_META = LATEST_DIR / "latest.json"

STATE_DIR = BASE / "ocr_state"
LAST_PROCESSED = STATE_DIR / "last_processed.json"

# Your OCR script lives in the ocr_scraped_logs directory (not scrape_logs)
OCR_SCRIPT = (
    BASE.parent / "ocr_scraped_logs" / "unified_table_ocr_fast.py"
).resolve()

DPI = "600"
OCR_BACKEND = "paddle"

# If you want to force the venv python here too, set it explicitly.
# Otherwise sys.executable is fine because systemd will call this using the venv python.
# VENV_PY = Path("/home/hacker/venvs/ucmpolice-ocr/bin/python")


def looks_like_pdf(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except FileNotFoundError:
        return False


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_last() -> dict:
    if LAST_PROCESSED.exists():
        return json.loads(LAST_PROCESSED.read_text("utf-8"))
    return {}


def save_last(obj: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PROCESSED.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_ocr(input_path: Path) -> subprocess.CompletedProcess:
    # This matches your working manual command:
    # python unified_table_ocr_fast.py "/path/to/downloads/latest" --dpi 600 --ocr paddle
    cmd = [
        sys.executable,  # or str(VENV_PY) if you prefer hard-pinning
        str(OCR_SCRIPT),
        str(input_path),
        "--dpi",
        DPI,
        "--ocr",
        OCR_BACKEND,
    ]
    return subprocess.run(cmd, text=True, capture_output=True)


def main() -> int:
    if not OCR_SCRIPT.exists():
        print(f"OCR script not found: {OCR_SCRIPT}")
        return 2

    if not LATEST_PDF.exists():
        print(f"No latest PDF found at {LATEST_PDF}; skipping OCR.")
        return 0

    # Basic validity checks before OCR
    if LATEST_PDF.stat().st_size < 2048 or not looks_like_pdf(LATEST_PDF):
        print(
            "latest.pdf does not appear to be a valid PDF (too small or missing %PDF- header); skipping OCR."
        )
        return 0

    current_hash = sha256_file(LATEST_PDF)
    last = load_last()
    if last.get("hash") == current_hash:
        print("OCR already processed the current latest.pdf (hash match); skipping.")
        return 0

    # Your current OCR invocation uses the directory containing PDFs:
    # .../downloads/latest
    res = run_ocr(LATEST_DIR)
    if res.returncode != 0:
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        print("OCR failed. Output:\n")
        print(combined.strip())
        return res.returncode

    # Success
    if res.stdout:
        print(res.stdout.strip())

    # Mark processed only after OCR succeeded
    downloaded_at = None
    canonical = None
    if LATEST_META.exists():
        try:
            meta = json.loads(LATEST_META.read_text("utf-8"))
            downloaded_at = meta.get("downloaded_at")
            canonical = meta.get("canonical")
        except Exception:
            pass

    save_last(
        {
            "latest_pdf": str(LATEST_PDF.resolve()),
            "latest_dir": str(LATEST_DIR.resolve()),
            "hash": current_hash,
            "canonical": canonical,
            "downloaded_at": downloaded_at,
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    print("OCR complete; state updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
