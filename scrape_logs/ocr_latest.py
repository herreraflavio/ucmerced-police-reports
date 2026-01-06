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

# OCR script lives in ocr_scraped_logs directory (not scrape_logs)
OCR_SCRIPT = (BASE.parent / "ocr_scraped_logs" / "unified_table_ocr_fast.py").resolve()

# Force a stable OCR output directory so downstream steps can find outputs reliably.
OCR_OUT_DIR = (BASE.parent / "ocr_scraped_logs" / "output" / "ocr_output").resolve()

DPI = "600"
OCR_BACKEND = "paddle"

# Optional: hard-pin venv python. Highly recommended if you run this manually sometimes.
# If systemd already runs this via the venv python, sys.executable will be the venv python anyway.
VENV_PY = Path("/home/hacker/venvs/ucmpolice-ocr/bin/python")
USE_VENV_PY = VENV_PY.exists()

# Optional post-processing script (you create this).
# It will be called with the OCR CSV path as argv[1].
POST_SCRIPT = (BASE.parent / "ocr_scraped_logs" / "post_process.py").resolve()


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
    if not LAST_PROCESSED.exists():
        return {}
    try:
        txt = LAST_PROCESSED.read_text("utf-8").strip()
        if not txt:
            return {}
        return json.loads(txt)
    except Exception:
        # Corrupt/partial file; treat as no state
        return {}


def save_last(obj: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LAST_PROCESSED.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(LAST_PROCESSED)


def _python_exe() -> str:
    if USE_VENV_PY:
        return str(VENV_PY)
    return sys.executable


def run_ocr(input_path: Path) -> subprocess.CompletedProcess:
    # Matches your working command, but adds --out-dir so output location is deterministic.
    # /home/hacker/venvs/ucmpolice-ocr/bin/python unified_table_ocr_fast.py <dir> --dpi 600 --ocr paddle --out-dir <fixed>
    cmd = [
        _python_exe(),
        str(OCR_SCRIPT),
        str(input_path),
        "--dpi",
        DPI,
        "--ocr",
        OCR_BACKEND,
        "--out-dir",
        str(OCR_OUT_DIR),
    ]
    return subprocess.run(cmd, text=True, capture_output=True)


def expected_csv_path() -> Path:
    """
    OCR script writes: <out-dir>/<stem>/<stem>.csv
    For latest/latest.pdf, stem is 'latest' -> <out-dir>/latest/latest.csv
    """
    stem = LATEST_PDF.stem  # "latest"
    return OCR_OUT_DIR / stem / f"{stem}.csv"


def run_post_process(csv_path: Path) -> subprocess.CompletedProcess:
    # Post script is optional; if it doesn't exist, we just skip.
    cmd = [_python_exe(), str(POST_SCRIPT), str(csv_path)]
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
        print("latest.pdf does not appear valid (too small or missing %PDF- header); skipping OCR.")
        return 0

    current_hash = sha256_file(LATEST_PDF)
    last = load_last()
    if last.get("hash") == current_hash:
        print("OCR already processed the current latest.pdf (hash match); skipping.")
        return 0

    # Run OCR on the directory containing latest.pdf
    res = run_ocr(LATEST_DIR)
    if res.returncode != 0:
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        print("OCR failed. Output:\n")
        print(combined.strip())
        return res.returncode

    if res.stdout:
        print(res.stdout.strip())

    # Verify expected output exists (helps catch path/out-dir issues early)
    csv_path = expected_csv_path()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"OCR reported success, but expected CSV was not found or empty: {csv_path}")
        return 3

    # OPTIONAL: post-processing step
    post_rc = None
    post_out = None
    if POST_SCRIPT.exists():
        pp = run_post_process(csv_path)
        post_rc = pp.returncode
        post_out = ((pp.stdout or "") + "\n" + (pp.stderr or "")).strip()
        if pp.returncode != 0:
            print("Post-processing failed. Output:\n")
            print(post_out)
            return pp.returncode
        else:
            if post_out:
                print(post_out)
    else:
        print(f"Post-processing script not found (skipping): {POST_SCRIPT}")

    # Pull optional metadata from latest.json
    downloaded_at = None
    canonical = None
    if LATEST_META.exists():
        try:
            meta = json.loads(LATEST_META.read_text("utf-8"))
            downloaded_at = meta.get("downloaded_at")
            canonical = meta.get("canonical")
        except Exception:
            pass

    # Mark processed only after OCR (+ post-processing if enabled) succeeded
    save_last(
        {
            "latest_pdf": str(LATEST_PDF.resolve()),
            "latest_dir": str(LATEST_DIR.resolve()),
            "hash": current_hash,
            "canonical": canonical,
            "downloaded_at": downloaded_at,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "ocr_out_dir": str(OCR_OUT_DIR),
            "ocr_csv": str(csv_path.resolve()),
            "post_script": str(POST_SCRIPT),
            "post_returncode": post_rc,
        }
    )

    print("OCR complete; state updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
