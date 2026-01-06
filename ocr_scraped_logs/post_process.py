#!/usr/bin/env python3
import argparse
import csv
import hashlib
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple


DEFAULT_CASE_COL = "Case/Incident"

# The existing schema uses "Location" (capital L). We keep it, but overwrite it with normalized.
LOCATION_COL = "Location"
RAW_LOCATION_COL = "raw_location"
NORM_LOCATION_COL = "location"

# Columns used to fingerprint the record when Case/Incident is empty (or even when it isn't).
# Excludes Case/Incident on purpose so the same record won't duplicate if that field changes later.
DEFAULT_FP_COLS = [
    "Nature",
    "Reported",
    "Occured",
    "Location",
    "Disposition",
    "On Campus?",
    "Area",
]


def norm(s: str) -> str:
    """Normalize text for stable comparisons."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200e", "").replace("\ufeff", "")  # common invisible chars
    s = " ".join(s.strip().split())                   # collapse whitespace
    return s


def clean_raw(s: str) -> str:
    """Preserve as 'raw' but remove common invisibles and trim edges."""
    if s is None:
        return ""
    s = str(s).replace("\u200e", "").replace("\ufeff", "")
    return s.strip()


def fingerprint_row(row: dict, fp_cols: list[str]) -> str:
    """Create a stable SHA256 fingerprint based on selected columns."""
    parts = [norm(row.get(c, "")).lower() for c in fp_cols]
    payload = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _add_backend_to_syspath(backend_root: Optional[str]) -> Path:
    """
    Ensure uc_merced_campus_event_api_backend is on sys.path so we can import:
      from helper.normalize_location import normalize_event_location
    """
    candidates: list[Path] = []

    if backend_root:
        candidates.append(Path(backend_root).expanduser().resolve())

    script_dir = Path(__file__).resolve().parent

    # Common layouts:
    # .../api/ucmerced-police-reports/post_process.py
    # .../api/uc_merced_campus_event_api_backend/helper/normalize_location.py
    candidates.extend([
        script_dir / "uc_merced_campus_event_api_backend",
        script_dir.parent / "uc_merced_campus_event_api_backend",
        script_dir.parents[1] / "uc_merced_campus_event_api_backend" if len(script_dir.parents) > 1 else script_dir,
        script_dir.parents[2] / "uc_merced_campus_event_api_backend" if len(script_dir.parents) > 2 else script_dir,
        Path("/mnt/usb/ucmercedhub/api/uc_merced_campus_event_api_backend"),
    ])

    for p in candidates:
        try:
            if (p / "helper" / "normalize_location.py").exists():
                sys.path.insert(0, str(p))
                return p
        except Exception:
            continue

    raise RuntimeError(
        "Could not locate 'uc_merced_campus_event_api_backend'. "
        "Provide --backend-root /path/to/uc_merced_campus_event_api_backend"
    )


def load_normalizer(backend_root: Optional[str]) -> Tuple[Callable[[str], str], Path]:
    backend_path = _add_backend_to_syspath(backend_root)
    # Import after sys.path injection
    from helper.normalize_location import normalize_event_location  # type: ignore
    return normalize_event_location, backend_path


def apply_location_normalization(row: dict, normalize_event_location: Callable[[str], str]) -> dict:
    """
    - raw_location gets the pre-normalized Location (cleaned minimally)
    - location gets normalized
    - Location (existing) is overwritten with normalized (fallback to raw if normalizer returns empty)
    """
    raw = row.get(RAW_LOCATION_COL, "")
    if not raw:
        raw = row.get(LOCATION_COL, "")
    raw = clean_raw(raw)

    normalized = ""
    if raw:
        try:
            normalized = normalize_event_location(raw) or ""
        except Exception:
            normalized = ""

    normalized = norm(normalized)
    row[RAW_LOCATION_COL] = raw
    row[NORM_LOCATION_COL] = normalized
    row[LOCATION_COL] = normalized or raw  # keep legacy column usable

    return row


def normalize_existing_aggregate_in_place(
    agg_path: Path,
    case_col: str,
    fp_cols: list[str],
    normalize_event_location: Optional[Callable[[str], str]],
    force_rewrite: bool = True,
) -> tuple[set[str], set[str], list[str] | None]:
    """
    If agg exists:
      - rewrites it to ensure RAW_LOCATION_COL + NORM_LOCATION_COL exist in header
      - normalizes location fields for all existing rows (if normalizer available)
      - returns dedupe sets computed from the normalized rows
    """
    if not agg_path.exists() or agg_path.stat().st_size == 0:
        return set(), set(), None

    existing_case_keys: set[str] = set()
    existing_fps: set[str] = set()

    ensure_parent_dir(agg_path)

    with agg_path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            return set(), set(), None

        fieldnames = list(reader.fieldnames)

        # Validate fingerprint columns exist in aggregate header
        missing = [c for c in fp_cols if c not in fieldnames]
        if missing:
            raise RuntimeError(
                f"Aggregated file exists but is missing fingerprint columns {missing}: {agg_path}"
            )
        if case_col not in fieldnames:
            raise RuntimeError(
                f"Aggregated file exists but is missing '{case_col}' column: {agg_path}"
            )

        out_fields = fieldnames[:]
        for c in (RAW_LOCATION_COL, NORM_LOCATION_COL):
            if c not in out_fields:
                out_fields.append(c)

        # Decide whether to rewrite:
        # - If we want to normalize existing rows, we must rewrite (header + updated rows).
        # - If header is missing new cols, we must rewrite.
        header_missing_new_cols = any(c not in fieldnames for c in (RAW_LOCATION_COL, NORM_LOCATION_COL))
        rewrite_needed = force_rewrite or header_missing_new_cols

        tmp_path = agg_path.with_name(agg_path.stem + ".tmp" + agg_path.suffix)

        if rewrite_needed:
            with tmp_path.open("w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
                writer.writeheader()

                for row in reader:
                    row = dict(row)

                    if normalize_event_location is not None:
                        row = apply_location_normalization(row, normalize_event_location)
                    else:
                        # Still ensure columns exist deterministically
                        raw = clean_raw(row.get(LOCATION_COL, ""))
                        row[RAW_LOCATION_COL] = row.get(RAW_LOCATION_COL, raw) or raw
                        row[NORM_LOCATION_COL] = row.get(NORM_LOCATION_COL, norm(row.get(LOCATION_COL, "")))

                    c = norm(row.get(case_col, ""))
                    if c:
                        existing_case_keys.add(c)
                    existing_fps.add(fingerprint_row(row, fp_cols))

                    writer.writerow(row)

            os.replace(tmp_path, agg_path)
            return existing_case_keys, existing_fps, out_fields

        # No rewrite: just compute sets (still normalize for fingerprint consistency if possible)
        for row in reader:
            row = dict(row)
            if normalize_event_location is not None:
                row = apply_location_normalization(row, normalize_event_location)

            c = norm(row.get(case_col, ""))
            if c:
                existing_case_keys.add(c)
            existing_fps.add(fingerprint_row(row, fp_cols))

        return existing_case_keys, existing_fps, fieldnames


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Append unique OCR rows to aggregated_cases.csv, deduping by Case/Incident, "
            "and normalizing Location using helper/normalize_location.py."
        )
    )
    ap.add_argument("input_csv", help="Path to OCR output CSV (e.g., .../latest/latest.csv)")
    ap.add_argument(
        "--agg",
        default="",
        help="Path to aggregated output CSV (default: ./output/post/aggregated_cases.csv next to this script)",
    )
    ap.add_argument(
        "--case-col",
        default=DEFAULT_CASE_COL,
        help=f"Case identifier column name (default: {DEFAULT_CASE_COL})",
    )
    ap.add_argument(
        "--fp-cols",
        default=",".join(DEFAULT_FP_COLS),
        help="Comma-separated list of columns used to fingerprint a record for dedupe",
    )
    ap.add_argument(
        "--backend-root",
        default="",
        help="Path to uc_merced_campus_event_api_backend (auto-detected if omitted)",
    )
    ap.add_argument(
        "--no-rewrite-agg",
        action="store_true",
        help="Do not rewrite/upgrade the existing aggregate file; only normalize new appended rows",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_csv = Path(args.input_csv).expanduser().resolve()

    agg_path = (
        Path(args.agg).expanduser().resolve()
        if args.agg
        else (script_dir / "output" / "post" / "aggregated_cases.csv").resolve()
    )

    case_col = args.case_col
    fp_cols = [c.strip() for c in args.fp_cols.split(",") if c.strip()]

    if not input_csv.exists():
        print(f"[post] Input CSV not found: {input_csv}", file=sys.stderr)
        return 2

    # Load normalizer
    try:
        normalize_event_location, backend_path = load_normalizer(args.backend_root or None)
    except Exception as e:
        print(f"[post] ERROR: could not import normalize_event_location: {e}", file=sys.stderr)
        return 2

    # Normalize/upgrade existing aggregate (and build dedupe sets)
    existing_case, existing_fp, agg_fieldnames = normalize_existing_aggregate_in_place(
        agg_path=agg_path,
        case_col=case_col,
        fp_cols=fp_cols,
        normalize_event_location=normalize_event_location,
        force_rewrite=(not args.no_rewrite_agg),
    )

    # Read input rows and append unique ones
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            print("[post] Input CSV has no headers; skipping.", file=sys.stderr)
            return 3

        in_fields = list(reader.fieldnames)

        # Validate required columns exist in input
        missing_fp = [c for c in fp_cols if c not in in_fields]
        if missing_fp:
            print(f"[post] Input CSV missing fingerprint columns {missing_fp}. Columns: {in_fields}", file=sys.stderr)
            return 3
        if case_col not in in_fields:
            print(f"[post] Input CSV missing '{case_col}'. Columns: {in_fields}", file=sys.stderr)
            return 3
        if LOCATION_COL not in in_fields:
            print(f"[post] Input CSV missing '{LOCATION_COL}'. Columns: {in_fields}", file=sys.stderr)
            return 3

        # Determine output header (must match existing aggregate if present)
        if agg_fieldnames is None:
            out_fields = in_fields[:]
            for c in (RAW_LOCATION_COL, NORM_LOCATION_COL):
                if c not in out_fields:
                    out_fields.append(c)
            new_file = True
        else:
            out_fields = agg_fieldnames
            new_file = (not agg_path.exists()) or (agg_path.stat().st_size == 0)

        ensure_parent_dir(agg_path)

        appended = 0
        seen_fp_this_run: set[str] = set()
        seen_case_this_run: set[str] = set()

        with agg_path.open("a", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()

            for row in reader:
                row = dict(row)

                c = norm(row.get(case_col, ""))
                if not c:
                    continue  # skip rows with no Case/Incident (preserving your existing behavior)

                # Normalize location and set raw_location/location/Location
                row = apply_location_normalization(row, normalize_event_location)

                fp = fingerprint_row(row, fp_cols)

                # Duplicate if fingerprint already exists OR case already exists
                if fp in existing_fp or fp in seen_fp_this_run:
                    continue
                if c in existing_case or c in seen_case_this_run:
                    continue

                writer.writerow(row)
                appended += 1

                existing_fp.add(fp)
                seen_fp_this_run.add(fp)

                existing_case.add(c)
                seen_case_this_run.add(c)

    now = datetime.utcnow().isoformat() + "Z"
    print(f"[post] {now} appended={appended} agg={agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
