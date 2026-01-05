#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Table OCR (multi-file & structured outputs)

What's new in this version
--------------------------
- Optional **local high-accuracy OCR backend** via PaddleOCR (CPU/GPU) in addition to Tesseract.
  Use: `--ocr paddle` (falls back to Tesseract if PaddleOCR is unavailable).
- **Higher resolution rendering** for PDFs (default --dpi=600).
- **Per-cell exports**:
  - Saves **bounding-box coordinates** and recognized text to **per-page JSON** files.
  - Optionally saves **PNG crops of every cell** to `cells/page_XXX/` (enable with `--save-cells`).
- Table grid sources preserved (OpenCV vs model proposal), plus line arrays (x_lines/y_lines) in JSON.
- Cell upscaling (`--cell-scale`) before OCR for tiny text.

Directory layout (per input PDF)
--------------------------------
output/<file-stem>/
  ├─ <file-stem>.csv                 # consolidated CSV over all pages (schema below)
  ├─ <file-stem>_001.png             # per-page debug overlay
  ├─ <file-stem>_002.png
  ├─ page_001.json                   # per-page JSON with bboxes+text
  ├─ page_002.json
  └─ cells/
       ├─ page_001/
       │    ├─ r000_c000.png
       │    ├─ r000_c001.png
       │    └─ ...
       └─ page_002/
            └─ ...

CSV schema (unchanged)
----------------------
Nature, Case/Incident, Reported, Occured, Location, Disposition, On Campus?, Area

"""

import os
import sys
import json
import csv
import re
import glob
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Attempt local OCR backends
PADDLE_AVAILABLE = False
PADDLE_LANG_MAP = {
    # Map tesseract-like tags to PaddleOCR language packs (extend as needed)
    "eng": "en",
}
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# Ensure third-party libs
try:
    import cv2
    import pytesseract
except ImportError as e:
    print("Missing dependency. Install with:\n"
          "  pip install opencv-python-headless pytesseract pillow numpy pdf2image python-dotenv openai",
          file=sys.stderr)
    raise

# OpenAI (optional fallback for grid proposal)
try:
    from openai import OpenAI
    OPENAI_CLIENT_OK = True
except Exception:
    OPENAI_CLIENT_OK = False

# ---------------- Env / Paths ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-thinking")
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
POPPLER_PATH = os.getenv("POPPLER_PATH", "")
# "ALL", "", or integer string
FORCE_PDF_PAGE_RAW = os.getenv("FORCE_PDF_PAGE", "ALL")


def _parse_forced_page(val: str):
    s = (val or "").strip()
    if not s:
        return None
    if s.upper() == "ALL":
        return "ALL"
    try:
        return int(s)
    except:
        return None


FORCE_PDF_PAGE = _parse_forced_page(FORCE_PDF_PAGE_RAW)

# >>> Hard-coded pixel offset for the TOP of the FIRST ROW <<<
# +N pushes the top boundary DOWN; -N pulls it UP (image pixels at render DPI).
TOP_ROW_OFFSET_PX = 10

if POPPLER_PATH:
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------- CSV Schema ----------------
SCHEMA = [
    "Nature",
    "Case/Incident",
    "Reported",
    "Occured",          # keep your exact spelling
    "Location",
    "Disposition",
    "On Campus?",
    "Area",
]

# header aliases (normalized; see _norm)
HEADER_ALIASES = {
    "Nature": {
        "nature", "type", "incidenttype", "classification", "offense", "crime", "violation"
    },
    "Case/Incident": {
        "caseincident", "case#", "incident#", "case", "incident", "caseno", "casenumber", "incidentnumber"
    },
    "Reported": {
        "reported", "datereported", "reporteddate", "reportedtime", "date/time reported", "reportdate"
    },
    "Occured": {  # map both spellings to your "Occured"
        "occured", "occurred", "dateoccurred", "dateoccured", "occurreddate", "occurredtime", "date/time occurred"
    },
    "Location": {
        "location", "loc", "address", "place", "building", "room", "locn"
    },
    "Disposition": {
        "disposition", "status", "outcome", "result", "case status"
    },
    "On Campus?": {
        "oncampus", "oncampus?", "on-campus", "oncampusyn", "campus", "on/off campus", "onoffcampus"
    },
    "Area": {
        "area", "zone", "sector", "district", "precinct", "beat"
    },
}


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("—", "-").replace("–", "-").replace("_", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _contains_or_equal(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or a in b or b in a


def guess_header_mapping(header_row):
    normed = [_norm(x) for x in header_row]
    mapping = {}
    used = set()
    # exact/alias
    for schema_col, aliases in HEADER_ALIASES.items():
        for idx, token in enumerate(normed):
            if idx in used:
                continue
            if token in aliases:
                mapping[schema_col] = idx
                used.add(idx)
                break
    # loose containment
    for schema_col, aliases in HEADER_ALIASES.items():
        if schema_col in mapping:
            continue
        for idx, token in enumerate(normed):
            if idx in used or not token:
                continue
            if any(_contains_or_equal(token, a) for a in aliases):
                mapping[schema_col] = idx
                used.add(idx)
                break
    return mapping


def looks_like_header(row):
    mapping = guess_header_mapping(row)
    return len(mapping) >= 4


def clean_cell(text: str) -> str:
    if text is None:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = t.replace(" | ", " ").replace("|", " ")
    return t


def normalize_on_campus(val: str) -> str:
    s = clean_cell(val).lower()
    if not s:
        return ""
    if "off campus" in s or "off-campus" in s or s in {"off", "offcampus"}:
        return "No"
    if "on campus" in s or "on-campus" in s or s in {"on", "oncampus"}:
        return "Yes"
    yes = {"y", "yes", "true", "t", "1"}
    no = {"n", "no", "false", "f", "0"}
    if s in yes:
        return "Yes"
    if s in no:
        return "No"
    return clean_cell(val)


def conform_to_schema(table_rows, detect_header=True):
    """Align raw OCR table rows to SCHEMA.
    Robustness improvement: when a header is detected but some SCHEMA columns
    aren't matched by aliases (common for "On Campus?"/"Area" variants),
    we **backfill by position** using remaining header indices. We bias the
    rightmost remaining indices to the last SCHEMA fields ("On Campus?", "Area").
    """
    rows_out = []
    used_header = False
    mapping = {col: None for col in SCHEMA}

    if not table_rows:
        return rows_out, {"used_header": False, "mapping": mapping, "dropped_columns": []}

    hdr = table_rows[0]
    hdr_len = len(hdr) if hdr else 0

    if detect_header and looks_like_header(hdr):
        used_header = True
        # 1) Alias/semantic mapping
        mapping.update(guess_header_mapping(hdr))

        # 2) Positional backfill for any unmapped SCHEMA columns
        used_idx = {i for i in mapping.values() if i is not None}
        remaining_idx = [i for i in range(hdr_len) if i not in used_idx]
        remaining_idx.sort()  # left→right

        need = [col for col in SCHEMA if mapping[col] is None]

        # Heuristic: map the last two SCHEMA fields to the rightmost available indices first
        # to catch cases where aliases failed but the layout is consistent.
        for special in ["Area", "On Campus?"]:
            if special in need and remaining_idx:
                mapping[special] = remaining_idx.pop()  # use rightmost
                need.remove(special)

        # Fill any other missing columns from left to right
        for col, idx in zip(need, remaining_idx):
            mapping[col] = idx

        data_rows = table_rows[1:]
    else:
        # No reliable header: align by position
        data_rows = table_rows
        for i, col in enumerate(SCHEMA):
            if i < hdr_len:
                mapping[col] = i

    # Build normalized rows
    for row in data_rows:
        aligned = []
        for col in SCHEMA:
            idx = mapping.get(col)
            val = row[idx] if (idx is not None and idx < len(row)) else ""
            val = clean_cell(val)
            if col == "On Campus?":
                val = normalize_on_campus(val)
            aligned.append(val)
        rows_out.append(aligned)

    # Track dropped columns (if source has more columns than SCHEMA)
    dropped = []
    if data_rows:
        max_len = max(len(r) for r in data_rows)
        if max_len > len(SCHEMA):
            dropped = list(range(len(SCHEMA), max_len))

    return rows_out, {"used_header": used_header, "mapping": mapping, "dropped_columns": dropped}


# ---------------- CLI ----------------
ap = argparse.ArgumentParser(
    description="Unified table OCR with OpenCV + (optional) PaddleOCR + GPT reasoning fallback (folder-aware)")
ap.add_argument("input", help="PDF/image path OR a directory containing PDFs")
ap.add_argument("--page", type=int, default=1,
                help="Page number for PDFs (1-based)")
ap.add_argument("--dpi", type=int, default=600,
                help="Render DPI for PDFs (higher = sharper; 600 is a good max)")
ap.add_argument("--lang", default="eng",
                help="Language(s), e.g., 'eng' or 'eng+spa' (Tesseract). For Paddle: 'eng' -> 'en'.")
ap.add_argument("--ocr", choices=["auto", "tesseract",
                "paddle"], default="auto", help="OCR backend")
ap.add_argument("--cell-scale", type=float, default=2.0,
                help="Upscale factor for each cell crop before OCR (e.g., 1.0 ~ 3.0)")
ap.add_argument("--save-cells", action="store_true",
                help="Save PNG crop for every cell bbox under cells/page_xxx/")
ap.add_argument("--save-json", action="store_true",
                help="Save per-page JSON with bboxes + text")
ap.add_argument("--out-dir", default=os.path.join("output", "ocr_output"),
                help="Root output directory")

ap.add_argument("--debug", action="store_true", help="Verbose logging")
ap.add_argument("--print-json", action="store_true",
                help="Print final JSON summary to stdout")
ap.add_argument("--always-reason", action="store_true",
                help="Use model proposal even if OpenCV finds a grid")
ap.add_argument("--no-header-detect", action="store_true",
                help="Treat first row as data (do not auto-detect header)")
args = ap.parse_args()


def log(*m):
    if args.debug:
        print("[DEBUG]", *m)

# ---------------- Image & OCR helpers ----------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_single_pdf_page(path, page=1, dpi=300):
    try:
        from pdf2image import convert_from_path
    except Exception:
        print("pdf2image is required for PDF input. Install with:\n  pip install pdf2image", file=sys.stderr)
        raise
    pages = convert_from_path(path, dpi=dpi, first_page=page, last_page=page)
    if not pages:
        raise RuntimeError(f"No page {page} in PDF")
    pil = pages[0].convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr


def load_all_pdf_pages(path, dpi=300):
    try:
        from pdf2image import convert_from_path
    except Exception:
        print("pdf2image is required for PDF input. Install with:\n  pip install pdf2image", file=sys.stderr)
        raise
    pil_pages = convert_from_path(path, dpi=dpi)
    if not pil_pages:
        raise RuntimeError("No pages in PDF")
    out = []
    for pil in pil_pages:
        bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        out.append(bgr)
    return out


def to_gray_bin(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return gray, bin_img


def deskew(img_bgr):
    # (Disabled in this build; keep placeholder for future re-enable.)
    return img_bgr, 0.0


def peak_positions(arr, min_gap, frac=0.4):
    if arr.size == 0:
        return []
    thr = frac * float(np.max(arr))
    pos = np.where(arr > thr)[0]
    if pos.size == 0:
        return []
    groups = []
    start = int(pos[0])
    prev = int(pos[0])
    for v in pos[1:]:
        v = int(v)
        if v - prev > min_gap:
            groups.append((start, prev))
            start = v
        prev = v
    groups.append((start, prev))
    centers = [int((a+b)//2) for a, b in groups]
    return centers


def find_grid(bin_img):
    h, w = bin_img.shape
    kx = max(10, w // 100)
    ky = max(10, h // 100)
    horiz = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (kx, 1)), iterations=1)
    vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, ky)), iterations=1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(
        cv2.MORPH_RECT, (kx, 1)), iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, ky)), iterations=1)
    x_proj = np.sum(vert > 0, axis=0)
    y_proj = np.sum(horiz > 0, axis=1)
    x_lines = peak_positions(x_proj, max(10, w//200))
    y_lines = peak_positions(y_proj, max(10, h//200))
    if len(x_lines) < 2 or len(y_lines) < 2:
        return None, None, None
    boxes = []
    for r in range(len(y_lines)-1):
        y1, y2 = y_lines[r], y_lines[r+1]
        for c in range(len(x_lines)-1):
            x1, x2 = x_lines[c], x_lines[c+1]
            padx = max(1, (x2-x1)//50)
            pady = max(1, (y2-y1)//50)
            boxes.append((r, c, x1+padx, y1+pady, x2-padx, y2-pady))
    return x_lines, y_lines, boxes


# -------- OCR backends --------
_paddle_engine = None


def _get_paddle_engine(lang: str):
    global _paddle_engine
    if _paddle_engine is not None:
        return _paddle_engine
    if not PADDLE_AVAILABLE:
        return None
    # map language token
    plang = PADDLE_LANG_MAP.get(lang.split("+")[0], "en")
    _paddle_engine = PaddleOCR(
        use_angle_cls=False,
        lang=plang,
        det=False,  # we supply boxes; do recognition only
        rec=True,
        show_log=False,
    )
    return _paddle_engine


def _tesseract_text(crop_bgr, lang="eng"):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 3)
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    config = "--psm 6 --oem 1"
    try:
        txt = pytesseract.image_to_string(bw, lang=lang, config=config)
    except Exception:
        txt = pytesseract.image_to_string(bw, config=config)
    return (txt or "").strip()


def _paddle_text(crop_bgr, lang="eng"):
    try:
        engine = _get_paddle_engine(lang)
        if engine is None:
            return None
        # PaddleOCR accepts numpy images (BGR) directly
        res = engine.ocr(crop_bgr, det=False, rec=True, cls=False)
        # res formats vary a bit; try to extract first recognized string
        # Common shape for det=False: [[(text, conf), (text, conf), ...]]
        if not res:
            return ""
        first = res[0]
        if isinstance(first, (list, tuple)) and first:
            # pick the longest text among candidates for robustness
            texts = []
            for item in first:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    texts.append(str(item[0]))
            if texts:
                texts.sort(key=lambda s: len(s or ""), reverse=True)
                return texts[0].strip()
        return ""
    except Exception as e:
        return None


def ocr_cell_img(crop_bgr, backend="auto", lang="eng", scale=1.0):
    # Optional upscale to help OCR on tiny text
    if scale and scale != 1.0:
        h, w = crop_bgr.shape[:2]
        nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
        crop_bgr = cv2.resize(crop_bgr, (nw, nh),
                              interpolation=cv2.INTER_CUBIC)

    # Backend selection
    if backend == "paddle" or (backend == "auto" and PADDLE_AVAILABLE):
        txt = _paddle_text(crop_bgr, lang=lang)
        if txt is not None:
            return txt
        # fall through to tesseract if paddle failed
    return _tesseract_text(crop_bgr, lang=lang)


def save_debug(img_bgr, boxes, out_path):
    dbg = img_bgr.copy()
    for (_, _, x1, y1, x2, y2) in boxes:
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.imwrite(out_path, dbg)


def b64_jpeg_from_bgr(img_bgr, quality=90):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    bio = BytesIO()
    pil.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


def propose_grid_with_openai(img_bgr, model=OPENAI_MODEL):
    if not (OPENAI_CLIENT_OK and OPENAI_API_KEY):
        raise RuntimeError(
            "OPENAI_API_KEY missing or OpenAI client not installed")
    client = OpenAI(api_key=OPENAI_API_KEY)
    import base64
    b64str = base64.b64encode(b64_jpeg_from_bgr(img_bgr)).decode("ascii")
    system = (
        "You are an expert at locating tables on a page.\n"
        "Return STRICT JSON ONLY, no prose, matching exactly:\n"
        "{\n"
        '  "table": {\n'
        '    "bbox": [x1, y1, x2, y2],\n'
        '    "rows": <int>,\n'
        '    "cols": <int>,\n'
        '    "x_lines": [<number>...],\n'
        '    "y_lines": [<number>...]\n'
        "  }\n"
        "}\n"
        'If no table is present, return {"table": null}.\n'
        "Coordinates are integer pixel positions in the provided image."
    )
    user_parts = [
        {"type": "input_text", "text": "Find the dominant table and propose a reasonable grid. Prefer ruling lines; otherwise infer columns and rows."},
        {"type": "input_image", "image_data": b64str, "mime_type": "image/jpeg"}
    ]
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "high"},
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user_parts}],
        )
        text = getattr(resp, "output_text", None)
        if not text:
            text = ""
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
        if not text:
            raise RuntimeError("Model returned no text")
        return json.loads(text)
    except Exception as e:
        log("Responses API failed or unavailable:", str(e))
    # Fallback: chat.completions (vision)
    try:
        import base64
        data_url = f"data:image/jpeg;base64,{base64.b64encode(b64_jpeg_from_bgr(img_bgr)).decode('ascii')}"
        client = OpenAI(api_key=OPENAI_API_KEY)
        chat = client.chat.completions.create(
            model=OPENAI_FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": system + "\nRespond with JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Find the dominant table and propose a grid."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            temperature=0.2,
        )
        text = chat.choices[0].message.content
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"OpenAI fallback also failed: {e}")


def lines_and_boxes_from_proposal(proposal):
    if not proposal or "table" not in proposal or proposal["table"] is None:
        return None, None, None
    t = proposal["table"]
    bbox = t.get("bbox")
    rows = int(t.get("rows", 0))
    cols = int(t.get("cols", 0))
    if not bbox or rows < 1 or cols < 1:
        return None, None, None
    x_lines = t.get("x_lines")
    y_lines = t.get("y_lines")
    if not x_lines or not y_lines:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x_lines = [int(round(x1 + i*(x2-x1)/cols)) for i in range(cols+1)]
        y_lines = [int(round(y1 + i*(y2-y1)/rows)) for i in range(rows+1)]
    else:
        x_lines = [int(round(v)) for v in x_lines]
        y_lines = [int(round(v)) for v in y_lines]
    boxes = []
    for r in range(len(y_lines)-1):
        for c in range(len(x_lines)-1):
            x1, x2 = x_lines[c], x_lines[c+1]
            y1, y2 = y_lines[r], y_lines[r+1]
            boxes.append((r, c, x1, y1, x2, y2))
    return x_lines, y_lines, boxes


# -------- Offset helper --------

def apply_top_row_offset(boxes, y_lines, dy, img_h):
    """
    Shift the TOP boundary of row 0 by 'dy' pixels.
    boxes: list[(r, c, x1, y1, x2, y2)]
    y_lines: optional list[int] of horizontal lines; y_lines[0] will be moved too.
    img_h: image height (for clamping).
    """
    if not boxes or dy == 0:
        return boxes, y_lines

    out = []
    for (r, c, x1, y1, x2, y2) in boxes:
        if r == 0:
            ny1 = y1 + dy
            ny1 = max(0, min(ny1, y2 - 1))
            out.append((r, c, x1, ny1, x2, y2))
        else:
            out.append((r, c, x1, y1, x2, y2))

    if y_lines and len(y_lines) >= 2:
        y0 = y_lines[0] + dy
        y0 = max(0, min(y0, y_lines[1] - 1))
        y_lines = [y0] + y_lines[1:]

    return out, y_lines


# -------- Per-page processing --------

def process_single_image(img, *, page_index: int, stem: str, file_dir: str,
                         lang: str, always_reason: bool, no_header_detect: bool,
                         debug_out_path: str, ocr_backend: str, cell_scale: float,
                         save_cells: bool, save_json: bool):
    img_dk, angle = deskew(img)
    _, bin_img = to_gray_bin(img_dk)
    x_lines, y_lines, boxes = find_grid(bin_img)

    used_grid = None
    proposal = None
    if boxes and not always_reason:
        used_grid = "opencv"
    else:
        try:
            proposal = propose_grid_with_openai(img_dk)
        except Exception as e:
            log("Model proposal failed:", str(e))
            proposal = {"table": None}
        x_lines2, y_lines2, boxes2 = lines_and_boxes_from_proposal(proposal)
        if boxes2:
            used_grid = "model"
            x_lines, y_lines, boxes = x_lines2, y_lines2, boxes2
        else:
            # no grid in this page
            if debug_out_path:
                cv2.imwrite(debug_out_path, img_dk)
            return {"status": "no-grid", "angle": angle, "rows": [], "grid_source": None,
                    "debug_image": debug_out_path, "page": page_index}

    # Apply top-row offset
    if TOP_ROW_OFFSET_PX != 0:
        h = img_dk.shape[0]
        boxes, y_lines = apply_top_row_offset(
            boxes, y_lines, TOP_ROW_OFFSET_PX, h)

    # OCR cells and optional saving
    boxes.sort(key=lambda b: (b[0], b[1]))
    rows_n = max(b[0] for b in boxes) + 1
    cols_n = max(b[1] for b in boxes) + 1
    table = [[""] * cols_n for _ in range(rows_n)]

    cells_dir = os.path.join(file_dir, "cells", f"page_{page_index:03d}")
    if save_cells:
        ensure_dir(cells_dir)

    cells_meta = []

    for (r, c, x1, y1, x2, y2) in boxes:
        # crop once, reuse for OCR and saving
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
        cx1, cy1 = max(0, cx1), max(0, cy1)
        crop = img_dk[cy1:cy2, cx1:cx2]
        txt = ocr_cell_img(crop, backend=ocr_backend,
                           lang=lang, scale=cell_scale)
        table[r][c] = txt

        cell_rel = None
        if save_cells and crop.size != 0:
            fname = f"r{r:03d}_c{c:03d}.png"
            out_path = os.path.join(cells_dir, fname)
            cv2.imwrite(out_path, crop)
            cell_rel = os.path.relpath(out_path, file_dir)

        cells_meta.append({
            "r": r, "c": c,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "text": txt,
            "image": cell_rel,
        })

    data_rows, info = conform_to_schema(
        table, detect_header=not no_header_detect)

    if debug_out_path:
        save_debug(img_dk, boxes, debug_out_path)

    page_json = {
        "page": page_index,
        "status": "ok",
        "grid_source": used_grid,
        "angle": angle,
        "x_lines": x_lines,
        "y_lines": y_lines,
        "cells": cells_meta,
        "mapping": info["mapping"],
        "used_header": info["used_header"],
        "dropped_columns": info["dropped_columns"],
        "debug_image": os.path.relpath(debug_out_path, file_dir) if debug_out_path else None,
    }

    # Write per-page JSON if requested
    if save_json:
        json_path = os.path.join(file_dir, f"page_{page_index:03d}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(page_json, jf, ensure_ascii=False, indent=2)

    return {"status": "ok", "rows": data_rows, "grid_source": used_grid, "angle": angle,
            "debug_image": debug_out_path, "page": page_index, "page_json": page_json}


# -------- One-PDF processing (handles ALL pages) --------

def process_pdf_file(pdf_path, out_root, lang, dpi, always_reason, no_header_detect, ocr_backend, cell_scale, save_cells, save_json):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    file_dir = os.path.join(out_root, stem)
    ensure_dir(file_dir)

    all_rows = []
    page_results = []
    debug_images = []

    if FORCE_PDF_PAGE == "ALL":
        images = load_all_pdf_pages(pdf_path, dpi=dpi)
        for i, img in enumerate(images, start=1):
            dbg_path = os.path.join(file_dir, f"{stem}_{i}.png")
            res = process_single_image(
                img, page_index=i, stem=stem, file_dir=file_dir,
                lang=lang, always_reason=always_reason, no_header_detect=no_header_detect,
                debug_out_path=dbg_path, ocr_backend=ocr_backend, cell_scale=cell_scale,
                save_cells=save_cells, save_json=save_json)
            page_results.append(res)
            debug_images.append(dbg_path)
            if res.get("status") == "ok":
                all_rows.extend(res["rows"])
    else:
        page_num = FORCE_PDF_PAGE if isinstance(
            FORCE_PDF_PAGE, int) else args.page
        img = load_single_pdf_page(pdf_path, page=page_num, dpi=dpi)
        dbg_path = os.path.join(file_dir, f"{stem}_{page_num}.png")
        res = process_single_image(
            img, page_index=page_num, stem=stem, file_dir=file_dir,
            lang=lang, always_reason=always_reason, no_header_detect=no_header_detect,
            debug_out_path=dbg_path, ocr_backend=ocr_backend, cell_scale=cell_scale,
            save_cells=save_cells, save_json=save_json)
        page_results.append(res)
        debug_images.append(dbg_path)
        if res.get("status") == "ok":
            all_rows.extend(res["rows"])

    # CSV path: one per PDF
    csv_path = os.path.join(file_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SCHEMA)
        for row in all_rows:
            w.writerow(row)

    return {
        "input": pdf_path,
        "csv": csv_path,
        "debug_images": debug_images,
        "rows_written": len(all_rows),
        "pages_processed": len(page_results),
        "page_results": page_results,
    }


# -------- Image (non-PDF) processing --------

def process_image_file(img_path, out_root, lang, always_reason, no_header_detect, ocr_backend, cell_scale, save_cells, save_json):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    file_dir = os.path.join(out_root, stem)
    ensure_dir(file_dir)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    dbg_path = os.path.join(file_dir, f"{stem}_1.png")
    res = process_single_image(
        img, page_index=1, stem=stem, file_dir=file_dir,
        lang=lang, always_reason=always_reason, no_header_detect=no_header_detect,
        debug_out_path=dbg_path, ocr_backend=ocr_backend, cell_scale=cell_scale,
        save_cells=save_cells, save_json=save_json)
    rows = res["rows"] if res.get("status") == "ok" else []

    # CSV per image
    csv_path = os.path.join(file_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SCHEMA)
        for row in rows:
            w.writerow(row)

    return {
        "input": img_path,
        "csv": csv_path,
        "debug_images": [dbg_path],
        "rows_written": len(rows),
        "pages_processed": 1,
        "page_results": [res],
    }


# ---------------- Main flow ----------------

def main():
    in_path = args.input
    out_root = args.out_dir
    ensure_dir(out_root)

    # decide OCR backend
    ocr_backend = args.ocr
    if ocr_backend == "paddle" and not PADDLE_AVAILABLE:
        print("[WARN] --ocr paddle requested but PaddleOCR is not installed. Falling back to Tesseract.")
        ocr_backend = "tesseract"
    if ocr_backend == "auto" and not PADDLE_AVAILABLE:
        ocr_backend = "tesseract"

    results = []

    if os.path.isdir(in_path):
        # Gather files using both cases
        raw_pdfs = glob.glob(os.path.join(in_path, "*.pdf")) + \
            glob.glob(os.path.join(in_path, "*.PDF"))

        # FIX: Convert to set to remove duplicates, then sort back into a list
        pdfs = sorted(list(set(raw_pdfs)))

        if not pdfs:
            print(f"No PDFs found in directory: {in_path}")
            sys.exit(1)
        for pdf in pdfs:
            log(f"Processing PDF: {pdf}")
            # ... rest of the loop remains the same ...
            info = process_pdf_file(
                pdf_path=pdf,
                out_root=out_root,
                lang=args.lang,
                dpi=args.dpi,
                always_reason=args.always_reason,
                no_header_detect=args.no_header_detect,
                ocr_backend=ocr_backend,
                cell_scale=args.cell_scale,
                save_cells=args.save_cells,
                save_json=args.save_json,
            )
            results.append(info)
            print(
                f"✅ {os.path.basename(pdf)} -> {info['csv']}  ({info['rows_written']} rows)")
    else:
        ext = os.path.splitext(in_path)[1].lower()
        if ext == ".pdf":
            info = process_pdf_file(
                pdf_path=in_path,
                out_root=out_root,
                lang=args.lang,
                dpi=args.dpi,
                always_reason=args.always_reason,
                no_header_detect=args.no_header_detect,
                ocr_backend=ocr_backend,
                cell_scale=args.cell_scale,
                save_cells=args.save_cells,
                save_json=args.save_json,
            )
            results.append(info)
            print(
                f"✅ {os.path.basename(in_path)} -> {info['csv']}  ({info['rows_written']} rows)")
        else:
            info = process_image_file(
                img_path=in_path,
                out_root=out_root,
                lang=args.lang,
                always_reason=args.always_reason,
                no_header_detect=args.no_header_detect,
                ocr_backend=ocr_backend,
                cell_scale=args.cell_scale,
                save_cells=args.save_cells,
                save_json=args.save_json,
            )
            results.append(info)
            print(
                f"✅ {os.path.basename(in_path)} -> {info['csv']}  ({info['rows_written']} rows)")

    if args.print_json:
        print(json.dumps({"status": "ok", "outputs": results}, indent=2))


if __name__ == "__main__":
    main()
