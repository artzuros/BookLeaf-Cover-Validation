# BookLeaf Cover Validation

> **BookLeaf — Technical Round 2 Assignment** · Automated award-badge overlap detection for book covers.

Automated visual inspection system that detects text overlap with the **"Winner of the 21st Century Emily Dickinson Award"** badge zone on BookLeaf book covers.

Upload a cover image (PNG or PDF) and receive a verdict — **PASS**, **WARNING**, or **VIOLATION** — along with annotated visuals, correction instructions, and revision tracking.

---
### Loom Video : https://www.loom.com/share/778c935633ae4817b25ee5eba66019ed
Also please check out the `helper.html` for a visual understanding of the algorithm.
---

## Architecture

```
Upload (PNG / PDF)
     │
     ▼
FastAPI server (:8010)
     ├─ 1. Spread splitting — detect gutter edge via Sobel energy
     ├─ 2. Zone computation — badge zone (bottom 9mm) + safe area (3mm margins)
     ├─ 3. OCR (Tesseract) — 5× upscaled Otsu binarization
     ├─ 4. MSER geometry detection — secondary corroboration
     ├─ 5. Three-tier verdict logic
     ├─ 6. Annotated PNG output (badge zone, OCR text, intrusion markers)
     └─ 7. SQLite revision tracking — per-ISBN history
```

Two modes:
- **FastAPI server** — `POST /verify` endpoint for integration
- **Standalone scripts** — `cover_verify.py` / `cover_visualize.py` for batch processing

---

## Features

- **Spread splitting** — Automatically detects the gutter fold on two-page wraparound cover spreads using Sobel edge energy and splits into back/front covers.
- **Award badge OCR** — 5× Lanczos upscaling + dual-polarity Otsu binarization + Tesseract OCR on the 9 mm badge zone (plus an upward buffer) to catch overlapping text.
- **Three-tier verdict:**
  - `PASS` — Badge is clean, no overlap concerns.
  - `WARNING` — Content near the badge zone but no confirmed overlap.
  - `VIOLATION` — Readable non-badge text confirmed in the badge zone.
- **Geometric corroboration** — MSER-based text-region detection in the badge zone, used as a secondary signal.
- **Resolution guard** — Flags covers under 1000 px front-cover width and degrades confidence.
- **Annotated visual output** — Colour-coded badge zone overlays, safe-area boundaries, OCR sidebar, and intrusion callouts.
- **Correction instructions** — Auto-generated step-by-step guidance per verdict.
- **Revision tracking** — SQLite database tracks each submission per ISBN with versioning.
- **n8n workflow** — Pre-configured automation workflow (`BookLeaf Cover Validation.json`) for no-code integration.
- **Batch-mode ground truth** — Built-in test set with 8 labelled covers for regression testing.

---

## Quick Start

### Requirements

- Python 3.10+
- Tesseract OCR engine (system-level `tesseract` binary)
- Conda environment named `bookleaf` (recommended)

Install system dependencies:
```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### Install (with Conda — recommended)

```bash
conda activate bookleaf
cd "Cover Validation"
pip install -r requirements.txt
```

This project was developed inside the `bookleaf` conda environment, which includes all required dependencies (OpenCV, PyTorch, Sentence Transformers, Supabase client, etc.) for the broader BookLeaf automation suite.

### Install (with venv)

```bash
git clone https://github.com/artzuros/BookLeaf-Cover-Validation/ "Cover Validation"
cd "Cover Validation"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8010 --reload
```

### Run Standalone Batch Verification

```bash
# Process all images in data/ and print a checklist report
python cover_verify.py

# Same, but also produce annotated visuals in output/
python cover_visualize.py
```

---

## API Reference

### `POST /verify`

Upload a cover file for badge overlap detection.

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Cover image (`.png`) or PDF (`.pdf`) |
| `isbn` | String | Yes | 13-digit ISBN for revision tracking |
| `author_name` | String | No | Required only for new (unregistered) ISBNs |
| `book_name` | String | No | Required only for new (unregistered) ISBNs |

**Example:**

```bash
curl -X POST http://localhost:8010/verify \
  -F "file=@cover.png" \
  -F "isbn=9781234567890" \
  -F "author_name=Sara Johnson" \
  -F "book_name=Whispers of the Soul"
```

**Response** (JSON):

```json
{
  "airtable_book_id": "9781234567890",
  "metadata": {
    "filename": "cover.png",
    "format": "PNG",
    "pages": 1,
    "dimensions": { "width": 1250, "height": 2000 }
  },
  "detection_timestamp": "2026-06-25T10:30:00+00:00",
  "issue_type": "text_overlap",
  "severity": "critical",
  "status": "Review Needed",
  "confidence_score": 85,
  "visual_annotations_url": "output/verify_9781234567890_VIOLATION.png",
  "correction_instructions": {
    "clear_status": "FAILED",
    "specific_issues": [
      {
        "marker": "❌ Text overlap detected",
        "details": "Readable text found in badge zone: \"A line of text...\"",
        "location": "Badge zone (bottom 9mm of front cover)"
      }
    ],
    "step_by_step": [
      "Increase vertical clearance between cover text and the award badge zone (bottom 9mm).",
      "Ensure no text lines extend into the 9mm badge-reserved area.",
      "Keep all non-badge content at least 3mm above the badge zone boundary.",
      "Re-export the cover file and re-submit for verification."
    ]
  },
  "revision_tracking": {
    "submission_id": "a1b2c3d4e5f6",
    "revision_id": "abc123def456",
    "version": 1,
    "author_name": "Sara Johnson",
    "book_name": "Whispers of the Soul",
    "isbn": "9781234567890"
  }
}
```

### `GET /`

Root endpoint listing available endpoints.

### `GET /health`

Health check — returns `{"status": "ok", "service": "cover-verification"}`.

### `GET /books`

List all books registered in the revision-tracking database.

### `GET /revisions`

List all revision records across all books.

---

## Visual Output

When a cover fails or warrants attention, the server generates an annotated PNG with:

| Annotation | Colour | Meaning |
|-----------|--------|---------|
| Badge zone fill | Green / Amber / Red | Semi-transparent overlay tinted by verdict |
| Badge zone outline | Thick border | Colour-coded by severity |
| Safe area boundary | Cyan | 3 mm margin boundary from each edge |
| Upward buffer | Amber dashed | Extended OCR crop region above badge |
| Intruding text | Red bullets | Clean readable lines that don't match badge text |
| OCR sidebar | Bottom strip | Raw Tesseract output with highlighting |
| Verdict banner | Top strip | Verdict, confidence, match ratio |

Example: `output/verify_9786473824165_VIOLATION.png`

---

## Detection Methodology

### 1. Spread Splitting

Many cover submissions arrive as full wraparound spreads (back + front + spine). The system detects the gutter fold by computing Sobel edge energy in a window around the horizontal midpoint. If a strong vertical edge exists (the fold), the image is split there; otherwise a simple 50/50 split is used.

### 2. Zone Computation

Based on BookLeaf's standard trim size (5" × 8"):

| Zone | Specification | Computed From |
|------|--------------|---------------|
| **Safe area** | 3 mm inward from all edges | `px_per_mm = DPI / 25.4` |
| **Badge zone** | Bottom 9 mm of front cover | `badge_rect.y = height - 9mm_in_px` |

DPI is estimated as the average of `width / 5"` and `height / 8"`.

### 3. OCR Pipeline

1. Extract a crop from just above the badge zone (including an upward buffer of ~7.7% of cover height) down to the bottom edge.
2. Upscale 5× with Lanczos interpolation.
3. Binarize with Otsu's threshold (both normal and inverted).
4. Run Tesseract with `--oem 1 --psm 6` on both binarized versions; keep the result with more alphabetic characters.
5. Fuzzy-match the full OCR output against `"Winner of the 21st Century Emily Dickinson Award"`.
6. Classify non-matching lines as clean readable text or OCR garbage using alphabetic density, word count, and word-length heuristics.

### 4. Geometric Detection (Secondary)

MSER (Maximally Stable Extremal Regions) detects text-like blobs in the strict badge zone. Glyphs are filtered by aspect ratio, merged into lines, and the count is reported as a corroborating signal. This is never the sole basis for a VIOLATION verdict.

### 5. Verdict Logic

| Condition | Verdict | Meaning |
|-----------|---------|---------|
| Clean readable non-badge text found | **VIOLATION** | Text definitely overlaps the badge |
| OCR garbage only + match ratio < 0.50 | **WARNING** | Likely edge proximity or layout congestion |
| Badge fully legible (match ≥ 0.85) | **PASS** | No concerns |
| Mid-range match with noise only | **PASS** | Likely OCR artifact |

---

## Revision Database

The SQLite database at `data/cover_revisions.db` tracks:

- **books** — Author name, book name, ISBN, created timestamp
- **revisions** — Per-submission records with verdict, original filename, version number

Pre-seeded with 12 author/book pairs (e.g., Sara Johnson / Whispers of the Soul). New ISBNs are auto-registered on first submission if `author_name` and `book_name` are provided.

---

## n8n Workflow

Import `BookLeaf Cover Validation.json` into an n8n instance (Workflows → Import from File). The workflow:

1. Receives a webhook trigger with cover file metadata
2. Calls the `/verify` endpoint
3. Routes responses based on verdict and confidence
4. Can be chained with the main BookLeaf query-bot workflow

---

## Standalone Scripts

### `cover_verify.py`

Batch-mode verification against the `data/` directory. Processes all PNG files found there, compares results against a hardcoded ground-truth table, and prints a detailed checklist-style report showing badge match ratios, leftover OCR lines, and match/mismatch status.

```bash
python cover_verify.py
```

### `cover_visualize.py`

Same as `cover_verify.py` but also generates annotated visualizations in the `output/` directory for WARNING and VIOLATION cases.

```bash
python cover_visualize.py
```

---

## Ground Truth / Test Set

The `data/` directory contains 8 labelled cover images used for regression testing:

| # | Title | Expected |
|---|-------|----------|
| 28 | Shabd (uncorrected) | VIOLATION |
| 29 | Shabd (corrected) | PASS |
| 31 | Offline Sorrows / Online Ghosts | PASS |
| 32 | Echoes Along the Way (uncorrected) | VIOLATION |
| 33 | Echoes Along the Way (corrected) | PASS |
| 34 | Inner Mirror (uncorrected) | WARNING |
| 35 | Inner Mirror (corrected) | PASS |
| 36 | Tainted By Emotion | VIOLATION |

Run `python cover_verify.py` to check all images and compare against expected verdicts.

---

## Project Structure

```
Cover Validation/
├── src/
│   ├── main.py              # FastAPI server entry point
│   ├── config.py            # Paths, constants, colour map
│   ├── models.py            # Pydantic response models + factory
│   ├── detector.py          # Core detection: OCR, MSER, zone logic
│   ├── visualizer.py        # Annotated PNG generation
│   ├── pdf_converter.py     # PDF → numpy array via PyMuPDF
│   ├── filename_parser.py   # ISBN extraction from filenames
│   ├── database.py          # SQLite revision tracking
│   └── routers/
│       └── verify.py        # POST /verify endpoint
├── cover_verify.py          # Standalone batch verification script
├── cover_visualize.py       # Standalone batch script + visual output
├── data/                    # Test images + SQLite DB
├── correct/                 # Sorted PASS images (post-verification)
├── wrong/                   # Sorted WARNING/VIOLATION images
├── output/                  # Generated annotated visuals
├── uploads/                 # Temp upload directory
├── BookLeaf Cover Validation.json  # n8n workflow export
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-python-headless` | Image I/O, MSER, drawing, resize |
| `pytesseract` | Tesseract OCR Python wrapper |
| `numpy` | Array operations |
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Response model validation |
| `PyMuPDF` | PDF page rendering |
| `python-dotenv` | Environment variable loading |

- **External:** Tesseract OCR engine (v5+ recommended)
- **Requires a system-level `tesseract` binary** — install via `apt install tesseract-ocr` or `brew install tesseract` before running.
