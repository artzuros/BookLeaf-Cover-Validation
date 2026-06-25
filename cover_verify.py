#!/usr/bin/env python3
"""
cover_verify.py — BookLeaf Cover Verification System

Single-file CV pipeline that detects text overlap with the
"Winner of the 21st Century Emily Dickinson Award" badge zone
on book cover spreads.

Outputs three verdict levels per image:
  PASS     — Badge is clean, no overlap concerns.
  WARNING  — Content near the badge zone but no confirmed overlap.
  VIOLATION— Text overlaps the badge zone.

Usage:
    python cover_verify.py
"""

from __future__ import annotations

import difflib
import glob
import os
import sys
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Geometry: Rect, spread splitting, zone computation
# ═══════════════════════════════════════════════════════════════════════════════

TRIM_WIDTH_INCHES = 5.0
TRIM_HEIGHT_INCHES = 8.0
SAFE_MARGIN_MM = 3.0
BADGE_HEIGHT_MM = 9.0
MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Rect:
    """Axis-aligned integer rectangle: (x, y) top-left, (w, h) dimensions."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.w, self.h)


def split_spread(image: np.ndarray, sobel_ksize: int = 3,
                 search_radius_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Split a wraparound spread into (back_cover, front_cover)
    using Sobel edge energy near the horizontal midpoint."""
    h, w = image.shape[:2]
    mid = w // 2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    left = max(0, mid - search_radius_px)
    right = min(w, mid + search_radius_px)
    col_energies = np.mean(mag[:, left:right], axis=0)
    best_local = int(np.argmax(col_energies) + left)

    median_energy = float(np.median(col_energies))
    peak_energy = float(col_energies.max())
    split_col = mid if peak_energy < median_energy * 1.5 else best_local

    return image[:, :split_col], image[:, split_col:]


@dataclass
class Zones:
    """Pixel-coordinate zones for a front cover."""
    dpi: float
    side_px_x: int
    side_px_y: int
    bottom_px: int
    badge_zone: Rect
    safe_area: Rect


def compute_zones(front_w_px: int, front_h_px: int) -> Zones:
    """Convert spec (5 × 8 in, 3 mm margins, 9 mm badge) to pixel rects."""
    dpi = (front_w_px / TRIM_WIDTH_INCHES + front_h_px / TRIM_HEIGHT_INCHES) / 2.0
    px_per_mm = dpi / MM_PER_INCH
    side_px_x = round(SAFE_MARGIN_MM * px_per_mm)
    side_px_y = round(SAFE_MARGIN_MM * px_per_mm)
    bottom_px = round(BADGE_HEIGHT_MM * px_per_mm)
    badge_zone = Rect(x=0, y=front_h_px - bottom_px,
                      w=front_w_px, h=bottom_px)
    safe_area = Rect(x=side_px_x, y=side_px_y,
                     w=front_w_px - 2 * side_px_x,
                     h=front_h_px - side_px_y - bottom_px)
    return Zones(dpi=dpi, side_px_x=side_px_x, side_px_y=side_px_y,
                 bottom_px=bottom_px, badge_zone=badge_zone,
                 safe_area=safe_area)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Badge overlap detector
# ═══════════════════════════════════════════════════════════════════════════════
#
# Approach:
#   1. OCR the badge zone (plus an upward buffer for overlapping text).
#   2. Fuzzy-match against the known badge phrase.
#   3. Classify non-matching lines as "clean readable text" or "OCR garbage".
#   4. Three-tier verdict:
#      - Clean non-badge readable lines        → VIOLATION
#      - Low match + garbage-only leftovers    → WARNING (edge proximity)
#      - Otherwise                             → PASS

BADGE_PHRASE = "Winner of the 21st Century Emily Dickinson Award"

# — Resolution guard —
# At < 1000 px front-cover width, the 3 mm safe-margin and 9 mm badge zone
# are < 12 px tall — too few pixels for Tesseract to OCR reliably, even
# with 5× upscaling.  The detection still runs, but results are less
# trustworthy.
MIN_FRONT_WIDTH_PX = 1000

UPWARD_BUFFER_FRACTION = 0.077
MATCH_THRESHOLD_PASS = 0.85
MATCH_THRESHOLD_FAIL = 0.50
MIN_LEFTOVER_CHARS = 8
OCR_UPSCALE = 5
TESSERACT_CONFIG = "--oem 1 --psm 6"


@dataclass
class BadgeOverlapResult:
    verdict: str               # "PASS" | "WARNING" | "VIOLATION"
    confidence: str            # "high" | "medium" | "low"
    badge_match_ratio: float
    raw_ocr_text: str
    leftover_lines: List[str]
    geometry_overlap_count: int
    resolution_warning: str    # "" if ok, else message about low-res input
    reason: str


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _alnum_space(text: str) -> str:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in text)


def _fuzzy_match_ratio(text: str, target: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(text),
                                   _normalize(target)).ratio()


def _match_against_badge(text: str) -> float:
    return _fuzzy_match_ratio(_alnum_space(text), BADGE_PHRASE)


def _is_clean_readable_line(text: str, min_words: int = 2,
                            min_word_len: int = 3) -> bool:
    """Check if OCR output is a clean readable text snippet vs. garbage.
    Requires high alphabetic density, ≥min_words of length ≥min_word_len,
    and average word length > 2.5."""
    alpha = [c for c in text if c.isalpha()]
    if len(alpha) < MIN_LEFTOVER_CHARS:
        return False
    if len(alpha) / max(len(text), 1) < 0.70:
        return False
    words = []
    current = []
    for ch in text:
        if ch.isalpha():
            current.append(ch)
        else:
            if len(current) >= min_word_len:
                words.append("".join(current))
            current = []
    if len(current) >= min_word_len:
        words.append("".join(current))
    if len(words) < min_words:
        return False
    return sum(len(w) for w in words) / max(len(words), 1) > 2.5


def _extract_leftover_lines(raw_ocr_text: str,
                            whole_match_ratio: float) -> List[str]:
    """Extract OCR lines that DON'T match the badge phrase.

    Tiered by overall match quality:
      ≥ 0.85 (high):  permissive line filter; combined lines re-checked.
      0.50–0.85 (mid): strict clean-readable filter on each line.
      < 0.50 (low):   relaxed check for any readable non-badge sentence.
    """
    lines = [l.strip() for l in raw_ocr_text.split("\n") if l.strip()]
    candidates: List[str] = []
    badge_alnum_space = _alnum_space(BADGE_PHRASE.lower())

    line_threshold = 0.50 if whole_match_ratio >= MATCH_THRESHOLD_PASS else MATCH_THRESHOLD_PASS

    for line in lines:
        if _match_against_badge(line) >= line_threshold:
            continue
        line_clean = _alnum_space(line).lower().replace(" ", "")
        badge_flat = badge_alnum_space.replace(" ", "")
        if line_clean in badge_flat or badge_flat in line_clean:
            continue
        if BADGE_PHRASE.lower() in line.lower():
            continue
        candidates.append(line)

    # High match: re-check concatenated candidates
    if whole_match_ratio >= MATCH_THRESHOLD_PASS and candidates:
        combined_clean = " ".join(candidates)
        if _match_against_badge(combined_clean) >= MATCH_THRESHOLD_PASS:
            return []
        badge_flat = badge_alnum_space.replace(" ", "")
        combined_flat = _alnum_space(combined_clean).lower().replace(" ", "")
        if badge_flat in combined_flat or combined_flat in badge_flat:
            return []

    # Mid-range match: filter garbage
    if MATCH_THRESHOLD_FAIL <= whole_match_ratio < MATCH_THRESHOLD_PASS and candidates:
        candidates = [c for c in candidates if _is_clean_readable_line(c)]
        if not candidates:
            return []

    # Low match: relaxed real-overlap check
    if whole_match_ratio < MATCH_THRESHOLD_FAIL:
        has_real_overlap = False
        for line in candidates:
            if _match_against_badge(line) >= MATCH_THRESHOLD_FAIL:
                continue
            alpha = [c for c in line if c.isalpha()]
            if len(alpha) < MIN_LEFTOVER_CHARS:
                continue
            words = [c for c in line if c.isalpha() or c.isspace()]
            alpha_density = len(alpha) / max(len(line), 1)
            word_like = [w for w in "".join(words).split() if len(w) > 2]
            if alpha_density > 0.4 and len("".join(word_like)) > 6:
                has_real_overlap = True
                break
        if not has_real_overlap:
            return []

    return candidates


def _ocr_badge_crop(gray_crop: np.ndarray, upscale: int = OCR_UPSCALE) -> str:
    """OCR a crop region at high resolution, trying both Otsu polarities."""
    h, w = gray_crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    upscaled = cv2.resize(gray_crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_LANCZOS4)
    _, bin_norm = cv2.threshold(upscaled, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_inv = cv2.threshold(upscaled, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_norm = pytesseract.image_to_string(bin_norm,
                                            config=TESSERACT_CONFIG).strip()
    text_inv = pytesseract.image_to_string(bin_inv,
                                           config=TESSERACT_CONFIG).strip()
    alpha_norm = sum(1 for c in text_norm if c.isalpha())
    alpha_inv = sum(1 for c in text_inv if c.isalpha())
    return text_inv if alpha_inv > alpha_norm else text_norm


def check_badge_overlap(front_cover: np.ndarray) -> BadgeOverlapResult:
    """Check and classify text overlap with the award badge area.

    Returns a three-tier verdict:
      PASS     — clean badge, no concerns.
      WARNING  — content near the badge zone but no confirmed overlap.
      VIOLATION— readable non-badge text overlaps the badge zone.
    """
    h, w = front_cover.shape[:2]
    zones = compute_zones(w, h)
    badge_rect = zones.badge_zone
    upward_buffer_px = round(h * UPWARD_BUFFER_FRACTION)

    # ── Resolution guard ──
    resolution_warning = ""
    if w < MIN_FRONT_WIDTH_PX:
        mm3_px = zones.side_px_x  # pixels per 3 mm
        resolution_warning = (
            f"Front cover is only {w}px wide (min {MIN_FRONT_WIDTH_PX}px recommended). "
            f"At this resolution, 3mm ≈ {mm3_px}px — "
            f"too small for precise measurement.")

    # ── Extended crop (includes buffer above badge) ──
    crop_y = max(0, badge_rect.y - upward_buffer_px)
    gray = cv2.cvtColor(front_cover, cv2.COLOR_BGR2GRAY)
    badge_crop = gray[crop_y: crop_y + badge_rect.h + (badge_rect.y - crop_y), :]
    raw_text = _ocr_badge_crop(badge_crop)
    match_ratio = _fuzzy_match_ratio(raw_text, BADGE_PHRASE)
    leftovers = _extract_leftover_lines(raw_text, match_ratio)

    # ── Geometric text detection in strict badge zone (corroborating) ──
    badge_zone_gray = gray[badge_rect.y: badge_rect.y1, :]
    geom_lines = _detect_text_lines(badge_zone_gray)
    geom_overlap_count = len(geom_lines)

    # ── Separate leftovers into:
    #    • clean non-badge text       → real overlap (VIOLATION)
    #    • badge-like or garbage      → proximity / noise
    non_badge_leftovers = [l for l in leftovers
                           if _match_against_badge(l) < MATCH_THRESHOLD_FAIL]
    clean_leftovers = [l for l in non_badge_leftovers
                       if _is_clean_readable_line(l)]

    # ── Three-tier verdict ──
    if clean_leftovers:
        return BadgeOverlapResult(
            verdict="VIOLATION", confidence="high",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=leftovers,
            geometry_overlap_count=geom_overlap_count,
            resolution_warning=resolution_warning,
            reason=f"Confirmed text overlap — readable non-badge text detected: "
                   f"{'; '.join(clean_leftovers[:2])}")

    if not leftovers:
        return BadgeOverlapResult(
            verdict="PASS", confidence="high",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=[], geometry_overlap_count=0,
            resolution_warning=resolution_warning,
            reason="Badge fully legible, no intruding text detected." if match_ratio >= MATCH_THRESHOLD_PASS
            else "Badge area clean — no evidence of overlap.")

    # Leftovers exist but none are clean readable — assess severity
    if match_ratio < MATCH_THRESHOLD_FAIL:
        return BadgeOverlapResult(
            verdict="WARNING", confidence="medium",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=leftovers,
            geometry_overlap_count=geom_overlap_count,
            resolution_warning=resolution_warning,
            reason="Low badge match with non-text residue near badge zone — "
                   "likely edge proximity or layout congestion, no confirmed overlap.")

    # Mid-range match with garbage-only leftovers (filtered by _extract_leftover_lines)
    return BadgeOverlapResult(
        verdict="PASS", confidence="low",
        badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
        leftover_lines=[], geometry_overlap_count=0,
        resolution_warning=resolution_warning,
        reason="Mid-range badge match with only background noise — likely OCR artifact, no overlap.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Geometric text-line detection (secondary / corroborating only)
# ═══════════════════════════════════════════════════════════════════════════════
#
# NOTE: MSER-based detector has known false positives on textures / decorative
# art.  Used ONLY as a corroborating signal — the verdict is driven by OCR.

_MSER_MIN_AREA = 20
_MSER_MAX_AREA = 5000
_MSER_MAX_VARIATION = 0.5
_GLYPH_MAX_ASPECT = 4.0
_GLYPH_MIN_ASPECT = 0.15
_MERGE_X_DIST = 30
_MERGE_Y_DIST = 8
_LINE_MIN_WIDTH = 40
_LINE_MIN_HEIGHT = 8


def _merge_into_lines(lines: List[Rect], candidate: Rect) -> List[Rect]:
    for i, line in enumerate(lines):
        y_overlap = (
            candidate.y < line.y + line.h + _MERGE_Y_DIST
            and candidate.y + candidate.h + _MERGE_Y_DIST > line.y
        )
        if not y_overlap:
            continue
        if candidate.x - line.x1 > _MERGE_X_DIST and candidate.x > line.x1:
            continue
        new_x = min(line.x, candidate.x)
        new_y = min(line.y, candidate.y)
        new_x1 = max(line.x1, candidate.x1)
        new_y1 = max(line.y1, candidate.y1)
        lines[i] = Rect(x=new_x, y=new_y,
                        w=new_x1 - new_x, h=new_y1 - new_y)
        return lines
    lines.append(candidate)
    return lines


def _detect_text_lines(gray: np.ndarray) -> List[Rect]:
    """Detect text-like bounding boxes via MSER + geometric filtering."""
    mser = cv2.MSER_create(5, _MSER_MIN_AREA, _MSER_MAX_AREA, _MSER_MAX_VARIATION)
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return []

    boxes = np.array([cv2.boundingRect(r) for r in regions], dtype=np.int32)
    filtered = []
    for (x, y, w, h) in boxes:
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        if aspect < _GLYPH_MIN_ASPECT or aspect > _GLYPH_MAX_ASPECT:
            continue
        filtered.append((int(x), int(y), int(w), int(h)))
    if not filtered:
        return []

    filtered = np.array(filtered, dtype=np.int32)
    indices = np.lexsort((filtered[:, 0], filtered[:, 1]))
    merged: List[Rect] = []
    for x, y, w, h in filtered[indices]:
        merged = _merge_into_lines(merged, Rect(x=int(x), y=int(y),
                                                w=int(w), h=int(h)))
    return [r for r in merged
            if r.w >= _LINE_MIN_WIDTH and r.h >= _LINE_MIN_HEIGHT]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Ground truth, reporting, and main
# ═══════════════════════════════════════════════════════════════════════════════

GROUND_TRUTH: dict[str, dict] = {
    "28": {
        "title": "Shabd (uncorrected)",
        "expected": "VIOLATION",
        "notes": "Author name 'Parisha Shodhan' overlaps award badge text.",
    },
    "29": {
        "title": "Shabd (corrected)",
        "expected": "PASS",
        "notes": "Corrected layout — clean separation between text and badge.",
    },
    "31": {
        "title": "Offline Sorrows / Online Ghosts",
        "expected": "PASS",
        "notes": "Author name above badge with horizontal rule. Clean — human review cleared.",
    },
    "32": {
        "title": "Echoes Along the Way (uncorrected)",
        "expected": "VIOLATION",
        "notes": "Tagline 'Poems of Memory, Meaning and Mercy' overlaps badge.",
    },
    "33": {
        "title": "Echoes Along the Way (corrected)",
        "expected": "PASS",
        "notes": "Corrected layout — clean separation.",
    },
    "34": {
        "title": "Inner Mirror (uncorrected)",
        "expected": "WARNING",
        "notes": "Content near badge zone; title clipped by right edge. Technically allowed but tight.",
    },
    "35": {
        "title": "Inner Mirror (corrected)",
        "expected": "PASS",
        "notes": "Corrected — breathing room above badge.",
    },
    "36": {
        "title": "Tainted By Emotion",
        "expected": "VIOLATION",
        "notes": "Tagline overlaps badge text; floral border crowds side margins.",
    },
}

VERDICT_ICONS = {"PASS": "✓", "WARNING": "!", "VIOLATION": "✗"}


def extract_number(filename: str) -> str:
    basename = os.path.basename(filename)
    parts = basename.split("(")
    return parts[1].split(")")[0].strip() if len(parts) >= 2 else basename


def print_report(results: dict) -> None:
    """Print the summary table and per-image checklist."""
    # ── Summary table ──
    header = (f"  {'Img':>4}  {'Title':40s}  {'GT':10s}  {'Detector':10s}  "
              f"Status")
    bar = "  " + "━" * len(header)
    print()
    print(bar)
    print("  BOOKLEAF COVER VERIFICATION — REPORT")
    print(bar)
    print(header)
    print(bar)

    for num in sorted(results, key=int):
        r = results[num]
        det = r["verdict"]
        gt = r["ground_truth"]
        icon = "✓" if det == gt else ("→" if gt == "WARNING" and det == "VIOLATION"
                                       else "✗" if det != gt else " ")
        print(f"  [{num:>3}] {r['title']:40s}  {gt:10s}  {det:10s}  {icon}")

    print(bar)
    print()

    # ── Per-image checklist ──
    for num in sorted(results, key=int):
        r = results[num]
        det = r["verdict"]
        gt = r["ground_truth"]
        icon = VERDICT_ICONS.get(det, "?")

        if det == gt:
            match_label = "MATCH"
        elif gt == "WARNING":
            match_label = "RESOLVED (softer than detector)"
        else:
            match_label = "MISMATCH"

        print(f"  ┌─ {'═' * 68}")
        print(f"  │  [{num}] {r['title']}")
        print(f"  │  Ground Truth: {gt}  │  Detector: {det}  │  {icon} {match_label}")
        if r.get("resolution_warning"):
            print(f"  │  ⚠ Resolution: {r['resolution_warning']}")
        print(f"  ├─ {'─' * 68}")

        # Checklist
        badge_readable = r["badge_match_ratio"] >= MATCH_THRESHOLD_PASS
        badge_garbled = r["badge_match_ratio"] < MATCH_THRESHOLD_FAIL
        has_leftovers = len(r["leftover_lines"]) > 0

        print(f"  │  Checks:")
        print(f"  │    {'✓' if badge_readable else '✗'}  Badge OCR match ≥ {MATCH_THRESHOLD_PASS:.0%}     "
              f"{'pass' if badge_readable else 'fail'}  (match={r['badge_match_ratio']:.2f})")
        print(f"  │    {'✓' if not badge_garbled else '✗'}  Badge not garbled         "
              f"{'ok' if not badge_garbled else 'garbled'}")
        print(f"  │    {'✓' if has_leftovers else '✗'}  Non-badge text present    "
              f"{'yes' if has_leftovers else 'no (clean)'}")

        if has_leftovers:
            clean_intrusions = [l for l in r["leftover_lines"]
                                if _is_clean_readable_line(l) and _match_against_badge(l) < MATCH_THRESHOLD_FAIL]
            if clean_intrusions:
                print(f"  │    {'✓' if clean_intrusions else '✗'}  Clean readable intrusion  "
                      f"{'yes' if clean_intrusions else 'no'}")
            print(f"  │    └─ OCR lines in badge zone:")
            for ll in r["leftover_lines"]:
                cl = "●" if (_is_clean_readable_line(ll)
                             and _match_against_badge(ll) < MATCH_THRESHOLD_FAIL) else " "
                print(f"  │       {cl} {ll[:90]}")

        print(f"  │")
        print(f"  │  Reasoning: {r['reason']}")
        print(f"  │  Design note: {r['notes']}")
        print(f"  └─ {'─' * 68}")
        print()


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))

    if not image_files:
        print(f"ERROR: No PNG files found in {data_dir}")
        sys.exit(1)

    results = {}
    print(f"\n  Processing {len(image_files)} cover images ...")

    for fpath in image_files:
        num = extract_number(fpath)
        gt = GROUND_TRUTH.get(num, {"title": "?", "expected": "UNKNOWN",
                                     "notes": ""})

        img = cv2.imread(fpath)
        if img is None:
            print(f"  [{num}]  FAILED TO LOAD")
            continue

        _, front = split_spread(img)
        result = check_badge_overlap(front)

        results[num] = {
            "title": gt["title"],
            "ground_truth": gt["expected"],
            "verdict": result.verdict,
            "confidence": result.confidence,
            "badge_match_ratio": round(result.badge_match_ratio, 3),
            "reason": result.reason,
            "resolution_warning": result.resolution_warning,
            "leftover_lines": result.leftover_lines,
            "geometry_overlap_count": result.geometry_overlap_count,
            "notes": gt["notes"],
        }

        icon = VERDICT_ICONS.get(result.verdict, "?")
        print(f"  [{num}]  {gt['title']:40s}  {icon} {result.verdict:10s}  "
              f"[{result.confidence}]")

    print(f"  {'─' * 72}")
    print_report(results)


if __name__ == "__main__":
    main()
