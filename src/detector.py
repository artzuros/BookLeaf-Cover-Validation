"""Core detection logic for award badge overlap on book covers."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract

# ── Spec constants ───────────────────────────────────────────────────────────
TRIM_WIDTH_INCHES = 5.0
TRIM_HEIGHT_INCHES = 8.0
SAFE_MARGIN_MM = 3.0
BADGE_HEIGHT_MM = 9.0
MM_PER_INCH = 25.4
MIN_FRONT_WIDTH_PX = 1000

# ── Badge phrase ─────────────────────────────────────────────────────────────
BADGE_PHRASE = "Winner of the 21st Century Emily Dickinson Award"

# ── Detection thresholds ─────────────────────────────────────────────────────
UPWARD_BUFFER_FRACTION = 0.077
MATCH_THRESHOLD_PASS = 0.85
MATCH_THRESHOLD_FAIL = 0.50
MIN_LEFTOVER_CHARS = 8
OCR_UPSCALE = 5
TESSERACT_CONFIG = "--oem 1 --psm 6"

# ── MSER geometry ────────────────────────────────────────────────────────────
_MSER_MIN_AREA = 20
_MSER_MAX_AREA = 5000
_MSER_MAX_VARIATION = 0.5
_GLYPH_MAX_ASPECT = 4.0
_GLYPH_MIN_ASPECT = 0.15
_MERGE_X_DIST = 30
_MERGE_Y_DIST = 8
_LINE_MIN_WIDTH = 40
_LINE_MIN_HEIGHT = 8


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Rect:
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


@dataclass
class Zones:
    dpi: float
    side_px_x: int
    side_px_y: int
    bottom_px: int
    badge_zone: Rect
    safe_area: Rect


@dataclass
class BadgeOverlapResult:
    verdict: str               # "PASS" | "WARNING" | "VIOLATION"
    confidence: str            # "high" | "medium" | "low"
    badge_match_ratio: float
    raw_ocr_text: str
    leftover_lines: List[str]
    geometry_overlap_count: int
    resolution_warning: str
    front_width: int
    front_height: int
    dpi: float
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry
# ═══════════════════════════════════════════════════════════════════════════════

def split_spread(image: np.ndarray, sobel_ksize: int = 3,
                 search_radius_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
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


def compute_zones(front_w_px: int, front_h_px: int) -> Zones:
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
#  Text helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _alnum_space(text: str) -> str:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in text)


def fuzzy_match_ratio(text: str, target: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(text),
                                   _normalize(target)).ratio()


def match_against_badge(text: str) -> float:
    return fuzzy_match_ratio(_alnum_space(text), BADGE_PHRASE)


def is_clean_readable_line(text: str, min_words: int = 2,
                           min_word_len: int = 3) -> bool:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometric text detection (secondary / corroborating)
# ═══════════════════════════════════════════════════════════════════════════════

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


def detect_text_lines(gray: np.ndarray) -> List[Rect]:
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
#  OCR
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_badge_crop(gray_crop: np.ndarray, upscale: int = OCR_UPSCALE) -> str:
    h, w = gray_crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    upscaled = cv2.resize(gray_crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_LANCZOS4)
    _, bin_norm = cv2.threshold(upscaled, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_inv = cv2.threshold(upscaled, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_norm = pytesseract.image_to_string(bin_norm, config=TESSERACT_CONFIG).strip()
    text_inv = pytesseract.image_to_string(bin_inv, config=TESSERACT_CONFIG).strip()
    alpha_norm = sum(1 for c in text_norm if c.isalpha())
    alpha_inv = sum(1 for c in text_inv if c.isalpha())
    return text_inv if alpha_inv > alpha_norm else text_norm


# ═══════════════════════════════════════════════════════════════════════════════
#  Leftover extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_leftover_lines(raw_ocr_text: str, whole_match_ratio: float) -> List[str]:
    lines = [l.strip() for l in raw_ocr_text.split("\n") if l.strip()]
    candidates: List[str] = []
    badge_alnum_space = _alnum_space(BADGE_PHRASE.lower())

    line_threshold = 0.50 if whole_match_ratio >= MATCH_THRESHOLD_PASS else MATCH_THRESHOLD_PASS

    for line in lines:
        if match_against_badge(line) >= line_threshold:
            continue
        line_clean = _alnum_space(line).lower().replace(" ", "")
        badge_flat = badge_alnum_space.replace(" ", "")
        if line_clean in badge_flat or badge_flat in line_clean:
            continue
        if BADGE_PHRASE.lower() in line.lower():
            continue
        candidates.append(line)

    if whole_match_ratio >= MATCH_THRESHOLD_PASS and candidates:
        combined_clean = " ".join(candidates)
        if match_against_badge(combined_clean) >= MATCH_THRESHOLD_PASS:
            return []
        badge_flat = badge_alnum_space.replace(" ", "")
        combined_flat = _alnum_space(combined_clean).lower().replace(" ", "")
        if badge_flat in combined_flat or combined_flat in badge_flat:
            return []

    if MATCH_THRESHOLD_FAIL <= whole_match_ratio < MATCH_THRESHOLD_PASS and candidates:
        candidates = [c for c in candidates if is_clean_readable_line(c)]
        if not candidates:
            return []

    if whole_match_ratio < MATCH_THRESHOLD_FAIL:
        has_real_overlap = False
        for line in candidates:
            if match_against_badge(line) >= MATCH_THRESHOLD_FAIL:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Main detection
# ═══════════════════════════════════════════════════════════════════════════════

def check_badge_overlap(front_cover: np.ndarray) -> BadgeOverlapResult:
    h, w = front_cover.shape[:2]
    zones = compute_zones(w, h)
    badge_rect = zones.badge_zone
    upward_buffer_px = round(h * UPWARD_BUFFER_FRACTION)

    # Resolution guard
    resolution_warning = ""
    if w < MIN_FRONT_WIDTH_PX:
        mm3_px = zones.side_px_x
        resolution_warning = (
            f"Front cover is only {w}px wide (min {MIN_FRONT_WIDTH_PX}px recommended). "
            f"At this resolution, 3mm ≈ {mm3_px}px — "
            f"too small for precise measurement.")

    # OCR the extended badge crop
    crop_y = max(0, badge_rect.y - upward_buffer_px)
    gray = cv2.cvtColor(front_cover, cv2.COLOR_BGR2GRAY)
    badge_crop = gray[crop_y: crop_y + badge_rect.h + (badge_rect.y - crop_y), :]
    raw_text = ocr_badge_crop(badge_crop)
    match_ratio = fuzzy_match_ratio(raw_text, BADGE_PHRASE)
    leftovers = extract_leftover_lines(raw_text, match_ratio)

    # Geometric text detection in strict badge zone
    badge_zone_gray = gray[badge_rect.y: badge_rect.y1, :]
    geom_lines = detect_text_lines(badge_zone_gray)
    geom_overlap_count = len(geom_lines)

    # Separate clean intrusions from garbage leftovers
    non_badge_leftovers = [l for l in leftovers
                           if match_against_badge(l) < MATCH_THRESHOLD_FAIL]
    clean_leftovers = [l for l in non_badge_leftovers
                       if is_clean_readable_line(l)]

    # Three-tier verdict
    if clean_leftovers:
        return BadgeOverlapResult(
            verdict="VIOLATION", confidence="high",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=leftovers,
            geometry_overlap_count=geom_overlap_count,
            resolution_warning=resolution_warning,
            front_width=w, front_height=h, dpi=zones.dpi,
            reason=f"Confirmed text overlap — readable non-badge text detected: "
                   f"{'; '.join(clean_leftovers[:2])}")

    if not leftovers:
        return BadgeOverlapResult(
            verdict="PASS", confidence="high",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=[], geometry_overlap_count=0,
            resolution_warning=resolution_warning,
            front_width=w, front_height=h, dpi=zones.dpi,
            reason="Badge fully legible, no intruding text detected." if match_ratio >= MATCH_THRESHOLD_PASS
            else "Badge area clean — no evidence of overlap.")

    if match_ratio < MATCH_THRESHOLD_FAIL:
        return BadgeOverlapResult(
            verdict="WARNING", confidence="medium",
            badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
            leftover_lines=leftovers,
            geometry_overlap_count=geom_overlap_count,
            resolution_warning=resolution_warning,
            front_width=w, front_height=h, dpi=zones.dpi,
            reason="Low badge match with non-text residue near badge zone — "
                   "likely edge proximity or layout congestion, no confirmed overlap.")

    return BadgeOverlapResult(
        verdict="PASS", confidence="low",
        badge_match_ratio=match_ratio, raw_ocr_text=raw_text,
        leftover_lines=[], geometry_overlap_count=0,
        resolution_warning=resolution_warning,
        front_width=w, front_height=h, dpi=zones.dpi,
        reason="Mid-range badge match with only background noise — likely OCR artifact, no overlap.")
