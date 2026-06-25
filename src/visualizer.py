"""Generate annotated visual outputs for badge overlap results."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.config import OUTPUT_DIR, ANNOTATION_COLOUR_MAP
from src.detector import (
    BadgeOverlapResult,
    UPWARD_BUFFER_FRACTION,
    Rect,
    compute_zones,
    detect_text_lines,
    is_clean_readable_line,
    match_against_badge,
    MATCH_THRESHOLD_FAIL,
)

GREEN = (60, 200, 60)
AMBER = (20, 180, 240)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (240, 200, 80)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    bg: tuple[int, int, int] | None = BLACK,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    if bg is not None:
        cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, colour, thickness,
                cv2.LINE_AA)


def generate_annotation(
    front_cover: np.ndarray,
    result: BadgeOverlapResult,
    isbn: str,
) -> str | None:
    """Create an annotated PNG for the detection result.

    Returns the file path (relative to OUTPUT_DIR) or None if PASS+no warning.
    """
    if result.verdict == "PASS" and not result.resolution_warning:
        return None  # nothing interesting to annotate

    h, w = front_cover.shape[:2]
    zones = compute_zones(w, h)
    badge_rect = zones.badge_zone
    verdict = result.verdict
    upward_buffer_px = round(h * UPWARD_BUFFER_FRACTION)

    colour = ANNOTATION_COLOUR_MAP.get(verdict, WHITE)
    vis = front_cover.copy()
    gray = cv2.cvtColor(front_cover, cv2.COLOR_BGR2GRAY)

    # ── Badge zone fill & outline ──
    overlay = vis.copy()
    cv2.rectangle(overlay,
                  (badge_rect.x, badge_rect.y),
                  (badge_rect.x + badge_rect.w, badge_rect.y + badge_rect.h),
                  colour, -1)
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    thick = 3 if verdict in ("WARNING", "VIOLATION") else 2
    cv2.rectangle(vis,
                  (badge_rect.x, badge_rect.y),
                  (badge_rect.x + badge_rect.w, badge_rect.y + badge_rect.h),
                  colour, thick)

    # ── Safe area boundary ──
    safe = zones.safe_area
    cv2.rectangle(vis, (safe.x, safe.y), (safe.x + safe.w, safe.y + safe.h), CYAN, 1)

    # ── Upward buffer ──
    crop_y = max(0, badge_rect.y - upward_buffer_px)
    cv2.rectangle(vis, (0, crop_y), (w, badge_rect.y),
                  AMBER if verdict in ("WARNING", "VIOLATION") else (100, 100, 100), 1)

    # ── Leftovers / intrusions ──
    if result.leftover_lines:
        clean_lines = [l for l in result.leftover_lines
                       if is_clean_readable_line(l)
                       and match_against_badge(l) < MATCH_THRESHOLD_FAIL]
        items = [f"● {cl[:70]}" for cl in clean_lines]
        if result.verdict == "WARNING":
            items.append(f"  + {len(result.leftover_lines)} non-text line(s)")
        if items:
            box_y = max(10, badge_rect.y - upward_buffer_px - 10)
            n = len(items)
            box_top = box_y - n * 18 - 12
            if box_top < 4:
                box_top = 4
            cv2.rectangle(vis, (6, box_top), (w - 6, badge_rect.y - 4), BLACK, -1)
            for i, line in enumerate(items):
                col = RED if line.startswith("●") else AMBER
                _draw_text(vis, line, (12, box_top + 14 + i * 18 + 10),
                           col, scale=0.45, thickness=1, bg=None)

    # ── Near-badge text markers ──
    if verdict in ("WARNING", "VIOLATION"):
        prox_px = round(zones.dpi / 25.4 * 3)
        prox_region = gray[max(0, badge_rect.y - prox_px * 2):badge_rect.y, :]
        if prox_region.shape[0] >= 5:
            for r in detect_text_lines(prox_region):
                x2, y2 = r.x, r.y + max(0, badge_rect.y - prox_px * 2)
                h2 = min(r.h, badge_rect.y - y2)
                if h2 >= 4:
                    cv2.rectangle(vis, (x2, y2), (x2 + r.w, y2 + h2), AMBER, 1)

    # ── OCR sidebar ──
    if result.raw_ocr_text and verdict in ("WARNING", "VIOLATION"):
        ocr_lines = [l.strip() for l in result.raw_ocr_text.split("\n") if l.strip()]
        n = min(len(ocr_lines), 6)
        strip_h = n * 16 + 16
        cv2.rectangle(vis, (0, h - strip_h), (w, h), BLACK, -1)
        _draw_text(vis, "OCR output:", (8, h - strip_h + 14), WHITE, 0.45, 1, bg=None)
        for i, line in enumerate(ocr_lines[:n]):
            col = RED if (is_clean_readable_line(line)
                          and match_against_badge(line) < MATCH_THRESHOLD_FAIL) else WHITE
            _draw_text(vis, f"  {i+1}. {line[:80]}",
                       (8, h - strip_h + 14 + (i + 1) * 16), col, 0.40, 1, bg=None)
        if len(ocr_lines) > n:
            _draw_text(vis, f"  ... ({len(ocr_lines)-n} more)",
                       (8, h - strip_h + 14 + (n + 1) * 16), (150, 150, 150), 0.40, 1, bg=None)

    # ── Top banner ──
    banner_h = 56 if result.resolution_warning else 44
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.85, vis, 0.15, 0, vis)
    cv2.rectangle(vis, (0, 0), (8, banner_h), colour, -1)
    icons = {"PASS": "✓", "WARNING": "!", "VIOLATION": "✗"}
    _draw_text(vis, f"{icons.get(verdict,'?')} {verdict}  [{result.confidence}]  "
               f"match={result.badge_match_ratio:.2f}",
               (16, banner_h - 10), colour, 0.65, 2)
    if result.resolution_warning:
        _draw_text(vis, result.resolution_warning[:100],
                   (16, banner_h - 28), AMBER, 0.40, 1)

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"verify_{isbn}_{verdict}.png"
    cv2.imwrite(str(out_path), vis)
    return str(out_path)
