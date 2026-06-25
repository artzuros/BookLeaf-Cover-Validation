#!/usr/bin/env python3
"""
cover_visualize.py — BookLeaf Cover Verification + Visual Output

Runs the same badge-overlap detection as cover_verify.py but also
produces annotated front-cover images in output/ for visual inspection.

Output images are saved as:
  output/cover_<num>_<VERDICT>.png

Only creates detailed annotation for WARNING and VIOLATION cases.
PASS images get a minimal outline for reference.

Usage:
    python cover_visualize.py
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Reuse detection logic from the main script
from cover_verify import (
    BADGE_PHRASE,
    MATCH_THRESHOLD_PASS,
    MATCH_THRESHOLD_FAIL,
    MIN_FRONT_WIDTH_PX,
    UPWARD_BUFFER_FRACTION,
    BadgeOverlapResult,
    Rect,
    Zones,
    check_badge_overlap,
    compute_zones,
    extract_number,
    split_spread,
    _detect_text_lines,
    _is_clean_readable_line,
    _match_against_badge,
    VERDICT_ICONS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Ground truth (same as cover_verify.py)
# ═══════════════════════════════════════════════════════════════════════════════

GROUND_TRUTH: dict[str, dict] = {
    "28": {"title": "Shabd (uncorrected)", "expected": "VIOLATION",
           "notes": "Author name overlaps award badge text."},
    "29": {"title": "Shabd (corrected)", "expected": "PASS",
           "notes": "Clean separation between text and badge."},
    "31": {"title": "Offline Sorrows / Online Ghosts", "expected": "PASS",
           "notes": "Clean — human review cleared."},
    "32": {"title": "Echoes Along the Way (uncorrected)", "expected": "VIOLATION",
           "notes": "Tagline overlaps badge."},
    "33": {"title": "Echoes Along the Way (corrected)", "expected": "PASS",
           "notes": "Clean separation."},
    "34": {"title": "Inner Mirror (uncorrected)", "expected": "WARNING",
           "notes": "Content near badge zone; title clipped. Technically allowed but tight."},
    "35": {"title": "Inner Mirror (corrected)", "expected": "PASS",
           "notes": "Breathing room above badge."},
    "36": {"title": "Tainted By Emotion", "expected": "VIOLATION",
           "notes": "Tagline overlaps badge; floral border crowds margins."},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Colour constants (BGR)
# ═══════════════════════════════════════════════════════════════════════════════

GREEN = (60, 200, 60)
AMBER = (20, 180, 240)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (240, 200, 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    bg: tuple[int, int, int] | None = BLACK,
) -> None:
    """Draw text with background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    if bg is not None:
        cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, colour, thickness,
                cv2.LINE_AA)


def _draw_verdict_banner(
    image: np.ndarray,
    verdict: str,
    confidence: str,
    match_ratio: float,
    resolution_warning: str = "",
) -> None:
    """Draw a top banner with the verdict and key metric."""
    h, w = image.shape[:2]
    banner_h = 56 if resolution_warning else 44

    colour = {"PASS": GREEN, "WARNING": AMBER, "VIOLATION": RED}.get(verdict, WHITE)
    icon = VERDICT_ICONS.get(verdict, "?")

    # Banner background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Verdict with colour strip
    cv2.rectangle(image, (0, 0), (8, banner_h), colour, -1)

    label = f"{icon} {verdict}  [{confidence}]  badge_match={match_ratio:.2f}"
    _draw_text(image, label, (16, banner_h - 10), colour, scale=0.65, thickness=2)

    # Resolution warning
    if resolution_warning:
        _draw_text(image, resolution_warning[:100],
                   (16, banner_h - 28), AMBER, scale=0.40, thickness=1)


def _draw_badge_zone(
    image: np.ndarray,
    badge_rect: Rect,
    verdict: str,
    zones: Zones,
) -> None:
    """Highlight the badge zone and safe area boundaries."""
    h, w = image.shape[:2]
    colour = {"PASS": GREEN, "WARNING": AMBER, "VIOLATION": RED}.get(verdict, WHITE)

    # Badge zone — filled semi-transparent
    overlay = image.copy()
    cv2.rectangle(overlay,
                  (badge_rect.x, badge_rect.y),
                  (badge_rect.x + badge_rect.w, badge_rect.y + badge_rect.h),
                  colour, -1)
    cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

    # Badge zone — outline
    thick = 3 if verdict in ("WARNING", "VIOLATION") else 2
    cv2.rectangle(image,
                  (badge_rect.x, badge_rect.y),
                  (badge_rect.x + badge_rect.w, badge_rect.y + badge_rect.h),
                  colour, thick)

    # Label
    label = f"AWARD BADGE ZONE ({badge_rect.h}px  /  {BADGE_PHRASE})"
    _draw_text(image, label,
               (badge_rect.x + 8, badge_rect.y - 8),
               colour, scale=0.45)

    # Safe area boundary (dashed-line effect via lighter colour)
    safe = zones.safe_area
    cv2.rectangle(image,
                  (safe.x, safe.y),
                  (safe.x + safe.w, safe.y + safe.h),
                  CYAN, 1)


def _draw_leftovers(
    image: np.ndarray,
    leftovers: list[str],
    badge_rect: Rect,
    upward_buffer_px: int,
) -> None:
    """Annotate OCR leftover lines near the badge zone."""
    if not leftovers:
        return

    h, w = image.shape[:2]
    line_h = 18
    x = max(8, safe_x := 12)  # use safe-left position

    # Group into clean intrusions vs garbage
    clean_lines = [l for l in leftovers
                   if _is_clean_readable_line(l)
                   and _match_against_badge(l) < MATCH_THRESHOLD_FAIL]
    garbage_lines = [l for l in leftovers if l not in clean_lines]

    # Draw a small callout box above the badge zone
    box_y = max(10, badge_rect.y - upward_buffer_px - 10)
    box_bottom = min(badge_rect.y - 4, h - 4)

    if clean_lines or garbage_lines:
        items = []
        if clean_lines:
            for cl in clean_lines:
                items.append(f"● {cl[:70]}")
        if garbage_lines:
            n = len(garbage_lines)
            items.append(f"  + {n} non-text residue line(s)")
        content = "\n".join(items)
        lines = content.split("\n")
        n = len(lines)
        box_top = box_y - n * line_h - 12
        if box_top < 4:
            box_top = 4
        cv2.rectangle(image, (6, box_top), (w - 6, box_bottom),
                      BLACK, -1)

        for i, line in enumerate(lines):
            is_clean = line.startswith("●")
            col = RED if is_clean else AMBER
            _draw_text(image, line, (x + 4, box_top + 14 + i * line_h + 10),
                       col, scale=0.45, thickness=1, bg=None)


def _draw_near_badge_text(
    image: np.ndarray,
    front_gray: np.ndarray,
    badge_rect: Rect,
    zones: Zones,
) -> None:
    """Detect and highlight text-like features near the badge zone."""
    prox_px = round(zones.dpi / 25.4 * 3)  # 3 mm in pixels

    # Crop just above badge zone
    crop_top = max(0, badge_rect.y - prox_px * 2)  # 6 mm band
    proximity_region = front_gray[crop_top:badge_rect.y, :]
    if proximity_region.shape[0] < 5:
        return

    lines = _detect_text_lines(proximity_region)
    for r in lines:
        # Shift back to full-image coords
        x, y = r.x, r.y + crop_top
        w, h2 = r.w, r.h
        if y + h2 > badge_rect.y:
            h2 = badge_rect.y - y  # clip at badge start
        if h2 < 4:
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h2), AMBER, 1)
        cv2.putText(image, "?", (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, AMBER, 1, cv2.LINE_AA)


def _draw_ocr_text_sidebar(
    image: np.ndarray,
    raw_ocr: str,
    verdict: str,
) -> None:
    """Add a sidebar or bottom strip showing raw OCR output."""
    h, w = image.shape[:2]
    lines = [l.strip() for l in raw_ocr.split("\n") if l.strip()]
    if not lines:
        return

    # Bottom strip
    n = min(len(lines), 6)
    strip_h = n * 16 + 16
    cv2.rectangle(image, (0, h - strip_h), (w, h), BLACK, -1)

    _draw_text(image, "OCR output:", (8, h - strip_h + 14),
               WHITE, scale=0.45, thickness=1, bg=None)
    for i, line in enumerate(lines[:n]):
        col = RED if (_is_clean_readable_line(line)
                      and _match_against_badge(line) < MATCH_THRESHOLD_FAIL) else WHITE
        _draw_text(image, f"  {i + 1}. {line[:80]}",
                   (8, h - strip_h + 14 + (i + 1) * 16),
                   col, scale=0.40, thickness=1, bg=None)
    if len(lines) > n:
        _draw_text(image, f"  ... ({len(lines) - n} more)",
                   (8, h - strip_h + 14 + (n + 1) * 16),
                   (150, 150, 150), scale=0.40, thickness=1, bg=None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main processing
# ═══════════════════════════════════════════════════════════════════════════════

def process_and_visualise(front_cover: np.ndarray, fh: int, fw: int,
                          result: BadgeOverlapResult,
                          raw_ocr: str,
                          num: str,
                          output_dir: str) -> str | None:
    """
    Create an annotated visualisation of the detection result.

    Returns the output file path, or None if skipped (PASS, minimal).
    """
    zones = compute_zones(fw, fh)
    badge_rect = zones.badge_zone
    verdict = result.verdict
    upward_buffer_px = round(fh * UPWARD_BUFFER_FRACTION)

    # ── Build annotated image ──
    vis = front_cover.copy()
    gray = cv2.cvtColor(front_cover, cv2.COLOR_BGR2GRAY)

    # Draw zones
    _draw_badge_zone(vis, badge_rect, verdict, zones)

    # Draw upward buffer boundary (extended OCR crop)
    crop_y = max(0, badge_rect.y - upward_buffer_px)
    cv2.rectangle(vis, (0, crop_y), (fw, badge_rect.y),
                  AMBER if verdict in ("WARNING", "VIOLATION") else (100, 100, 100), 1)

    # Draw leftovers / intruding text
    if result.leftover_lines:
        _draw_leftovers(vis, result.leftover_lines, badge_rect, upward_buffer_px)

    # Draw detected text lines near the badge zone
    if verdict in ("WARNING", "VIOLATION"):
        _draw_near_badge_text(vis, gray, badge_rect, zones)

    # OCR sidebar
    if raw_ocr and verdict in ("WARNING", "VIOLATION"):
        _draw_ocr_text_sidebar(vis, raw_ocr, verdict)

    # Banner
    _draw_verdict_banner(vis, verdict, result.confidence, result.badge_match_ratio,
                         result.resolution_warning)

    # Title on image
    gt = GROUND_TRUTH.get(num, {})
    title = gt.get("title", f"Image {num}")
    subtitle = f"[{num}] {title}  |  GT: {gt.get('expected', '?')}"
    _draw_text(vis, subtitle, (16, vis.shape[0] - 4 if not raw_ocr else vis.shape[0] - 68),
               (200, 200, 200), scale=0.50, thickness=1)

    # Reasoning
    if result.reason:
        _draw_text(vis, result.reason[:120], (16, vis.shape[0] - 4),
                   (180, 180, 180), scale=0.40, thickness=1, bg=None)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cover_{num}_{verdict}.png")
    cv2.imwrite(out_path, vis)

    return out_path


def print_report(results: dict) -> None:
    """Print the same checklist-style report as cover_verify.py."""
    header = (f"  {'Img':>4}  {'Title':40s}  {'GT':10s}  {'Detector':10s}  "
              f"Status  Visual")
    bar = "  " + "━" * len(header)
    print(f"\n{bar}")
    print("  BOOKLEAF COVER VERIFICATION — VISUAL REPORT")
    print(bar)
    print(header)
    print(bar)

    for num in sorted(results, key=int):
        r = results[num]
        det = r["verdict"]
        gt = r["ground_truth"]
        icon = "✓" if det == gt else "→" if gt == "WARNING" else "✗"
        vis = r.get("visual_output", "")
        vis_label = f" → {os.path.basename(vis)}" if vis else " (minimal)"
        print(f"  [{num:>3}] {r['title']:40s}  {gt:10s}  {det:10s}  {icon}{vis_label}")

    print(bar)
    print()

    # Per-image details
    for num in sorted(results, key=int):
        r = results[num]
        det = r["verdict"]
        gt = r["ground_truth"]
        icon = VERDICT_ICONS.get(det, "?")
        match_label = "MATCH" if det == gt else ("RESOLVED" if gt == "WARNING" else "MISMATCH")

        print(f"  ┌─ {'═' * 68}")
        print(f"  │  [{num}] {r['title']}")
        print(f"  │  GT: {gt}  │  Detector: {det}  │  {icon} {match_label}")
        if r.get("resolution_warning"):
            print(f"  │  ⚠ Resolution: {r['resolution_warning']}")
        print(f"  ├─ {'─' * 68}")
        print(f"  │  Checks:")
        badge_readable = r["badge_match_ratio"] >= MATCH_THRESHOLD_PASS
        badge_garbled = r["badge_match_ratio"] < MATCH_THRESHOLD_FAIL
        has_leftovers = len(r["leftover_lines"]) > 0
        print(f"  │    {'✓' if badge_readable else '✗'}  Badge OCR match ≥ {MATCH_THRESHOLD_PASS:.0%}     "
              f"{'pass' if badge_readable else 'fail'}  (match={r['badge_match_ratio']:.2f})")
        print(f"  │    {'✓' if not badge_garbled else '✗'}  Badge not garbled         "
              f"{'ok' if not badge_garbled else 'garbled'}")
        print(f"  │    {'✓' if has_leftovers else '✗'}  Non-badge text present    "
              f"{'yes' if has_leftovers else 'no (clean)'}")
        if has_leftovers:
            clean_intrusions = [l for l in r["leftover_lines"]
                                if _is_clean_readable_line(l)
                                and _match_against_badge(l) < MATCH_THRESHOLD_FAIL]
            for ll in r["leftover_lines"]:
                cl = "●" if (_is_clean_readable_line(ll)
                             and _match_against_badge(ll) < MATCH_THRESHOLD_FAIL) else " "
                print(f"  │       {cl} {ll[:90]}")
        print(f"  │")
        print(f"  │  Reasoning: {r['reason']}")
        print(f"  │  Notes: {r['notes']}")
        vis_path = r.get("visual_output", "")
        if vis_path:
            print(f"  │  Visual: {vis_path}")
        print(f"  └─ {'─' * 68}")
        print()


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))

    if not image_files:
        print(f"ERROR: No PNG files found in {data_dir}")
        sys.exit(1)

    results = {}
    print(f"  Processing {len(image_files)} cover images ...\n")

    for fpath in image_files:
        num = extract_number(fpath)
        gt = GROUND_TRUTH.get(num, {"title": "?", "expected": "UNKNOWN", "notes": ""})

        img = cv2.imread(fpath)
        if img is None:
            print(f"  [{num}]  FAILED TO LOAD")
            continue

        _, front = split_spread(img)
        fh, fw = front.shape[:2]

        result = check_badge_overlap(front)

        # Get raw OCR text from the internal function (we exposed it in the result)
        raw_ocr = result.raw_ocr_text

        # Only save detailed visuals for non-trivial cases
        needs_visual = result.verdict in ("WARNING", "VIOLATION")

        vis_path = None
        if needs_visual:
            vis_path = process_and_visualise(
                front, fh, fw, result, raw_ocr, num, output_dir)

        results[num] = {
            "title": gt["title"],
            "ground_truth": gt["expected"],
            "verdict": result.verdict,
            "confidence": result.confidence,
            "badge_match_ratio": round(result.badge_match_ratio, 3),
            "reason": result.reason,
            "resolution_warning": result.resolution_warning,
            "leftover_lines": result.leftover_lines,
            "notes": gt["notes"],
            "visual_output": vis_path,
        }

        icon = VERDICT_ICONS.get(result.verdict, "?")
        vis_note = f"  → {os.path.basename(vis_path)}" if vis_path else ""
        print(f"  [{num}]  {gt['title']:40s}  {icon} {result.verdict:10s}{vis_note}")

    print(f"  {'─' * 72}")
    print_report(results)


if __name__ == "__main__":
    main()
