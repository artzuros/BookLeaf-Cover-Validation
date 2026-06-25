"""Pydantic response models for the cover verification API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.detector import (
    MATCH_THRESHOLD_FAIL,
    is_clean_readable_line,
    match_against_badge,
)


# ── Nested models ────────────────────────────────────────────────────────────

class Dimensions(BaseModel):
    width: int
    height: int


class FileMetadata(BaseModel):
    filename: str
    format: str          # "PNG" | "PDF"
    pages: int           # page count (1 for images, N for PDF)
    dimensions: Dimensions | None = None


class SpecificIssue(BaseModel):
    marker: str          # e.g. "❌ Text overlap detected"
    details: str         # human-readable detail
    location: str = ""   # e.g. "Badge zone (bottom 9mm of cover)"


class CorrectionInstructions(BaseModel):
    clear_status: str           # "PASSED" | "FAILED" | "REVIEW_NEEDED"
    specific_issues: List[SpecificIssue]
    step_by_step: List[str]     # ordered correction instructions


class RevisionTracking(BaseModel):
    submission_id: str
    revision_id: str = ""
    version: int = 1
    previous_status: str | None = None
    author_name: str = ""
    book_name: str = ""
    isbn: str = ""


# ── Top-level response ───────────────────────────────────────────────────────

class VerificationResponse(BaseModel):
    airtable_book_id: str                   # ISBN extracted from filename
    metadata: FileMetadata
    detection_timestamp: str                # ISO-8601
    issue_type: str                         # "text_overlap" | "edge_proximity" | "low_resolution" | "none"
    severity: str                           # "critical" | "warning" | "info"
    status: str                             # "Pass" | "Review Needed"
    confidence_score: int                   # 0-100
    visual_annotations_url: str | None = None
    correction_instructions: CorrectionInstructions
    revision_tracking: RevisionTracking


# ── Error response ───────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


# ── Factory ──────────────────────────────────────────────────────────────────

def build_response(
    isbn: str,
    filename: str,
    file_format: str,
    dimensions: tuple[int, int] | None,
    verdict: str,
    confidence: str,
    match_ratio: float,
    reason: str,
    leftovers: List[str],
    resolution_warning: str,
    dpi: float,
    visual_url: str | None = None,
    submission_id: str | None = None,
    *,
    author_name: str = "",
    book_name: str = "",
    revision_id: str = "",
    version: int = 1,
) -> VerificationResponse:
    """Build a full VerificationResponse from detection results."""

    # ── Map verdict → status, issue_type, severity ──
    if verdict == "VIOLATION":
        status = "Review Needed"
        issue_type = "text_overlap"
        severity = "critical"
        clear_status = "FAILED"
    elif verdict == "WARNING":
        status = "Review Needed"
        issue_type = "edge_proximity"
        severity = "warning"
        clear_status = "REVIEW_NEEDED"
    else:  # PASS
        status = "Pass"
        issue_type = "none"
        severity = "info"
        clear_status = "PASSED"

    # Override to low_resolution if the guard fired
    if resolution_warning and verdict == "PASS":
        issue_type = "low_resolution"
        severity = "info"

    # ── Confidence score 0-100 ──
    confidence_map = {"high": 90, "medium": 70, "low": 50}
    base_score = confidence_map.get(confidence, 50)
    # Adjust by match ratio
    score = int(base_score * match_ratio) if match_ratio < 0.85 else base_score
    score = max(10, min(99, score))
    if resolution_warning:
        score = max(10, score - 10)

    # ── Specific issues ──
    issues: List[SpecificIssue] = []
    if verdict == "VIOLATION":
        for ll in leftovers:
            if is_clean_readable_line(ll) and match_against_badge(ll) < MATCH_THRESHOLD_FAIL:
                issues.append(SpecificIssue(
                    marker="❌ Text overlap detected",
                    details=f"Readable text found in badge zone: \"{ll[:80]}\"",
                    location="Badge zone (bottom 9mm of front cover)",
                ))
        if not issues:
            issues.append(SpecificIssue(
                marker="❌ Text overlap detected",
                details="Non-badge text overlapping the award badge zone.",
                location="Badge zone",
            ))
    elif verdict == "WARNING":
        issues.append(SpecificIssue(
            marker="⚠ Content near badge zone",
            details=reason,
            location="Area immediately above badge zone",
        ))
    if resolution_warning:
        issues.append(SpecificIssue(
            marker="❌ Low resolution",
            details=resolution_warning,
            location=f"Entire cover ({dpi:.0f} DPI)",
        ))
    if verdict == "PASS" and not issues:
        issues.append(SpecificIssue(
            marker="✓ No issues detected",
            details="Badge zone is clean, no overlapping text found.",
            location="Badge zone",
        ))

    # ── Step-by-step corrections ──
    steps = _build_correction_steps(verdict, resolution_warning)

    # ── Dimensions ──
    dims = Dimensions(width=dimensions[0], height=dimensions[1]) if dimensions else None

    return VerificationResponse(
        airtable_book_id=isbn,
        metadata=FileMetadata(
            filename=filename,
            format=file_format,
            pages=1,
            dimensions=dims,
        ),
        detection_timestamp=datetime.now(timezone.utc).isoformat(),
        issue_type=issue_type,
        severity=severity,
        status=status,
        confidence_score=score,
        visual_annotations_url=visual_url,
        correction_instructions=CorrectionInstructions(
            clear_status=clear_status,
            specific_issues=issues,
            step_by_step=steps,
        ),
        revision_tracking=RevisionTracking(
            submission_id=submission_id or uuid4().hex[:12],
            revision_id=revision_id,
            version=version,
            author_name=author_name,
            book_name=book_name,
            isbn=isbn,
        ),
    )


def _build_correction_steps(verdict: str, resolution_warning: str) -> List[str]:
    steps: List[str] = []
    if verdict == "VIOLATION":
        steps.append("Increase vertical clearance between cover text and the award badge zone (bottom 9mm).")
        steps.append("Ensure no text lines extend into the 9mm badge-reserved area.")
        steps.append("Keep all non-badge content at least 3mm above the badge zone boundary.")
        steps.append("Re-export the cover file and re-submit for verification.")
    elif verdict == "WARNING":
        steps.append("Review the area immediately above the award badge zone for content proximity.")
        steps.append("If possible, shift layout elements upward by 2-3mm for better clearance.")
        steps.append("Ensure the title/author block does not crowd the badge zone.")
        steps.append("Re-submit for verification after adjustments.")
    else:
        steps.append("No corrections needed — cover passes badge zone verification.")

    if resolution_warning:
        steps.insert(0, "Upload a higher-resolution file (minimum 1000px front-cover width for accurate measurement).")
    return steps
