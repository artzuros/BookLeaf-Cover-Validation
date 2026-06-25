"""FastAPI router for the /verify endpoint — now with revision tracking."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, UploadFile, File

from src.config import (
    UPLOAD_DIR,
    CORRECT_DIR,
    WRONG_DIR,
    ALLOWED_EXTENSIONS,
)
from src.database import get_book_by_isbn, create_book, create_revision
from src.detector import check_badge_overlap, split_spread
from src.filename_parser import parse_label
from src.models import build_response, ErrorResponse
from src.pdf_converter import pdf_page_to_ndarray, HAS_PDF_SUPPORT
from src.visualizer import generate_annotation

router = APIRouter(prefix="/verify", tags=["verification"])


@router.post("", summary="Verify a book cover for badge overlap")
async def verify_cover(
    file: UploadFile = File(...),
    isbn: str = Form(...),
    author_name: str = Form(""),
    book_name: str = Form(""),
) -> dict:
    """Upload a book cover image (PNG or PDF) to check for award badge overlap.

    The file should follow the naming convention ``ISBN_text.extension``,
    e.g. ``9781234567890_front_cover.png``.

    ``isbn`` is **required** — used to look up the book in the local
    revision-tracking database.  If the ISBN isn't found and both
    ``author_name`` and ``book_name`` are provided, a new book entry is
    created automatically.

    After verification the uploaded file is renamed to ``{ISBN}_cover.ext``
    and saved under the **correct/** (PASS) or **wrong/** (VIOLATION / WARNING)
    directory.
    """
    # ── Validate filename extension ──
    original_name = file.filename or "unknown"
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Resolve ISBN against the database ────────────────────────────────
    book_data = get_book_by_isbn(isbn)
    book_id: int | None = None

    if book_data:
        # ISBN exists — use the stored author/book info
        book_id = book_data["id"]
        db_author = book_data["author_name"]
        db_book = book_data["book_name"]
        db_isbn = book_data["isbn"]
    elif author_name and book_name:
        # New ISBN — register it with the provided author + book
        entry = create_book(author_name, book_name, isbn)
        book_id = entry["id"]
        db_author = entry["author_name"]
        db_book = entry["book_name"]
        db_isbn = entry["isbn"]
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No book found with ISBN {isbn}. "
                   "Provide author_name and book_name to register a new entry.",
        )

    # ── Save uploaded file with ISBN-based name ────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{isbn}_cover{ext}"
    save_path = UPLOAD_DIR / stored_name
    try:
        with open(save_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File write error: {exc}")

    # ── Process image ─────────────────────────────────────────────────────
    try:
        if ext == ".pdf":
            if not HAS_PDF_SUPPORT:
                raise HTTPException(
                    status_code=500,
                    detail="PDF support requires PyMuPDF. Install with: pip install PyMuPDF",
                )
            image, page_count = pdf_page_to_ndarray(save_path)
            file_format = "PDF"
        else:
            import cv2

            image = cv2.imread(str(save_path))
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode PNG image.")
            file_format = "PNG"

        # Split spread → front cover
        _, front = split_spread(image)
        fh, fw = front.shape[:2]

        # Run detection
        result = check_badge_overlap(front)

        # Generate visual annotation (if warranted)
        visual_path = generate_annotation(front, result, isbn)
        visual_url = visual_path if visual_path else None

        # ── Sort into correct / wrong ─────────────────────────────────────
        is_correct = result.verdict == "PASS"
        target_dir = CORRECT_DIR if is_correct else WRONG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / stored_name
        shutil.copy2(save_path, dest_path)

        # ── Create revision record ────────────────────────────────────────
        rev_info = {"revision_id": "", "version": 1}
        if book_id is not None:
            rev_info = create_revision(
                book_id=book_id,
                verdict=result.verdict,
                original_filename=original_name,
                stored_filename=stored_name,
            )

        # ── Build enriched response ───────────────────────────────────────
        response = build_response(
            isbn=isbn,
            filename=original_name,
            file_format=file_format,
            dimensions=(fw, fh),
            verdict=result.verdict,
            confidence=result.confidence,
            match_ratio=result.badge_match_ratio,
            reason=result.reason,
            leftovers=result.leftover_lines,
            resolution_warning=result.resolution_warning,
            dpi=result.dpi,
            visual_url=visual_url,
            author_name=db_author,
            book_name=db_book,
            revision_id=rev_info["revision_id"],
            version=rev_info["version"],
        )

        return response.model_dump()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Cleanup temp upload
        if save_path.exists():
            save_path.unlink(missing_ok=True)
