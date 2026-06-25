"""PDF → PNG conversion for cover verification."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False


def pdf_page_to_ndarray(pdf_path: str | Path, page_index: int = 0,
                        dpi: int = 200) -> tuple[np.ndarray, int]:
    """Convert a PDF page to an OpenCV BGR ndarray.

    Returns (image, page_count).
    Raises RuntimeError if PyMuPDF is not installed.
    """
    if not HAS_PDF_SUPPORT:
        raise RuntimeError(
            "PDF support requires PyMuPDF. Install with: pip install PyMuPDF"
        )

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    if page_index >= page_count:
        doc.close()
        raise ValueError(
            f"Page index {page_index} exceeds document length ({page_count})"
        )

    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Convert raw pixmap bytes → numpy → BGR
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    doc.close()
    return img, page_count
