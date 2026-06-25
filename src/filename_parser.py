"""Parse ISBN and label from the filename convention: ISBN_text.extension"""
import re
from pathlib import Path


def parse_isbn(filename: str) -> str:
    """Extract the ISBN from the filename.

    Convention: ``ISBN_text.extension``, e.g. ``1234567890123_front_cover.png``.

    The ISBN is the first contiguous run of digits before the first underscore
    or dot.  Returns an empty string if no digits are found.
    """
    basename = Path(filename).stem  # strip extension
    m = re.match(r"(\d+)", basename)
    return m.group(1) if m else ""


def parse_label(filename: str) -> str:
    """Extract the human-readable label after the ISBN.

    Returns everything between the first underscore and the extension.
    """
    stem = Path(filename).stem
    # Remove leading ISBN and optional underscore
    rest = re.sub(r"^\d+_?", "", stem)
    return rest if rest else stem
