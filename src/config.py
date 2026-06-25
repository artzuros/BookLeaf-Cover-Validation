"""Application configuration."""
import os
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Uploads and outputs
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Correct / wrong folders for sorted cover images (by verification verdict)
CORRECT_DIR = PROJECT_ROOT / "correct"
WRONG_DIR = PROJECT_ROOT / "wrong"

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8010"))

# Allowed formats
ALLOWED_IMAGE_EXTENSIONS = {".png"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS

# Visualisation
ANNOTATION_COLOUR_MAP = {
    "PASS": (60, 200, 60),
    "WARNING": (20, 180, 240),
    "VIOLATION": (60, 60, 230),
}
