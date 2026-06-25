"""BookLeaf Cover Verification API — FastAPI server.

Usage:
    uvicorn src.main:app --host 0.0.0.0 --port 8010 --reload
    # or
    python -m src.main
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import HOST, PORT, OUTPUT_DIR
from src.database import init_db
from src.routers.verify import router as verify_router

app = FastAPI(
    title="BookLeaf Cover Verification API",
    description="Detect text overlap with the award badge zone on book covers. "
                "Supports PNG and PDF uploads with filename convention ISBN_text.ext.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files for visual annotations ──────────────────────────────────────
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ── Database initialisation ─────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    """Ensure the SQLite database and seed data exist on app boot."""
    init_db()


# ── Admin endpoints ─────────────────────────────────────────────────────────

@app.get("/books")
async def list_books():
    """List all books in the revision-tracking database."""
    from src.database import list_books as _list_books
    return {"books": _list_books()}


@app.get("/revisions")
async def list_revisions():
    """List all revision records."""
    from src.database import list_revisions as _list_revisions
    return {"revisions": _list_revisions()}


# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(verify_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "cover-verification"}


@app.get("/")
async def root():
    return {
        "service": "BookLeaf Cover Verification",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /verify": "Upload a cover file for badge overlap detection",
            "GET  /books": "List books with ISBNs in the revision DB",
            "GET  /revisions": "List revision records",
        },
    }


def main():
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
