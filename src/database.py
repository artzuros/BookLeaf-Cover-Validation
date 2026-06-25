"""SQLite database for book cover revision tracking.

Stores author/book metadata, generated ISBNs, and revision history
so the /verify endpoint can return proper revision-tracking fields.
"""

from __future__ import annotations

import sqlite3
import uuid
import random
import string
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "cover_revisions.db"

# ── Seed data ────────────────────────────────────────────────────────────────
# Pre-populated author + book pairs; each gets a random ISBN on first init.
SEED_DATA: list[tuple[str, str]] = [
    ("Sara Johnson",      "Whispers of the Soul"),
    ("John Smith",        "The Last Horizon"),
    ("Emily Davis",       "Echoes of Tomorrow"),
    ("Michael Brown",     "Shadows in the Mist"),
    ("Sarah Wilson",      "Dancing with the Stars"),
    ("James Anderson",    "The Silent Echo"),
    ("Lisa Thompson",     "Beyond the Veil"),
    ("Robert Taylor",     "Crimson Dawn"),
    ("Amanda White",      "Starlight Serenade"),
    ("David Lee",         "The Forgotten Path"),
    ("Maria Garcia",      "Letters to the Moon"),
    ("Christopher Wang",  "The Clockwork Heart"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_isbn() -> str:
    """Generate a random 13-digit ISBN-like identifier."""
    prefix = random.choice(["978", "979"])
    body = "".join(random.choices(string.digits, k=10))
    return prefix + body


# ── Connection ───────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Open a connection to the local SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Initialisation (call once at app startup) ────────────────────────────────

def init_db() -> None:
    """Create tables and seed with author/book pairs if the DB is empty."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS books (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                author_name  TEXT    NOT NULL,
                book_name    TEXT    NOT NULL,
                isbn         TEXT    UNIQUE,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS revisions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id           INTEGER NOT NULL,
                revision_id       TEXT    NOT NULL UNIQUE,
                version           INTEGER NOT NULL DEFAULT 1,
                verdict           TEXT    NOT NULL,
                original_filename TEXT,
                stored_filename   TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books(id)
            );
        """)

        # Seed only when the books table is empty
        count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        if count == 0:
            for author, book in SEED_DATA:
                isbn = _generate_isbn()
                conn.execute(
                    "INSERT OR IGNORE INTO books (author_name, book_name, isbn) VALUES (?, ?, ?)",
                    (author, book, isbn),
                )
            conn.commit()
    finally:
        conn.close()


# ── Book lookups ─────────────────────────────────────────────────────────────

def lookup_or_create_book(
    author_name: str,
    book_name: str,
) -> tuple[int, str, str, str]:
    """Look up a book by author + title, or create a fresh row with a random ISBN.

    Returns ``(book_id, author_name, book_name, isbn)``.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, author_name, book_name, isbn FROM books "
            "WHERE author_name = ? AND book_name = ?",
            (author_name, book_name),
        ).fetchone()

        if row is not None:
            return (row["id"], row["author_name"], row["book_name"], row["isbn"])

        # New book → generate ISBN
        isbn = _generate_isbn()
        conn.execute(
            "INSERT INTO books (author_name, book_name, isbn) VALUES (?, ?, ?)",
            (author_name, book_name, isbn),
        )
        conn.commit()
        book_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return (book_id, author_name, book_name, isbn)
    finally:
        conn.close()


def get_book_by_isbn(isbn: str) -> Optional[dict]:
    """Return book info dict for a given ISBN, or ``None``."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, author_name, book_name, isbn FROM books WHERE isbn = ?",
            (isbn,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_book(author_name: str, book_name: str, isbn: str) -> dict:
    """Insert a book with a specific ISBN (does not generate a new one).

    Raises ``ValueError`` if the ISBN already exists in the database.
    Returns ``{id, author_name, book_name, isbn}``.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO books (author_name, book_name, isbn) VALUES (?, ?, ?)",
            (author_name, book_name, isbn),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, author_name, book_name, isbn FROM books WHERE isbn = ?",
            (isbn,),
        ).fetchone()
        return dict(row)  # type: ignore[return-value]
    except sqlite3.IntegrityError:
        raise ValueError(f"A book with ISBN {isbn} already exists in the database.")
    finally:
        conn.close()


def list_books() -> list[dict]:
    """Return every book in the database."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, author_name, book_name, isbn, created_at FROM books "
            "ORDER BY author_name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Revision tracking ────────────────────────────────────────────────────────

def create_revision(
    book_id: int,
    verdict: str,
    original_filename: str,
    stored_filename: str,
) -> dict:
    """Insert a revision record and return ``{revision_id, version}``."""
    conn = get_connection()
    try:
        max_version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM revisions WHERE book_id = ?",
            (book_id,),
        ).fetchone()[0]

        version = max_version + 1
        revision_id = uuid.uuid4().hex[:12]

        conn.execute(
            "INSERT INTO revisions "
            "(book_id, revision_id, version, verdict, original_filename, stored_filename) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_id, revision_id, version, verdict, original_filename, stored_filename),
        )
        conn.commit()

        return {"revision_id": revision_id, "version": version}
    finally:
        conn.close()


def list_revisions(book_id: Optional[int] = None) -> list[dict]:
    """Return revision records, optionally filtered by book_id."""
    conn = get_connection()
    try:
        if book_id is not None:
            rows = conn.execute(
                "SELECT r.*, b.author_name, b.book_name, b.isbn "
                "FROM revisions r JOIN books b ON r.book_id = b.id "
                "WHERE r.book_id = ? ORDER BY r.version DESC",
                (book_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT r.*, b.author_name, b.book_name, b.isbn "
                "FROM revisions r JOIN books b ON r.book_id = b.id "
                "ORDER BY r.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
