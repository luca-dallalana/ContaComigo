"""
Feedback store for the IRS Assistant.

Persists user ratings (helpful / not helpful) for question-answer pairs
to a local SQLite file. The schema is append-only: ratings are never
updated or deleted, providing a clean audit trail.
"""

import json
import sqlite3

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT    NOT NULL,
    answer     TEXT    NOT NULL,
    sources    TEXT    NOT NULL,
    rating     INTEGER NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


def init_db(db_path: str) -> None:
    """Create the feedback table if it does not already exist.

    Args:
        db_path: Path to the SQLite file (created if absent).
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)


def save_feedback(
    db_path: str,
    question: str,
    answer: str,
    sources: list[str],
    rating: int,
) -> None:
    """Persist a single user rating.

    Args:
        db_path: Path to the SQLite file.
        question: The user's original question.
        answer: The assistant's response text.
        sources: List of source labels shown to the user
            (e.g. "Principais Prazos IRS 2025 — DE ABRIL A JUNHO DE 2026").
        rating: 1 for helpful, -1 for not helpful.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback (question, answer, sources, rating) VALUES (?, ?, ?, ?)",
            (question, answer, json.dumps(sources, ensure_ascii=False), rating),
        )
