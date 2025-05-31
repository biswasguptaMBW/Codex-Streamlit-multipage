import sqlite3
from typing import List, Tuple


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def insert_history(conn: sqlite3.Connection, question: str, answer: str, source: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (question, answer, source) VALUES (?, ?, ?)",
        (question, answer, source),
    )
    conn.commit()


def fetch_history(conn: sqlite3.Connection) -> List[Tuple[str, str, str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT question, answer, source, timestamp FROM history ORDER BY id DESC"
    )
    return cur.fetchall()
