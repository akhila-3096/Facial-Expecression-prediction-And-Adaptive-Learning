import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "emotion_learning.db"

def get_connection():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendation TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)

def create_session():
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO sessions(start_time) VALUES (?)",
            (datetime.now().isoformat(timespec="seconds"),)
        )
        return cur.lastrowid

def add_observation(session_id, emotion, confidence, recommendation):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO observations
            (session_id, timestamp, emotion, confidence, recommendation)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(timespec="seconds"),
              emotion, float(confidence), recommendation))

def end_session(session_id):
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET end_time=? WHERE id=?",
            (datetime.now().isoformat(timespec="seconds"), session_id)
        )

def get_sessions():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

def get_observations():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM observations ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

def get_database_summary():
    with get_connection() as conn:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        observations = conn.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        return {
            "total_sessions": sessions,
            "total_observations": observations
        }

init_db()
