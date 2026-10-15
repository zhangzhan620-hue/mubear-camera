import json
import sqlite3
import uuid
from datetime import datetime

import cv2
import numpy as np

from . import config


def init_db():
    config.PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                best_frame INTEGER,
                person_count INTEGER,
                all_in_frame INTEGER,
                score REAL,
                detector_note TEXT,
                voice_log TEXT,
                public_url TEXT
            )
        """)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)")]
        if "public_url" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN public_url TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                idx INTEGER,
                path TEXT,
                score REAL,
                analysis_json TEXT
            )
        """)


def save_session(frames: list, analyses: list, scores: list, best_idx: int, voice_log: list) -> str:
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now()
    day_dir = config.PHOTOS_DIR / now.strftime("%Y-%m-%d") / session_id
    day_dir.mkdir(parents=True, exist_ok=True)

    best = analyses[best_idx]
    note = ""
    if not best["detector_available"]:
        note = "detector_unavailable"

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                now.isoformat(timespec="seconds"),
                best_idx,
                best["person_count"],
                1 if best["all_in_frame"] else 0,
                scores[best_idx],
                note,
                json.dumps(voice_log, ensure_ascii=False),
                None,
            ),
        )
        for i, (frame, analysis, score) in enumerate(zip(frames, analyses, scores)):
            name = "best.jpg" if i == best_idx else f"frame_{i}.jpg"
            path = day_dir / name
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                path.write_bytes(buf.tobytes())
            conn.execute(
                "INSERT INTO frames (session_id, idx, path, score, analysis_json) VALUES (?,?,?,?,?)",
                (session_id, i, str(path.relative_to(config.PHOTOS_DIR)), score,
                 json.dumps(analysis, ensure_ascii=False)),
            )
    return session_id


def get_session(session_id: str):
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            return None
        frames = conn.execute(
            "SELECT * FROM frames WHERE session_id=? ORDER BY idx", (session_id,)
        ).fetchall()
        return {"session": dict(row), "frames": [dict(f) for f in frames]}


def set_public_url(session_id: str, url: str):
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE sessions SET public_url=? WHERE id=?", (url, session_id))


def list_sessions(limit: int = 20):
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def photo_file(rel_path: str):
    path = (config.PHOTOS_DIR / rel_path).resolve()
    if config.PHOTOS_DIR.resolve() not in path.parents:
        return None
    return path if path.exists() else None


def best_photo_path(session_id: str):
    info = get_session(session_id)
    if not info:
        return None
    for f in info["frames"]:
        if f["idx"] == info["session"]["best_frame"]:
            return photo_file(f["path"])
    return None
