"""Per-user usage logging (SQLite)."""

import sqlite3
from datetime import datetime, timezone

from app.config import USAGE_DB_PATH


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(USAGE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            ts TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_username ON usage(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts)")
    return conn


def log(
    username: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO usage (username, ts, prompt_tokens, completion_tokens, cost_usd)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                username,
                datetime.now(timezone.utc).isoformat(),
                int(prompt_tokens),
                int(completion_tokens),
                float(cost_usd),
            ),
        )


def aggregate_by_user(since_iso: str | None = None) -> list[dict]:
    sql = (
        "SELECT username, COUNT(*), COALESCE(SUM(prompt_tokens), 0),"
        " COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost_usd), 0)"
        " FROM usage"
    )
    params: tuple = ()
    if since_iso:
        sql += " WHERE ts >= ?"
        params = (since_iso,)
    sql += " GROUP BY username ORDER BY 5 DESC"

    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        {
            "username": r[0],
            "queries": r[1],
            "prompt_tokens": r[2],
            "completion_tokens": r[3],
            "cost_usd": r[4],
        }
        for r in rows
    ]


def month_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def day_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
