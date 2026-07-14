"""Durable attendance outbox — a local, crash-safe buffer so no recognition is
lost when PostgreSQL is briefly unavailable.

Design (the "transactional outbox" / local-WAL pattern):
  * PostgreSQL stays the source of truth. The outbox only holds events that
    could NOT be written yet (DB down/unreachable).
  * On a DB outage, ``add()`` fsync-persists the raw event to a SQLite file on
    the pipeline's persistent volume. When the DB recovers, ``flush()`` replays
    them in order and deletes each once it lands in Postgres.
  * One file per company (``COMPANY_ID``) so concurrent company pipelines don't
    contend on a shared file.

This gives effectively-100% retention for attendance-scale traffic (bounded only
by local-disk survival) without a message broker.
"""
import json
import logging
import os
import sqlite3
import time

log = logging.getLogger("outbox")

OUTBOX_DIR = os.getenv("OUTBOX_DIR", "/workspace/data/outbox")


class DBUnavailable(Exception):
    """Raised by the writer when Postgres is unreachable — event must be retried,
    never dropped. (Distinct from an intentional skip such as throttle/unknown-id.)"""


def _path() -> str:
    company = os.getenv("COMPANY_ID", "default")
    return os.path.join(OUTBOX_DIR, f"{company}.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    c = sqlite3.connect(_path(), timeout=10)
    c.execute("PRAGMA journal_mode=WAL")     # concurrent-reader friendly
    c.execute("PRAGMA synchronous=FULL")     # fsync on commit -> survives power loss
    c.execute(
        """CREATE TABLE IF NOT EXISTS pending (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               payload TEXT NOT NULL,
               created_at REAL NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0)"""
    )
    return c


def add(task: dict) -> None:
    """Durably buffer one attendance event (DB was unavailable)."""
    with _conn() as c:
        c.execute(
            "INSERT INTO pending(payload, created_at) VALUES(?, ?)",
            (json.dumps(task, default=str), time.time()),
        )
    log.warning("attendance buffered to outbox (DB unavailable): emp=%s cam=%s",
                task.get("emp_id"), task.get("camera_name"))


def count() -> int:
    try:
        with _conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM pending").fetchone()[0])
    except Exception:
        return 0


def flush(writer, limit: int = 500) -> int:
    """Replay buffered events oldest-first via ``writer(task)`` and delete each
    that lands. Returns the number drained this call.

    ``writer`` contract:
      * returns anything truthy/falsy on success or intentional-skip  -> row removed
      * raises ``DBUnavailable``                                       -> stop, keep rows (retry later)
      * raises any other exception                                    -> row removed (un-retryable; logged)
    """
    drained = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT id, payload FROM pending ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        for rid, payload in rows:
            try:
                writer(json.loads(payload))
            except DBUnavailable:
                break  # Postgres still down — leave this and the rest for next time
            except Exception as e:  # noqa: BLE001 — poison-pill guard
                log.error("outbox: dropping un-retryable event id=%s: %s", rid, e)
            c.execute("DELETE FROM pending WHERE id=?", (rid,))
            drained += 1
    if drained:
        log.info("outbox: flushed %d buffered attendance event(s)", drained)
    return drained
