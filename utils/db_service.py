"""
Hardened DB + webhook service for the enterprise pipeline.

Replaces the security/ops weaknesses of utils/posgres_service.py (which the
legacy pipeline still uses, untouched):
  * NO hardcoded credentials -- DB password, webhook token, and company id come
    from the environment (optionally a .env file); the service refuses to start
    if required secrets are missing instead of silently using a baked-in default;
  * webhook TLS verification is ON by default (was hardcoded verify=False);
  * a psycopg2 ThreadedConnectionPool replaces the open-a-new-connection-per-
    event pattern (point DB_HOST at your existing pgbouncer for extra pooling);
  * connections that go bad are discarded and replaced rather than reused.

Business logic (schema, throttling, auto in/out pairing, webhook payload) is
preserved byte-for-byte from posgres_service so behavior is unchanged.
"""
import os
import time
import logging
import traceback
from datetime import datetime
from contextlib import contextmanager
from queue import Empty

import psycopg2
from psycopg2 import pool as pgpool
import requests

from .outbox import add as outbox_add, flush as outbox_flush, DBUnavailable

log = logging.getLogger("db_service")

# psycopg2 errors that mean "Postgres is unreachable right now" (retry, don't drop),
# as opposed to data/logic errors (skip).
_DB_DOWN = (psycopg2.OperationalError, psycopg2.InterfaceError)


# --------------------------------------------------------------------------- #
# Configuration (env-driven; tiny .env loader, no external dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path=None):
    """Load KEY=VALUE lines from a .env file WITHOUT overriding real env vars."""
    path = path or os.getenv("ENV_FILE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "facial_recognition_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),          # no default -- must be set
    "port": os.getenv("DB_PORT", "5432"),
}
COMPANY_ID = os.getenv("COMPANY_ID")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:5001/api/add-entry")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN")
WEBHOOK_VERIFY_TLS = os.getenv("WEBHOOK_VERIFY_TLS", "true").lower() != "false"
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "10"))
WEBHOOK_RETRIES = int(os.getenv("WEBHOOK_RETRIES", "3"))
POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
POOL_MAX = int(os.getenv("DB_POOL_MAX", "4"))
THROTTLE_SEC = int(os.getenv("ATTENDANCE_THROTTLE_SEC", "30"))
OUTBOX_FLUSH_SEC = int(os.getenv("OUTBOX_FLUSH_SEC", "15"))   # retry buffered events every N s


def require_secrets():
    """Fail fast with a clear message if required secrets are absent."""
    missing = [name for name, val in
               [("DB_PASSWORD", DB_CONFIG["password"]),
                ("COMPANY_ID", COMPANY_ID),
                ("WEBHOOK_TOKEN", WEBHOOK_TOKEN)] if not val]
    if missing:
        raise RuntimeError(
            "Missing required secrets: " + ", ".join(missing) +
            ". Set them in the environment or a .env file (see .env.example).")


# --------------------------------------------------------------------------- #
# Connection pool
# --------------------------------------------------------------------------- #
_pool = None
_db_initialized = False


def _get_pool():
    global _pool
    if _pool is None:
        require_secrets()
        _pool = pgpool.ThreadedConnectionPool(POOL_MIN, POOL_MAX, **DB_CONFIG)
        log.info("DB pool created (min=%d max=%d host=%s)",
                 POOL_MIN, POOL_MAX, DB_CONFIG["host"])
    return _pool


@contextmanager
def get_db_connection():
    """Borrow a pooled connection; discard it from the pool if it goes bad."""
    global _db_initialized
    pool = _get_pool()
    conn = pool.getconn()
    broken = False
    try:
        if not _db_initialized:
            _init_db_and_migrate(conn)
            _db_initialized = True
        yield conn
    except Exception as e:
        broken = True
        log.error("DB error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn, close=broken)


def _init_db_and_migrate(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance1 (
                id SERIAL PRIMARY KEY,
                emp_id VARCHAR(255) NOT NULL,
                company_id UUID NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                attendance_date DATE NOT NULL,
                attendance_time TIME NOT NULL,
                check_type TEXT NOT NULL,
                image_url TEXT,
                camera_name TEXT,
                duration REAL,
                check_in_id INTEGER,
                sent_to_webhook BOOLEAN DEFAULT FALSE,
                webhook_response TEXT,
                FOREIGN KEY(emp_id, company_id)
                    REFERENCES user_data(emp_id, company_id) ON DELETE CASCADE
            )""")
        conn.commit()


# --------------------------------------------------------------------------- #
# Webhook
# --------------------------------------------------------------------------- #
def _webhook_headers():
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {WEBHOOK_TOKEN}"}


def send_to_webhook(payload):
    for attempt in range(WEBHOOK_RETRIES):
        try:
            r = requests.post(WEBHOOK_URL, headers=_webhook_headers(),
                              json=payload, timeout=WEBHOOK_TIMEOUT,
                              verify=WEBHOOK_VERIFY_TLS)
            if r.status_code in (200, 201):
                return True, r.text
            log.warning("webhook %s -> %s", r.status_code, r.text[:200])
        except requests.exceptions.RequestException as e:
            log.warning("webhook attempt %d failed: %s", attempt + 1, e)
        time.sleep(1)
    return False, "Max retries reached"


def mark_attendance_sent(attendance_id, response_text):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE attendance1 SET sent_to_webhook=TRUE, "
                        "webhook_response=%s WHERE id=%s",
                        (str(response_text)[:500], attendance_id))
            conn.commit()


# --------------------------------------------------------------------------- #
# Core logic (preserved from posgres_service)
# --------------------------------------------------------------------------- #
_last_processed = {}


def get_user_info(emp_id, company_id):
    """Return (first_name, last_name, emp_id, image_path) or 4x None."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT first_name, last_name, emp_id, image_path "
                        "FROM user_data WHERE emp_id=%s AND company_id=%s",
                        (str(emp_id), str(company_id)))
            row = cur.fetchone()
            return tuple(row) if row else (None, None, None, None)


def log_attendance(emp_id, company_id, first_name=None, last_name=None,
                   attendance_date=None, attendance_time=None, check_type='in',
                   image_url=None, camera_name="Entrance", image_b64=None):
    now_ts = time.time()
    if emp_id in _last_processed and (now_ts - _last_processed[emp_id] < THROTTLE_SEC):
        return False

    try:
        user = get_user_info(emp_id, company_id)
    except _DB_DOWN as e:
        raise DBUnavailable(str(e))          # buffer to the outbox, retry later
    if not user[0]:
        log.info("event ignored: id %s not in user_data", emp_id)
        return False
    db_first, db_last, _, db_image = user

    now = datetime.now()
    date_obj, time_obj = now.date(), now.time()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM attendance1
                    WHERE emp_id=%s AND company_id=%s AND attendance_date=%s
                      AND check_type='in' AND id NOT IN (
                        SELECT check_in_id FROM attendance1 WHERE check_in_id IS NOT NULL)
                    ORDER BY attendance_time DESC LIMIT 1
                """, (str(emp_id), str(company_id), date_obj))
                open_checkin = cur.fetchone()
                final_type, parent_id = ('out', open_checkin[0]) if open_checkin else ('in', None)

                cur.execute("""
                    INSERT INTO attendance1 (emp_id, company_id, first_name, last_name,
                        attendance_date, attendance_time, check_type, camera_name,
                        image_url, check_in_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """, (str(emp_id), str(company_id), db_first, db_last, date_obj,
                      time_obj, final_type, camera_name, db_image, parent_id))
                new_id = cur.fetchone()[0]
                conn.commit()

        _last_processed[emp_id] = now_ts
        payload = {"emp_id": str(emp_id), "company_id": str(company_id),
                   "first_name": db_first, "last_name": db_last,
                   "check_type": final_type, "check_in_id": parent_id,
                   "attendance_date": date_obj.isoformat(),
                   "attendance_time": time_obj.isoformat(), "image_url": db_image,
                   "image_b64": image_b64}  # live snapshot (proof-of-presence)
        ok, resp = send_to_webhook(payload)
        if ok:
            mark_attendance_sent(new_id, resp)
        return True
    except _DB_DOWN as e:
        raise DBUnavailable(str(e))          # buffer to the outbox, retry later
    except Exception as e:
        log.error("log_attendance failed: %s\n%s", e, traceback.format_exc())
        return False


def _flush_outbox():
    """Best-effort drain of buffered events (no-op/cheap when the outbox is empty)."""
    try:
        outbox_flush(log_attendance)
    except Exception as e:  # never let a flush hiccup kill the worker
        log.warning("outbox flush error: %s", e)


def attendance_worker(q):
    """Separate-process consumer of attendance tasks.

    Never drops: if Postgres is unavailable the event is buffered to a durable
    local outbox and replayed automatically once the DB recovers. Also drains any
    events left over from a previous run (crash recovery) on startup.
    """
    log.info("[%s] attendance worker started", os.getpid())
    _flush_outbox()                                   # crash recovery
    last_flush = time.time()
    while True:
        try:
            task = q.get(timeout=OUTBOX_FLUSH_SEC)     # wake periodically to retry backlog
        except Empty:
            _flush_outbox()
            last_flush = time.time()
            continue
        if task is None:
            _flush_outbox()
            log.info("[%s] worker shutting down", os.getpid())
            break
        try:
            log_attendance(**task)
        except DBUnavailable:
            outbox_add(task)                          # durable buffer — no loss
        except Exception as e:
            log.error("worker error: %s", e)
            traceback.print_exc()
        if time.time() - last_flush > OUTBOX_FLUSH_SEC:
            _flush_outbox()                           # drain backlog while streaming
            last_flush = time.time()
