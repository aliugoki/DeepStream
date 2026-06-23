# utils/probe/tracking.py

import time
from datetime import date

# (source_id, person_id, date) -> dict
BEST_FACE_STATE = {}

# Cleanup tuning
_RETENTION_SECONDS = 60 * 60 * 24 * 2   # keep 2 days
_CLEAN_INTERVAL = 300                  # every 5 minutes
_last_cleanup_ts = 0


def cleanup_best_face_state():
    """Remove stale BEST_FACE_STATE entries."""
    global _last_cleanup_ts

    now = time.time()
    if now - _last_cleanup_ts < _CLEAN_INTERVAL:
        return

    today = date.today()
    keys_to_delete = []

    for key, value in BEST_FACE_STATE.items():
        _, _, entry_date = key
        if (today - entry_date).days >= 2:
            keys_to_delete.append(key)

    for k in keys_to_delete:
        BEST_FACE_STATE.pop(k, None)

    _last_cleanup_ts = now
