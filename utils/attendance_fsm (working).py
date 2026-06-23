import redis
from datetime import datetime, date

# -------------------------------
# Redis Connection
# -------------------------------
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# -------------------------------
# FSM Constants
# -------------------------------
STATE_IDLE = "IDLE"
STATE_WORKING = "WORKING"
STATE_ON_BREAK = "ON_BREAK"
STATE_CHECKED_OUT = "CHECKED_OUT"

EVENT_CHECK_IN = "CHECK_IN"
EVENT_BREAK_OUT = "BREAK_OUT"
EVENT_BREAK_IN = "BREAK_IN"
EVENT_CHECK_OUT = "CHECK_OUT"

ALLOWED_TRANSITIONS = {
    (STATE_IDLE, EVENT_CHECK_IN): STATE_WORKING,
    (STATE_WORKING, EVENT_BREAK_OUT): STATE_ON_BREAK,
    (STATE_ON_BREAK, EVENT_BREAK_IN): STATE_WORKING,
    (STATE_WORKING, EVENT_CHECK_OUT): STATE_CHECKED_OUT,
}

# -------------------------------
# Helpers
# -------------------------------
def _redis_key(company_id, emp_id, day):
    return f"attendance:{company_id}:{emp_id}:{day}"

# -------------------------------
# FSM Core
# -------------------------------
def process_event(company_id, emp_id, event_type, event_ts=None):
    """
    Redis-backed FSM transition.
    Returns True if transition accepted, False otherwise.
    """
    if event_ts is None:
        event_ts = datetime.now()

    today = date.today().isoformat()
    key = _redis_key(company_id, emp_id, today)

    pipe = redis_client.pipeline()

    # Fetch current state
    data = redis_client.hgetall(key)

    state = data.get("state", STATE_IDLE)
    last_ts = float(data.get("last_state_change_ts", event_ts.timestamp()))
    work_seconds = float(data.get("work_seconds", 0))
    break_seconds = float(data.get("break_seconds", 0))

    transition_key = (state, event_type)

    if transition_key not in ALLOWED_TRANSITIONS:
        # Invalid transition → ignore safely
        return False

    now_ts = event_ts.timestamp()
    delta = now_ts - last_ts

    # Accumulate time
    if state == STATE_WORKING:
        work_seconds += delta
    elif state == STATE_ON_BREAK:
        break_seconds += delta

    next_state = ALLOWED_TRANSITIONS[transition_key]

    pipe.hset(key, mapping={
        "state": next_state,
        "last_event_ts": now_ts,
        "last_state_change_ts": now_ts,
        "work_seconds": int(work_seconds),
        "break_seconds": int(break_seconds)
    })

    # Expire after 2 days (safety)
    pipe.expire(key, 172800)

    pipe.execute()
    return True
