import logging
from utils.posgres_redis import redis_client

logger = logging.getLogger("AttendanceFSM")

def get_last_state(emp_id, company_id):
    key = f"fsm:{company_id}:{emp_id}"
    state = redis_client.get(key)
    if state is None:
        logger.info(f"[REDIS MISS] {key}")
    else:
        logger.info(f"[REDIS READ] {key} = {state}")
    return state

def set_last_state(emp_id, company_id, new_state):
    key = f"fsm:{company_id}:{emp_id}"
    redis_client.setex(key, 86400, new_state)
    logger.info(f"[REDIS WRITE] {key} → {new_state} (TTL=86400)")

def process_event(emp_id, company_id, detected_event, camera_name=None, image_url=None):
    detected_event = detected_event.upper()
    logger.info(f"[FSM EVENT] emp={emp_id} company={company_id} event={detected_event}")

    last_state = get_last_state(emp_id, company_id)
    new_state = detected_event

    # Always update FSM state (audit), do not block logging
    set_last_state(emp_id, company_id, new_state)
    logger.info(f"[FSM AUDIT] emp={emp_id} {last_state} → {new_state}")
    return True  # Always allow logging

# def process_event(emp_id, company_id, detected_event, camera_name=None, image_url=None):
#     detected_event = detected_event.upper()
#     logger.info(f"[FSM EVENT] emp={emp_id} company={company_id} event={detected_event}")

#     last_state = get_last_state(emp_id, company_id)

#     # First ever event
#     if last_state is None:
#         if detected_event == "CHECK_IN":
#             set_last_state(emp_id, company_id, detected_event)
#             return True
#         else:
#             logger.warning(f"[FSM BLOCKED] Cannot CHECK_OUT before CHECK_IN")
#             return False

#     # Prevent duplicate CHECK_IN
#     if last_state == "CHECK_IN" and detected_event == "CHECK_IN":
#         logger.warning(f"[FSM BLOCKED] Duplicate CHECK_IN")
#         return False

#     # Prevent duplicate CHECK_OUT
#     if last_state == "CHECK_OUT" and detected_event == "CHECK_OUT":
#         logger.warning(f"[FSM BLOCKED] Duplicate CHECK_OUT")
#         return False

#     # Valid transitions
#     if (last_state == "CHECK_IN" and detected_event == "CHECK_OUT") or \
#        (last_state == "CHECK_OUT" and detected_event == "CHECK_IN"):

#         set_last_state(emp_id, company_id, detected_event)
#         logger.info(f"[FSM TRANSITION] {last_state} → {detected_event}")
#         return True

#     return False
