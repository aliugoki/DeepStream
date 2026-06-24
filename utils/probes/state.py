# utils/probe/state.py

import multiprocessing

CAMERA_STATE = {}
TRACKED_OBJECTS = {}
FACE_SAVE_STATE = {}
FACE_SAVE_LOCK = multiprocessing.Lock()

_LAST_FACE_CLEANUP_TS = 0


# from datetime import date, datetime
# import time
# import multiprocessing
# import logging

# logger = logging.getLogger(__name__)

# FACE_SAVE_STATE = {}
# FACE_SAVE_LOCK = multiprocessing.Lock()

# RETENTION_DAYS = 2
# CLEAN_INTERVAL = 300
# _last_cleanup = 0

# def cleanup():
#     global _last_cleanup
#     now = time.time()
#     if now - _last_cleanup < CLEAN_INTERVAL:
#         return

#     cutoff = date.today().toordinal() - RETENTION_DAYS

#     with FACE_SAVE_LOCK:
#         keys = [k for k in FACE_SAVE_STATE if k[2].toordinal() < cutoff]
#         for k in keys:
#             del FACE_SAVE_STATE[k]

#     _last_cleanup = now


# def should_save(source_id, emp_id):
#     today = date.today()
#     key = (source_id, emp_id, today)

#     with FACE_SAVE_LOCK:
#         if key in FACE_SAVE_STATE:
#             return False

#         FACE_SAVE_STATE[key] = datetime.now()
#         return True
