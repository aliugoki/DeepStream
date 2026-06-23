import logging
logger = logging.getLogger("ATT_WORKER")
import multiprocessing
attendance_q = multiprocessing.Queue()

def attendance_worker(q):
    """Background worker for DB writes."""
    logger.info(f"Worker {id(q)} online")
    while True:
        task = q.get()
        if task is None:
            break
        try:
            from ..posgres_service import log_attendance  # if posgres_service is under utils/
            log_attendance(**task)
        except Exception as e:
            logger.error(f"DB write error: {e}")

