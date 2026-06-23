import threading
import logging
from ..posgres_service import log_attendance

logger = logging.getLogger("ATT_WORKER")

def attendance_worker(q):
    logger.info(
        f"ATT WORKER STARTED | thread={threading.current_thread().name} | queue_id={id(q)}"
    )

    while True:
        task = q.get()
        logger.info(f"ATT WORKER GOT TASK: {task}")

        if task is None:
            logger.info("ATT WORKER EXIT")
            break

        try:
            log_attendance(**task)
            logger.info("ATT WORKER DB WRITE OK")
        except Exception:
            logger.exception("ATT WORKER DB WRITE FAILED")
