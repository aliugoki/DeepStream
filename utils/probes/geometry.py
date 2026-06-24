# utils/probe/geometry.py

from .constants import DETECTION_AREA, DETECTION_LINE
from .state import TRACKED_OBJECTS
from datetime import datetime

def is_in_area(obj_meta):
    r = obj_meta.rect_params
    cx = r.left + r.width / 2
    cy = r.top + r.height / 2
    return (DETECTION_AREA['x1'] <= cx <= DETECTION_AREA['x2'] and
            DETECTION_AREA['y1'] <= cy <= DETECTION_AREA['y2'])

def process_line_crossing(src_id, obj_meta, now):
    bottom_y = obj_meta.rect_params.top + obj_meta.rect_params.height
    line_y = DETECTION_LINE['y1']

    obj_id = obj_meta.object_id
    if obj_id not in TRACKED_OBJECTS:
        TRACKED_OBJECTS[obj_id] = {'last_y': bottom_y, 'last_cross_ts': None}

    prev_y = TRACKED_OBJECTS[obj_id]['last_y']
    TRACKED_OBJECTS[obj_id]['last_y'] = bottom_y

    direction = None
    if prev_y < line_y <= bottom_y:
        direction = "IN"
    elif prev_y > line_y >= bottom_y:
        direction = "OUT"

    last_t = TRACKED_OBJECTS[obj_id]['last_cross_ts']
    if direction and (last_t is None or (now - last_t).total_seconds() > 10):
        TRACKED_OBJECTS[obj_id]['last_cross_ts'] = now
        return direction
    return None
