# utils/probe.py
import os
import sys
import numpy as np
import traceback
from datetime import datetime, date
import logging
import multiprocessing
import time
import psycopg2

# ✅ Use your actual PostgreSQL functions
from . import posgres_service




# NOTE: The posgres module is assumed to exist and contain the following functions
# get_user_info(emp_id, company_id)
# log_attendance(...)
# COMPANY_ID
# To run this code in production, replace the MockPosgres with your real module.
# class MockPosgres:
#     def get_user_info(self, emp_id, company_id):
#         return "John", "Doe", "EMP123", "/path/to/image.jpg"
#     def log_attendance(self, **kwargs):
#         print(f"[{os.getpid()}] ⏳ Logging attendance for {kwargs.get('first_name')} {kwargs.get('last_name')}...")
#         time.sleep(1) # Simulate a blocking database write
#         print(f"[{os.getpid()}] ✅ Successfully logged attendance for {kwargs.get('first_name')}.")
#     COMPANY_ID = "MyCompany"

# # Replace this with: from posgres import get_user_info, log_attendance, COMPANY_ID
# posgres = MockPosgres()
get_user_info = posgres_service.get_user_info
log_attendance = posgres_service.log_attendance
COMPANY_ID = posgres_service.COMPANY_ID

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GStreamer imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except Exception as e:
    logger.error(f"Failed to import GStreamer: {str(e)}")
    sys.exit(1)

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)

try:
    import pyds
except Exception as e:
    logger.error(f"Failed to import pyds (DeepStream Python Bindings): {str(e)}")
    sys.exit(1)

# --- Global State for Per-Camera Tracking ---
CAMERA_STATE = {}
GLOBAL_STATUS_DISPLAY_COOLDOWN_SECONDS = 5

# Detection area configuration
DETECTION_AREA = {
    'x1': 0,
    'y1': 250,
    'x2': 2500,
    'y2': 2000,
}
AREA_COLOR = (0.0, 0.0, 0.0, 0.0)
LINE_COLOR = (0.0, 1.0, 0.0, 0.8)

# Line Crossing Detection Configuration
DETECTION_LINE = {
    'x1': 0,
    'y1': 250,
    'x2': 1920,
    'y2': 250
}
LINE_CROSS_COLOR = (1.0, 0.0, 0.0, 0.8)

# --- IMPORTANT: Maximum display elements per NvDsDisplayMeta ---
MAX_DISPLAY_LABELS = 16
MAX_DISPLAY_LINES = 16
MAX_DISPLAY_RECTS = 16
MAX_DISPLAY_POLYGONS = 16

TRACKED_OBJECTS = {}
ATTENDANCE_COOLDOWN_SECONDS = 10
RECOGNITION_THRESHOLD = 0.5

# --- Attendance Worker Function ---
def attendance_worker(q):
    """
    Runs in a separate process. Consumes tasks from the queue and calls the blocking log_attendance function.
    """
    logger.info(f"[{os.getpid()}] Attendance worker started, waiting for tasks...")
    while True:
        try:
            task = q.get()
            if task is None:
                logger.info(f"[{os.getpid()}] Shutdown signal received. Worker exiting.")
                break
            # Call the blocking DB write (replace with your real implementation)
            log_attendance(**task)
        except Exception as e:
            logger.error(f"[{os.getpid()}] Error processing task: {e}")
            traceback.print_exc()

# ---------------- Drawing helpers ----------------
def draw_detection_area(frame_meta, batch_meta):
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta:
            if display_meta.num_rects >= MAX_DISPLAY_RECTS:
                logger.warning("Max rectangles reached for display meta. Skipping drawing detection area.")
                pyds.nvds_release_display_meta(display_meta)
                return

            display_meta.num_rects = 1
            rect = display_meta.rect_params[0]
            rect.left = DETECTION_AREA['x1']
            rect.top = DETECTION_AREA['y1']
            rect.width = DETECTION_AREA['x2'] - DETECTION_AREA['x1']
            rect.height = DETECTION_AREA['y2'] - DETECTION_AREA['y1']
            rect.border_width = 2
            rect.border_color.set(*LINE_COLOR)
            rect.has_bg_color = 1
            rect.bg_color.set(*AREA_COLOR)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error drawing detection area: {e}")
        traceback.print_exc()

def draw_detection_line(frame_meta, batch_meta):
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta:
            if display_meta.num_lines >= MAX_DISPLAY_LINES:
                logger.warning("Max lines reached for display meta. Skipping drawing detection line.")
                pyds.nvds_release_display_meta(display_meta)
                return

            line_params = display_meta.line_params[display_meta.num_lines]
            line_params.x1 = DETECTION_LINE['x1']
            line_params.y1 = DETECTION_LINE['y1']
            line_params.x2 = DETECTION_LINE['x2']
            line_params.y2 = DETECTION_LINE['y2']
            line_params.line_width = 3
            line_params.line_color.set(*LINE_CROSS_COLOR)
            display_meta.num_lines += 1

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error drawing detection line: {e}")
        traceback.print_exc()

# --------------- Area and crossing helpers ---------------
def is_in_detection_area(obj_meta):
    try:
        if not obj_meta or not obj_meta.rect_params:
            return False
        center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
        center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
        return (DETECTION_AREA['x1'] <= center_x <= DETECTION_AREA['x2'] and
                DETECTION_AREA['y1'] <= center_y <= DETECTION_AREA['y2'])
    except Exception:
        return False

def check_detection_area(obj_meta, name):
    obj_id = obj_meta.object_id
    current_timestamp = datetime.now()
    current_in_area = is_in_detection_area(obj_meta)

    if obj_id not in TRACKED_OBJECTS:
        TRACKED_OBJECTS[obj_id] = {
            'in_area': False,
            'name': None,
            'last_logged_time': None,
            'last_y_position': obj_meta.rect_params.top + obj_meta.rect_params.height,
            'last_cross_logged_time': None
        }

    prev_data = TRACKED_OBJECTS[obj_id]
    TRACKED_OBJECTS[obj_id]['in_area'] = current_in_area
    TRACKED_OBJECTS[obj_id]['name'] = name

    if current_in_area:
        if prev_data['last_logged_time'] is None or \
           (current_timestamp - prev_data['last_logged_time']).total_seconds() > ATTENDANCE_COOLDOWN_SECONDS:
            TRACKED_OBJECTS[obj_id]['last_logged_time'] = current_timestamp
            return True
    return False

def check_line_crossing(obj_meta, obj_id, current_timestamp, frame_meta, batch_meta, display_name="Unknown"):
    try:
        obj_bottom_center_y = obj_meta.rect_params.top + obj_meta.rect_params.height
        line_y = DETECTION_LINE['y1']

        if obj_id not in TRACKED_OBJECTS:
            TRACKED_OBJECTS[obj_id] = {
                'in_area': False,
                'name': None,
                'last_logged_time': None,
                'last_y_position': obj_bottom_center_y,
                'last_cross_logged_time': None,
                'last_displayed_direction': None
            }

        prev_y = TRACKED_OBJECTS[obj_id]['last_y_position']
        TRACKED_OBJECTS[obj_id]['last_y_position'] = obj_bottom_center_y

        crossed = False
        direction = None
        if prev_y < line_y and obj_bottom_center_y >= line_y:
            crossed = True; direction = "IN"
        elif prev_y > line_y and obj_bottom_center_y <= line_y:
            crossed = True; direction = "OUT"

        if crossed:
            prev_cross_time = TRACKED_OBJECTS[obj_id].get('last_cross_logged_time')
            if prev_cross_time is None or \
               (current_timestamp - prev_cross_time).total_seconds() > ATTENDANCE_COOLDOWN_SECONDS:
                logger.info(f"Object {obj_id} ({display_name}) crossed line: {direction}")
                TRACKED_OBJECTS[obj_id]['last_cross_logged_time'] = current_timestamp
                TRACKED_OBJECTS[obj_id]['last_displayed_direction'] = direction

                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                if display_meta and display_meta.num_labels < MAX_DISPLAY_LABELS:
                    text_params = display_meta.text_params[display_meta.num_labels]
                    text_params.display_text = f"{display_name} ({direction})"
                    text_params.x_offset = int(obj_meta.rect_params.left)
                    text_params.y_offset = int(max(0, obj_meta.rect_params.top - 40))
                    text_params.font_params.font_name = "Serif"
                    text_params.font_params.font_size = 15
                    text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    text_params.set_bg_clr = 1
                    text_params.text_bg_clr.set(0.8, 0.2, 0.2, 0.6)
                    display_meta.num_labels += 1
                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                else:
                    if display_meta: pyds.nvds_release_display_meta(display_meta)
                    logger.warning(f"Max labels reached for frame {frame_meta.frame_num}. Cannot display direction for {obj_id}.")
                return direction
        return None
    except Exception as e:
        logger.error(f"Error checking line crossing for object {obj_id}: {e}")
        traceback.print_exc()
        return None

# --------------- Display helpers ---------------
def display_name_on_frame(obj_meta, frame_meta, batch_meta, name):
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return False
        if display_meta.num_labels >= MAX_DISPLAY_LABELS:
            pyds.nvds_release_display_meta(display_meta)
            return False
        text_params = display_meta.text_params[display_meta.num_labels]
        text_params.display_text = name
        text_params.x_offset = int(obj_meta.rect_params.left)
        text_params.y_offset = int(max(0, obj_meta.rect_params.top - 20))
        text_params.font_params.font_name = "Serif"
        text_params.font_params.font_size = 15
        text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
        display_meta.num_labels += 1
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        return True
    except Exception as e:
        logger.error(f"Error displaying name on frame: {e}")
        traceback.print_exc()
        return False

def display_global_status(frame_meta, batch_meta, status):
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return
        if display_meta.num_labels >= MAX_DISPLAY_LABELS:
            pyds.nvds_release_display_meta(display_meta)
            return
        text_params = display_meta.text_params[display_meta.num_labels]
        text_params.display_text = f"GLOBAL STATUS: {status}"
        text_params.x_offset = 10
        text_params.y_offset = 10
        text_params.font_params.font_name = "Sans"
        text_params.font_params.font_size = 18
        text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.1, 0.1, 0.1, 0.7)
        display_meta.num_labels += 1
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error displaying global status: {e}")
        traceback.print_exc()

def display_daily_counts(frame_meta, batch_meta, in_count, out_count, camera_type):
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return

        if camera_type == 'entrance':
            if display_meta.num_labels + 1 > MAX_DISPLAY_LABELS:
                pyds.nvds_release_display_meta(display_meta); return
            text_params_in = display_meta.text_params[display_meta.num_labels]
            text_params_in.display_text = f"IN: {in_count}"
            text_params_in.x_offset = 10
            text_params_in.y_offset = 50
            text_params_in.font_params.font_name = "Sans"
            text_params_in.font_params.font_size = 18
            text_params_in.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            text_params_in.set_bg_clr = 1
            text_params_in.text_bg_clr.set(0.1, 0.1, 0.1, 0.7)
            display_meta.num_labels += 1
        elif camera_type == 'exit':
            if display_meta.num_labels + 1 > MAX_DISPLAY_LABELS:
                pyds.nvds_release_display_meta(display_meta); return
            text_params_out = display_meta.text_params[display_meta.num_labels]
            text_params_out.display_text = f"OUT: {out_count}"
            text_params_out.x_offset = 10
            text_params_out.y_offset = 50
            text_params_out.font_params.font_name = "Sans"
            text_params_out.font_params.font_size = 18
            text_params_out.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
            text_params_out.set_bg_clr = 1
            text_params_out.text_bg_clr.set(0.1, 0.1, 0.1, 0.7)
            display_meta.num_labels += 1
        else:
            if display_meta.num_labels + 2 > MAX_DISPLAY_LABELS:
                pyds.nvds_release_display_meta(display_meta); return
            text_params_in = display_meta.text_params[display_meta.num_labels]
            text_params_in.display_text = f"IN: {in_count}"
            text_params_in.x_offset = 10
            text_params_in.y_offset = 50
            text_params_in.font_params.font_name = "Sans"
            text_params_in.font_params.font_size = 18
            text_params_in.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            text_params_in.set_bg_clr = 1
            text_params_in.text_bg_clr.set(0.1, 0.1, 0.1, 0.7)
            display_meta.num_labels += 1

            text_params_out = display_meta.text_params[display_meta.num_labels]
            text_params_out.display_text = f"OUT: {out_count}"
            text_params_out.x_offset = 10
            text_params_out.y_offset = 90
            text_params_out.font_params.font_name = "Sans"
            text_params_out.font_params.font_size = 18
            text_params_out.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
            text_params_out.set_bg_clr = 1
            text_params_out.text_bg_clr.set(0.1, 0.1, 0.1, 0.7)
            display_meta.num_labels += 1

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error displaying daily counts: {e}")
        traceback.print_exc()

# --------------- PGIE probe (filter by confidence) ---------------
def pgie_src_filter_probe(pad, info, u_data):
    try:
        if not pad or not info:
            logger.error("Invalid probe parameters in pgie_src_filter_probe.")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if not frame_meta:
                break
            l_obj = frame_meta.obj_meta_list
            objects_to_remove = []
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    if obj_meta and hasattr(obj_meta, 'confidence'):
                        PGIE_CONFIDENCE_THRESHOLD = 0.6
                        if obj_meta.confidence <= PGIE_CONFIDENCE_THRESHOLD:
                            objects_to_remove.append(obj_meta)
                except Exception:
                    traceback.print_exc()
                finally:
                    l_obj = l_obj.next if hasattr(l_obj, 'next') else None

            for o in objects_to_remove:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, o)

            l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in pgie_src_filter_probe: {e}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK

# -----------------------------
# PROBE: SGIE feature extraction
# -----------------------------
def sgie_feature_extract_probe(pad, info, user_data):
    """
    Extract face features from SGIE output and match against known faces.
    user_data must be a dict with keys:
      - 'loaded_faces' (dict)
      - 'attendance_queue' (multiprocessing.Queue)
      - 'feature_save_dir' (str or None)
      - 'sources' (list)
    """
    global CAMERA_STATE
    try:
        if not pad or not info or not user_data:
            logger.error("Invalid probe parameters in sgie_feature_extract_probe.")
            return Gst.PadProbeReturn.OK

        if not isinstance(user_data, dict):
            logger.error(f"user_data is not a dict (got {type(user_data)}). Probe will no-op.")
            return Gst.PadProbeReturn.OK

        loaded_faces = user_data.get('loaded_faces') or {}
        attendance_queue = user_data.get('attendance_queue', None)
        feature_save_dir = user_data.get('feature_save_dir', None)
        sources = user_data.get('sources', [])

        if not hasattr(attendance_queue, 'put'):
            logger.critical("Attendance queue missing or invalid. Cannot log attendance.")
            return Gst.PadProbeReturn.OK

        if not isinstance(loaded_faces, dict) or len(loaded_faces) == 0:
            # Log only once per probe invocation; avoid per-frame spam (handled upstream).
            logger.warning("No loaded faces provided for feature extraction and matching. Recognition will be skipped.")

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if not frame_meta:
                break

            source_id = frame_meta.source_id
            camera_name = f"Camera {source_id}"
            camera_type = "unknown"
            if 0 <= source_id < len(sources):
                source_info = sources[source_id]
                camera_name = source_info.get("id", camera_name)
                camera_type = source_info.get("type", camera_type)

            if source_id not in CAMERA_STATE:
                CAMERA_STATE[source_id] = {
                    'in_count': 0,
                    'out_count': 0,
                    'last_reset_date': date.today(),
                    'last_global_status': {'status': 'Standby', 'timestamp': datetime.now()}
                }

            camera_status = CAMERA_STATE[source_id]

            draw_detection_area(frame_meta, batch_meta)
            draw_detection_line(frame_meta, batch_meta)

            current_timestamp = datetime.now()
            current_date = date.today()
            if current_date != camera_status['last_reset_date']:
                camera_status['in_count'] = 0
                camera_status['out_count'] = 0
                camera_status['last_reset_date'] = current_date
                camera_status['last_global_status'] = {'status': 'Standby', 'timestamp': current_timestamp}

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    if not obj_meta:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
                        continue

                    face_feature = get_face_feature(obj_meta, frame_meta.frame_num, feature_save_dir)
                    display_name_to_use = "Unknown"

                    line_cross_direction = check_line_crossing(
                        obj_meta, obj_meta.object_id, current_timestamp, frame_meta, batch_meta, display_name_to_use
                    )

                    if face_feature is not None and loaded_faces:
                        best_match_id, best_score = match_faces(face_feature, loaded_faces)

                        if best_match_id and best_score >= RECOGNITION_THRESHOLD:
                            display_name_to_use = f"ID:{best_match_id} (Recognized)"

                            if is_in_detection_area(obj_meta):
                                logger.info(f"Recognized person {best_match_id} in detection area. Preparing attendance log.")

                                int_best_match_id = None
                                try:
                                    int_best_match_id = str(best_match_id).strip()
                                except (ValueError, TypeError):
                                    logger.error(f"Error converting best_match_id '{best_match_id}' to int; skipping user lookup.")
                                if int_best_match_id is not None:
                                    first_name, last_name, emp_id_from_db, image_path = get_user_info(int_best_match_id, COMPANY_ID)
                                    if first_name or last_name or emp_id_from_db or image_path or COMPANY_ID:
                                        display_name_to_use = f"{first_name} {last_name} (Emp ID: {emp_id_from_db})"
                                        check_type = None
                                        if camera_type == 'entrance':
                                            check_type = 'in'
                                        elif camera_type == 'exit':
                                            check_type = 'out'

                                        if check_type:
                                            attendance_data = {
                                                "company_id": COMPANY_ID,
                                                "emp_id": int_best_match_id,
                                                "first_name": first_name,
                                                "last_name": last_name,
                                                "image_url": image_path,
                                                "attendance_date": current_date.strftime("%d-%m-%Y"),
                                                "attendance_time": current_timestamp.strftime("%H:%M:%S"),
                                                "check_type": check_type,
                                                "camera_name": camera_name
                                            }
                                            attendance_queue.put(attendance_data)
                                            logger.info(f"[{os.getpid()}] Submitted attendance task for {first_name} to queue.")
                                    else:
                                        display_name_to_use = f"ID:{best_match_id} (No Info)"
                                else:
                                    display_name_to_use = f"ID:{best_match_id} (Invalid Format)"
                            else:
                                logger.info(f"Recognized person {best_match_id} not in detection area. Not logging attendance.")
                    elif face_feature is None:
                        display_name_to_use = "No Feature"
                    else:
                        display_name_to_use = "No Known Faces"

                    display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name_to_use)

                    if line_cross_direction == "IN" and camera_type == 'entrance':
                        camera_status['in_count'] += 1
                        camera_status['last_global_status'] = {'status': 'IN', 'timestamp': current_timestamp}
                    elif line_cross_direction == "OUT" and camera_type == 'exit':
                        camera_status['out_count'] += 1
                        camera_status['last_global_status'] = {'status': 'OUT', 'timestamp': current_timestamp}

                except Exception as e:
                    logger.error(f"Error processing object meta in sgie_feature_extract_probe: {e}")
                    traceback.print_exc()
                finally:
                    l_obj = l_obj.next if hasattr(l_obj, 'next') else None

            time_since_last_global_cross = (current_timestamp - camera_status['last_global_status']['timestamp']).total_seconds()
            if time_since_last_global_cross > GLOBAL_STATUS_DISPLAY_COOLDOWN_SECONDS:
                camera_status['last_global_status']['status'] = 'Standby'

            display_global_status(frame_meta, batch_meta, camera_status['last_global_status']['status'])
            display_daily_counts(frame_meta, batch_meta, camera_status['in_count'], camera_status['out_count'], camera_type)

            l_frame = l_frame.next if hasattr(l_frame, 'next') else None

    except Exception as e:
        logger.critical(f"CRITICAL ERROR in sgie_feature_extract_probe: {e}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK

# --------------- Matching & feature helpers ---------------
def match_faces(face_feature, loaded_faces):
    best_match_id = None
    best_score = -1.0
    try:
        face_feature_flat = face_feature.flatten()
        for user_id, known_feature in loaded_faces.items():
            if known_feature is None:
                continue
            known_feature_flat = known_feature.flatten()
            if face_feature_flat.shape != known_feature_flat.shape:
                logger.warning(f"Feature dimension mismatch for user {user_id}. Skipping.")
                continue
            score = np.dot(face_feature_flat, known_feature_flat)
            if score > best_score:
                best_score = score
                best_match_id = user_id
    except Exception as e:
        logger.error(f"CRITICAL ERROR in match_faces: {e}")
        traceback.print_exc()
    return best_match_id, best_score

def get_face_feature(obj_meta, frame_num, feature_save_dir=None):
    try:
        if not obj_meta or not hasattr(obj_meta, 'obj_user_meta_list'):
            return None
        l_user_meta = obj_meta.obj_user_meta_list
        while l_user_meta:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                if not user_meta:
                    break
                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    if not tensor_meta:
                        break
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    if not layer or not layer.buffer:
                        logger.warning("Tensor layer or buffer not found for feature extraction.")
                        break

                    FEATURE_VECTOR_SIZE = 512
                    output_list = []
                    for i in range(FEATURE_VECTOR_SIZE):
                        try:
                            val = pyds.get_detections(layer.buffer, i)
                            if val is None:
                                logger.warning(f"Failed to get detection at index {i}.")
                                output_list = []; break
                            output_list.append(val)
                        except Exception as e:
                            logger.error(f"ERROR getting detection {i}: {e}")
                            traceback.print_exc()
                            output_list = []; break

                    if len(output_list) == FEATURE_VECTOR_SIZE:
                        res = np.array(output_list, dtype=np.float32).reshape((1, FEATURE_VECTOR_SIZE))
                        norm = np.linalg.norm(res)
                        if norm > 0:
                            normal_array = res / norm
                            if feature_save_dir:
                                try:
                                    os.makedirs(feature_save_dir, exist_ok=True)
                                    save_path = os.path.join(feature_save_dir, f"{obj_meta.object_id}-frame{frame_num}-{datetime.now().strftime('%H%M%S_%f')}.npy")
                                    np.save(save_path, normal_array)
                                except Exception as e:
                                    logger.error(f"ERROR saving feature: {e}")
                                    traceback.print_exc()
                            return normal_array
                        else:
                            logger.warning("Feature vector has zero norm.")
                    else:
                        logger.warning(f"Feature length mismatch: {len(output_list)} != {FEATURE_VECTOR_SIZE}")
            except Exception as e:
                logger.error(f"Error processing user meta in get_face_feature: {e}")
                traceback.print_exc()
            finally:
                l_user_meta = l_user_meta.next if hasattr(l_user_meta, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in get_face_feature: {e}")
        traceback.print_exc()
    return None

# ---------------------------------------
# Helper to attach the SGIE probe safely
# ---------------------------------------
def attach_sgie_probe(sgie, loaded_faces, attendance_queue, feature_save_dir, sources):
    """
    Attaches the SGIE src pad probe with a DICT user_data payload to prevent
    unpacking errors and ensure the queue is available inside the probe.
    """
    try:
        sgie_src_pad = sgie.get_static_pad("src")
        if sgie_src_pad:
            # ✅ Always pass the actual mutable dict (no `or {}`) so updates are seen live
            user_data = {
                "loaded_faces": loaded_faces,
                "attendance_queue": attendance_queue,
                "feature_save_dir": feature_save_dir,
                "sources": sources or []
            }
            sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, user_data)
            logger.info("SGIE src pad probe attached.")
        else:
            logger.error("Could not get sgie src pad for probe. SGIE probe not attached.")
    except Exception as e:
        logger.error(f"Failed to attach SGIE probe: {e}")
        traceback.print_exc()

