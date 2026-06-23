import os
import sys
import numpy as np
import traceback
from datetime import datetime, date
import logging
from .posgres import get_user_info, log_attendance

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
# A dictionary to store state for each camera, keyed by source_id.
# This replaces the global daily_in_count and daily_out_count variables.
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
RECOGNITION_THRESHOLD = 0.2

def draw_detection_area(frame_meta, batch_meta):
    """Draw the detection area on the frame"""
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
        logger.error(f"Error drawing detection area: {str(e)}")
        traceback.print_exc()

def draw_detection_line(frame_meta, batch_meta):
    """Draw the detection line on the frame."""
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
        logger.error(f"Error drawing detection line: {str(e)}")
        traceback.print_exc()

def is_in_detection_area(obj_meta):
    """Check if object is within the detection area"""
    try:
        if not obj_meta or not obj_meta.rect_params:
            logger.warning("Object metadata or rect_params are invalid for area check.")
            return False

        center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
        center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2

        return (DETECTION_AREA['x1'] <= center_x <= DETECTION_AREA['x2'] and
                DETECTION_AREA['y1'] <= center_y <= DETECTION_AREA['y2'])
    except Exception as e:
        logger.error(f"Error checking detection area for object {obj_meta.object_id if obj_meta else 'N/A'}: {str(e)}")
        traceback.print_exc()
        return False

def check_detection_area(obj_meta, name):
    """
    Check if object is currently in the detection area AND if enough time has passed
    since its last attendance log to prevent re-logging too frequently.
    This function also updates the tracking state of the object.
    Returns True if attendance should be logged, False otherwise.
    """
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
    """
    Checks if an object has crossed the defined line and logs 'in' or 'out' movement.
    It also displays the direction on the frame.
    Returns 'IN' or 'OUT' if a new, logged crossing occurred, otherwise None.
    """
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
            crossed = True
            direction = "IN"
        elif prev_y > line_y and obj_bottom_center_y <= line_y:
            crossed = True
            direction = "OUT"

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
            else:
                pass

        return None
    except Exception as e:
        logger.error(f"Error checking line crossing for object {obj_id}: {str(e)}")
        traceback.print_exc()
        return None

def display_name_on_frame(obj_meta, frame_meta, batch_meta, name):
    """Display the name on the frame"""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            logger.warning("Failed to acquire display meta.")
            return False

        if display_meta.num_labels >= MAX_DISPLAY_LABELS:
            pyds.nvds_release_display_meta(display_meta)
            logger.warning("Exceeded max labels for display meta. Skipping adding new label.")
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
        logger.error(f"Error displaying name on frame for object {obj_meta.object_id if obj_meta else 'N/A'}: {str(e)}")
        traceback.print_exc()
        return False

def display_global_status(frame_meta, batch_meta, status):
    """
    Displays the current global status (e.g., "IN", "OUT", "Standby") on the frame.
    """
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            logger.warning("Failed to acquire display meta for global status.")
            return

        if display_meta.num_labels >= MAX_DISPLAY_LABELS:
            pyds.nvds_release_display_meta(display_meta)
            logger.warning("Exceeded max labels for display meta. Skipping global status display.")
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
        logger.error(f"Error displaying global status: {str(e)}")
        traceback.print_exc()

def display_daily_counts(frame_meta, batch_meta, in_count, out_count, camera_type):
    """
    Displays the daily IN/OUT counts on the frame, based on camera type.
    """
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            logger.warning("Failed to acquire display meta for daily counts.")
            return

        # Display counts based on camera type
        if camera_type == 'entrance':
            if display_meta.num_labels + 1 > MAX_DISPLAY_LABELS:
                pyds.nvds_release_display_meta(display_meta)
                logger.warning("Exceeded max labels for display meta. Skipping IN count display.")
                return

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
                pyds.nvds_release_display_meta(display_meta)
                logger.warning("Exceeded max labels for display meta. Skipping OUT count display.")
                return

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
        else: # Default case, display both counts
            if display_meta.num_labels + 2 > MAX_DISPLAY_LABELS:
                pyds.nvds_release_display_meta(display_meta)
                logger.warning("Exceeded max labels for display meta. Skipping daily counts display.")
                return

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
        logger.error(f"Error displaying daily counts: {str(e)}")
        traceback.print_exc()

def pgie_src_filter_probe(pad, info, u_data):
    """
    Filter face detections after primary inference based on confidence.
    Removes objects (presumably faces) with confidence below a threshold.
    """
    try:
        if not pad or not info:
            logger.error("Invalid probe parameters in pgie_src_filter_probe.")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("Unable to get GstBuffer in pgie_src_filter_probe.")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
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
                    except Exception as e:
                        logger.error(f"Error processing object meta in pgie_src_filter_probe: {str(e)}")
                        traceback.print_exc()
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None

                for obj_to_remove in objects_to_remove:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_to_remove)

            except Exception as e:
                logger.error(f"Error processing frame meta in pgie_src_filter_probe: {str(e)}")
                traceback.print_exc()
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in pgie_src_filter_probe: {str(e)}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK


def sgie_feature_extract_probe(pad, info, data):
    """
    Extract face features from SGIE output and match against known faces.
    Handles attendance logging, name display, and daily counts.
    """
    global CAMERA_STATE

    try:
        if not pad or not info or not data:
            logger.error("Invalid probe parameters in sgie_feature_extract_probe.")
            return Gst.PadProbeReturn.OK

        loaded_faces = data[0] if len(data) > 0 else None
        feature_save_dir = data[2]
        sources = data[3] if len(data) > 3 else []

        if not loaded_faces:
            logger.warning("No loaded faces provided for feature extraction and matching. This means no recognition can occur.")
            pass

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("Unable to get GstBuffer in sgie_feature_extract_probe.")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
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

                # Initialize camera state if it doesn't exist
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

                # --- Per-camera Daily Count Reset Logic ---
                current_date = date.today()
                if current_date != camera_status['last_reset_date']:
                    logger.info(f"New day detected for camera {source_id} ({current_date}). Resetting daily counts.")
                    camera_status['in_count'] = 0
                    camera_status['out_count'] = 0
                    camera_status['last_reset_date'] = current_date
                    camera_status['last_global_status'] = {'status': 'Standby', 'timestamp': current_timestamp}

                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, feature_save_dir)
                        display_name_to_use = "Unknown"

                        line_cross_direction = check_line_crossing(obj_meta, obj_meta.object_id, current_timestamp, frame_meta, batch_meta, display_name_to_use)

                        if face_feature is not None and loaded_faces:
                            best_match_id, best_score = match_faces(face_feature, loaded_faces)

                            if best_match_id and best_score >= RECOGNITION_THRESHOLD:
                                display_name_to_use = f"ID:{best_match_id} (Recognized)"

                                if is_in_detection_area(obj_meta):
                                    logger.info(f"Recognized person {best_match_id} is in the detection area. Logging attendance...")

                                    int_best_match_id = None
                                    try:
                                        int_best_match_id = int(str(best_match_id))
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"Error converting best_match_id '{best_match_id}' to integer: {e}. Skipping user info lookup.")

                                    if int_best_match_id is not None:
                                        first_name, last_name, emp_id_from_db = get_user_info(int_best_match_id)

                                        if first_name or last_name or emp_id_from_db:
                                            display_name_to_use = f"{first_name} {last_name} (Emp ID: {emp_id_from_db})"

                                            check_type = None
                                            if camera_type == 'entrance':
                                                check_type = 'in'
                                            elif camera_type == 'exit':
                                                check_type = 'out'

                                            if check_type:
                                                log_attendance(
                                                    emp_id=int_best_match_id,
                                                    first_name=first_name,
                                                    last_name=last_name,
                                                    Attendance_date=current_date.strftime("%d-%m-%Y"),
                                                    Attendance_time=current_timestamp.strftime("%H:%M:%S"),
                                                    check_type=check_type,
                                                    camera_name=camera_name
                                                )
                                                logger.info(f"Recognized: ID {int_best_match_id} ({display_name_to_use}) from {camera_name} ({camera_type}). Attendance logged as '{check_type}'.")

                                                output_dir = "/workspace/face_final/output_faces"
                                                os.makedirs(output_dir, exist_ok=True)
                                                crop_filename = f"face_{int_best_match_id}_{current_timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                                # save_face_crop(gst_buffer, frame_meta, obj_meta, output_dir, crop_filename)
                                        else:
                                            logger.warning(f"Recognized face with ID {best_match_id}, but user details not found.")
                                            display_name_to_use = f"ID:{best_match_id} (No Info)"
                                    else:
                                        display_name_to_use = f"ID:{best_match_id} (Invalid Format)"
                                else:
                                    logger.info(f"Recognized person {best_match_id} is not in the detection area. Not logging attendance.")
                            # else:
                            #     if best_match_id:
                            #         display_name_to_use = f"Unknown (ID:{best_match_id}, {best_score:.2f})"
                            #     else:
                            #         display_name_to_use = "Unknown"
                            #     display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name_to_use)
                        elif face_feature is None:
                            logger.debug(f"No face feature extracted for object {obj_meta.object_id}.")
                            display_name_to_use = "No Feature"
                        else:
                            logger.debug("No known faces to match against.")
                            display_name_to_use = "No Known Faces"

                        display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name_to_use)

                        if line_cross_direction == "IN" and camera_type == 'entrance':
                            camera_status['in_count'] += 1
                            camera_status['last_global_status'] = {'status': 'IN', 'timestamp': current_timestamp}
                            logger.info(f"CAMERA {source_id} IN INCREMENTED: {camera_status['in_count']}")
                        elif line_cross_direction == "OUT" and camera_type == 'exit':
                            camera_status['out_count'] += 1
                            camera_status['last_global_status'] = {'status': 'OUT', 'timestamp': current_timestamp}
                            logger.info(f"CAMERA {source_id} OUT INCREMENTED: {camera_status['out_count']}")

                    except Exception as e:
                        logger.error(f"Error processing object meta in sgie_feature_extract_probe: {str(e)}")
                        traceback.print_exc()
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                logger.error(f"Error processing frame meta in sgie_feature_extract_probe: {str(e)}")
                traceback.print_exc()
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None

            time_since_last_global_cross = (current_timestamp - camera_status['last_global_status']['timestamp']).total_seconds()
            if time_since_last_global_cross > GLOBAL_STATUS_DISPLAY_COOLDOWN_SECONDS:
                camera_status['last_global_status']['status'] = 'Standby'

            display_global_status(frame_meta, batch_meta, camera_status['last_global_status']['status'])
            display_daily_counts(frame_meta, batch_meta, camera_status['in_count'], camera_status['out_count'], camera_type)

    except Exception as e:
        logger.critical(f"CRITICAL ERROR in sgie_feature_extract_probe: {str(e)}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK

def match_faces(face_feature, loaded_faces):
    """
    Match a single face feature against known faces and return the best matching ID and similarity score.
    Assumes face_feature and known_feature are L2-normalized.
    """
    best_match_id = None
    best_score = -1.0

    try:
        face_feature_flat = face_feature.flatten()

        for user_id, known_feature in loaded_faces.items():
            if known_feature is None:
                continue

            known_feature_flat = known_feature.flatten()

            if face_feature_flat.shape != known_feature_flat.shape:
                logger.warning(f"Feature dimension mismatch for user {user_id}. Skipping match.")
                continue

            score = np.dot(face_feature_flat, known_feature_flat)

            if score > best_score:
                best_score = score
                best_match_id = user_id

    except Exception as e:
        logger.error(f"CRITICAL ERROR in match_faces: {str(e)}")
        traceback.print_exc()

    return best_match_id, best_score

def get_face_feature(obj_meta, frame_num, feature_save_dir=None):
    """
    Extract face features (embedding) from SGIE's tensor output.
    """
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
                                logger.warning(f"Failed to get detection at index {i}. Buffer might be smaller than expected. Expected size: {FEATURE_VECTOR_SIZE}")
                                output_list = []
                                break
                            output_list.append(val)
                        except Exception as e:
                            logger.error(f"ERROR getting detection {i} from tensor buffer: {str(e)}")
                            traceback.print_exc()
                            output_list = []
                            break

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
                                    logger.debug(f"Saved feature for obj {obj_meta.object_id} to {save_path}")
                                except Exception as e:
                                    logger.error(f"ERROR saving feature for object {obj_meta.object_id}: {str(e)}")
                                    traceback.print_exc()
                            return normal_array
                        else:
                            logger.warning(f"Feature vector for object {obj_meta.object_id} has zero norm. Cannot normalize.")
                    else:
                        logger.warning(f"Extracted feature vector size {len(output_list)} does not match expected size {FEATURE_VECTOR_SIZE}. Feature not returned.")
            except Exception as e:
                logger.error(f"Error processing user meta in get_face_feature for object {obj_meta.object_id}: {str(e)}")
                traceback.print_exc()
            finally:
                l_user_meta = l_user_meta.next if hasattr(l_user_meta, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in get_face_feature: {str(e)}")
        traceback.print_exc()
    return None
