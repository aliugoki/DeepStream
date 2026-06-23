import os
import sys
import numpy as np
import traceback
from datetime import datetime, timedelta
#from .file import log_attendance
from .file import log_attendance
import time
# GStreamer imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except Exception as e:
    print(f"ERROR: Failed to import GStreamer: {str(e)}")
    sys.exit(1)

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)

import pyds
tracked_persons = {}
# Detection area configuration
DETECTION_AREA = {
    'x1': 0,  # Left boundary
    'y1': 150,  # Top boundary
    'x2': 2500,  # Right boundary
    'y2': 2000,  # Bottom boundary
}
AREA_COLOR = (0, 0, 0, 0)  # RGBA color
LINE_COLOR = (111, 0, 0, 0.8)  # RGBA color
TRACKED_OBJECTS = {}  # Track objects in detection area


def pgie_person_src_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        # Get camera source information
        sources = user_data.get('sources', {})

        # Proper iteration through frame_meta_list
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                if not frame_meta:
                    break

                # Draw detection area here - will be visible for all subsequent processing
                draw_detection_area(frame_meta, batch_meta)

                # Get source camera info
                source_id = frame_meta.source_id
                source_info = sources.get(f"source_{source_id}", {}).get("name", f"Camera {source_id}")

                # Rest of your existing person tracking code...
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        if obj_meta.class_id == 0:  # COCO "person" class
                            track_id = obj_meta.object_id
                            current_time = time.time()

                            # Initialize or update tracking
                            if track_id not in tracked_persons:
                                tracked_persons[track_id] = {
                                    "first_seen": current_time,
                                    "last_seen": current_time,
                                    "source": source_info
                                }
                            else:
                                tracked_persons[track_id]["last_seen"] = current_time

                            # Calculate dwell time
                            dwell_time = current_time - tracked_persons[track_id]["first_seen"]

                            # Display info with camera source
                            display_text = f"ID {track_id}: {dwell_time:.1f}s ({source_info})"
                            obj_meta.text_params.display_text = display_text

                    except Exception as e:
                        print(f"Error processing object meta: {str(e)}")
                    finally:
                        l_obj = l_obj.next

            except Exception as e:
                print(f"Error processing frame meta: {str(e)}")
            finally:
                l_frame = l_frame.next

    except Exception as e:
        print(f"Critical error in person probe: {str(e)}")

    return Gst.PadProbeReturn.OK

def draw_detection_area(frame_meta, batch_meta):
    """Draw the detection area on the frame"""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta:
            display_meta.num_rects = 1
            rect = display_meta.rect_params[0]
            rect.left = DETECTION_AREA['x1']
            rect.top = DETECTION_AREA['y1']
            rect.width = DETECTION_AREA['x2'] - DETECTION_AREA['x1']
            rect.height = DETECTION_AREA['y2'] - DETECTION_AREA['y1']
            rect.border_width = 1
            rect.border_color.set(*LINE_COLOR)
            rect.has_bg_color = 1
            rect.bg_color.set(*AREA_COLOR)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        print(f"Error drawing detection area: {str(e)}")

def is_in_detection_area(obj_meta):
    """Check if object is within the detection area"""
    try:
        center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
        center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
        return (DETECTION_AREA['x1'] <= center_x <= DETECTION_AREA['x2'] and
                DETECTION_AREA['y1'] <= center_y <= DETECTION_AREA['y2'])
    except Exception as e:
        print(f"Error checking detection area: {str(e)}")
        return False

def check_detection_area(obj_meta, frame_meta, name):
    """Check if object has entered the detection area"""
    obj_id = obj_meta.object_id
    current_in_area = is_in_detection_area(obj_meta)
    prev_data = TRACKED_OBJECTS.get(obj_id, {'in_area': False, 'name': None})

    if current_in_area and not prev_data['in_area']:
        TRACKED_OBJECTS[obj_id] = {'in_area': True, 'name': name}
        return True

    TRACKED_OBJECTS[obj_id] = {'in_area': current_in_area, 'name': name}
    return current_in_area


def display_name_on_frame(obj_meta, frame_meta, batch_meta, name):
    """Display the name on the frame"""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return False

        display_meta.num_labels = 1
        text_params = display_meta.text_params[0]
        text_params.display_text = name
        text_params.x_offset = int(obj_meta.rect_params.left)
        text_params.y_offset = int(max(10, obj_meta.rect_params.top - 10))
        text_params.font_params.font_name = "Serif"
        text_params.font_params.font_size = 20
        text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        return pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        return False

def pgie_src_filter_probe(pad, info, u_data):
    """Filter face detections after primary inference"""
    try:
        if not pad or not info:
            print("ERROR: Invalid probe parameters")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("ERROR: Unable to get GstBuffer")
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
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if obj_meta and hasattr(obj_meta, 'confidence') and obj_meta.confidence <= 0.6:
                            pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                    except Exception as e:
                        print(f"ERROR processing object meta: {str(e)}")
                        traceback.print_exc()
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                print(f"ERROR processing frame meta: {str(e)}")
                traceback.print_exc()
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        print(f"CRITICAL ERROR in pgie_src_filter_probe: {str(e)}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK

##MAIN SGIE:
#def sgie_feature_extract_probe(pad, info, data):
#    """Extract face features and match against known faces with camera source info"""
#    try:
#        if not pad or not info or not data:
#            print("ERROR: Invalid probe parameters")
#            return Gst.PadProbeReturn.OK
#
#        #WHEN CHANGING CODE TO DISPLAY CAMERA NAME REMOVE THE FOLLOWING LINE:
#        #loaded_faces = data[0] if data and len(data) > 0 else None
#        #AND ADD FOLLOWING CODE:
#        # Access dictionary values properly
#        loaded_faces = data.get('known_face_features')
#        save_feature = data.get('save_feature', False)
#        save_path = data.get('save_path')
#        sources = data.get('sources', {})
#        #*****************END OF CHANGE***********************
#
#        if not loaded_faces:
#            print("WARNING: No loaded faces provided")
#            return Gst.PadProbeReturn.OK
#
#        gst_buffer = info.get_buffer()
#        if not gst_buffer:
#            print("ERROR: Unable to get GstBuffer")
#            return Gst.PadProbeReturn.OK
#
#        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
#        if not batch_meta:
#            return Gst.PadProbeReturn.OK
#
#        l_frame = batch_meta.frame_meta_list
#        while l_frame is not None:
#            try:
#                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
#                if not frame_meta:
#                    break
#                #ADD FOLLOWING CODE WHEN CHANGES ARE BEING MADE FOR CMERA NAME DISPLAY:
#                # Get the source information for this frame
#                source_id = frame_meta.source_id
#                source_info = sources.get(f"source_{source_id}", {}).get("name", f"Camera {source_id}")
#                #*****************END OF CHANGE***********************
#                draw_detection_area(frame_meta, batch_meta)
#                l_obj = frame_meta.obj_meta_list
#                while l_obj is not None:
#                    try:
#                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
#                        if not obj_meta:
#                            continue
#
#                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, data)
#                        if face_feature is not None and loaded_faces:
#                            best_match_id, best_score = match_faces(
#                                obj_meta, frame_meta, batch_meta, face_feature, loaded_faces
#                            )
#
#                            # Only process matches with sufficient confidence
#                            if best_match_id and best_score > 0.3:
#                                in_area = check_detection_area(obj_meta, frame_meta, str(best_match_id))
#
#                                if in_area:
#                                    # Get user details from database
#                                    from .attendance import get_user_info, log_attendance
#                                    first_name, last_name = get_user_info(best_match_id)
#                                    if first_name and last_name:
#
#                                        #CHANGES MADE WHEN CAMERA NAME DISPLAY CODE IS ADDED:
#                                        #FROM THIS LINE:
#                                        #log_attendance(best_match_id)
#                                        #TO THIS LINE:
#                                        # Log attendance with source information
#                                        log_attendance(best_match_id, source=source_info)
#
#                                        display_name = f"{first_name} {last_name}"
#
#                                        #FROM THIS:
#                                        #display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)
#                                        #TO THIS:
#                                        display_name_on_frame(obj_meta, frame_meta, batch_meta,
#                                                           f"{best_match_id} ({source_info})")
#                                        #ADD THIS LINE:
#                                        # Print recognition with camera info
#                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                        print(f"[{timestamp}] {source_info}: Recognized {display_name} (ID: {best_match_id}, Score: {best_score:.2f})")
#                                    else:
#                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                        print(f"[{timestamp}] {source_info}: User ID {best_match_id} not found in database")
#                            elif save_feature:
#                                # For unknown faces being saved
#                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                print(f"[{timestamp}] {source_info}: Detected unknown face")
#                            #**************END OF CHANGE****************
#                    except Exception as e:
#                        print(f"ERROR processing object meta: {str(e)}")
#                        traceback.print_exc()
#                    finally:
#                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
#            except Exception as e:
#                print(f"ERROR processing frame meta: {str(e)}")
#                traceback.print_exc()
#            finally:
#                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
#    except Exception as e:
#        print(f"CRITICAL ERROR in sgie_feature_extract_probe: {str(e)}")
#        traceback.print_exc()
#    return Gst.PadProbeReturn.OK

#SGIE WITH WEB_FORM.PY:
def sgie_feature_extract_probe(pad, info, data):
    """Extract face features and match against known faces with camera source info"""
    try:
        if not pad or not info or not data:
            print("ERROR: Invalid probe parameters")
            return Gst.PadProbeReturn.OK

        # Access dictionary values properly
        loaded_faces = data.get('known_face_features')
        save_feature = data.get('save_feature', False)
        save_path = data.get('save_path')
        sources = data.get('sources', {})

        if not loaded_faces:
            print("WARNING: No loaded faces provided")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("ERROR: Unable to get GstBuffer")
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
                draw_detection_area(frame_meta, batch_meta)

                # Get the source information for this frame
                source_id = frame_meta.source_id
                source_key = f"source_{source_id}"
                source_info = sources.get(source_key, {})
                source_name = source_info.get("name", f"Camera {source_id}")
                camera_type = source_info.get("type", "unknown")  # 'entrance' or 'exit'

                #draw_detection_area(frame_meta, batch_meta)
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, data)
                        if face_feature is not None and loaded_faces:
                            best_match_id, best_score = match_faces(
                                obj_meta, frame_meta, batch_meta, face_feature, loaded_faces
                            )

                            # Only process matches with sufficient confidence
                            if best_match_id and best_score > 0.5:
                                in_area = check_detection_area(obj_meta, frame_meta, str(best_match_id))

                                if in_area:
                                    # Get user details from database
                                    from .file import get_user_info, log_attendance
                                    first_name, last_name = get_user_info(best_match_id)
                                    if first_name and last_name:
                                        # Get current timestamp
                                        now = datetime.now()
                                        current_date = now.strftime("%d-%m-%Y")
                                        current_time = now.strftime("%H:%M:%S")

                                        # Handle attendance based on camera type
                                        if camera_type == 'entrance':
                                            # For entrance camera - check in
                                            log_attendance(
                                                emp_id=best_match_id,
                                                first_name=first_name,
                                                last_name=last_name,
                                                Attendance_date=current_date,
                                                Attendance_time=current_time,
                                                check_type='in',
                                                camera_name=source_name
                                            )
                                        elif camera_type == 'exit':
                                            # For exit camera - check out
                                            log_attendance(
                                                emp_id=best_match_id,
                                                first_name=first_name,
                                                last_name=last_name,
                                                Attendance_date=current_date,
                                                Attendance_time=current_time,
                                                check_type='out',
                                                camera_name=source_name
                                            )

                                        display_name = f"{first_name} {last_name}"
                                        display_name_on_frame(
                                            obj_meta, frame_meta, batch_meta,
                                            f"{display_name} ({camera_type})"
                                        )

                                        # Print recognition with camera info
                                        timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
                                        print(f"[{timestamp}] {source_name} ({camera_type}): Recognized {display_name} (ID: {best_match_id}, Score: {best_score:.2f})")
                                    else:
                                        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                        print(f"[{timestamp}] {source_name}: User ID {best_match_id} not found in database")
                            elif save_feature:
                                # For unknown faces being saved
                                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                print(f"[{timestamp}] {source_name}: Detected unknown face")
                    except Exception as e:
                        print(f"ERROR processing object meta: {str(e)}")
                        traceback.print_exc()
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                print(f"ERROR processing frame meta: {str(e)}")
                traceback.print_exc()
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        print(f"CRITICAL ERROR in sgie_feature_extract_probe: {str(e)}")
        traceback.print_exc()
    return Gst.PadProbeReturn.OK

#SGIE WITH WEB_INTERFACE.PY:
#def sgie_feature_extract_probe(pad, info, data):
#    """Extract face features and match against known faces with camera source info"""
#    try:
#        if not pad or not info or not data:
#            print("ERROR: Invalid probe parameters")
#            return Gst.PadProbeReturn.OK
#
#        # Access dictionary values properly
#        loaded_faces = data.get('known_face_features')
#        save_feature = data.get('save_feature', False)
#        save_path = data.get('save_path')
#        sources = data.get('sources', {})
#
#        if not loaded_faces:
#            print("WARNING: No loaded faces provided")
#            return Gst.PadProbeReturn.OK
#
#        gst_buffer = info.get_buffer()
#        if not gst_buffer:
#            print("ERROR: Unable to get GstBuffer")
#            return Gst.PadProbeReturn.OK
#
#        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
#        if not batch_meta:
#            return Gst.PadProbeReturn.OK
#
#        l_frame = batch_meta.frame_meta_list
#        while l_frame is not None:
#            try:
#                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
#                if not frame_meta:
#                    break
#
#                # Get the source information for this frame
#                source_id = frame_meta.source_id
#                source_info = sources.get(f"source_{source_id}", {}).get("name", f"Camera {source_id}")
#
#                draw_detection_area(frame_meta, batch_meta)
#                l_obj = frame_meta.obj_meta_list
#                while l_obj is not None:
#                    try:
#                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
#                        if not obj_meta:
#                            continue
#
#                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, data)
#                        if face_feature is not None and loaded_faces:
#                            best_match_id, best_score = match_faces(
#                                obj_meta, frame_meta, batch_meta, face_feature, loaded_faces
#                            )
#
#                            # Only process matches with sufficient confidence
#                            if best_match_id and best_score > 0.3:
#                                in_area = check_detection_area(obj_meta, frame_meta, str(best_match_id))
#
#                                if in_area:
#                                    # Get user details from database
#                                    from .file import get_user_info, log_attendance
#                                    first_name, last_name = get_user_info(best_match_id)
#                                    if first_name and last_name:
#                                        # Determine if this is check-in or check-out based on time of day
#                                        current_hour = datetime.now().hour
#                                        is_check_in = current_hour < 12  # Before noon is check-in
#
#                                        # Log attendance with source information and check-in/check-out status
#                                        log_attendance(
#                                            user_id=best_match_id,
#                                            source=source_info,
#                                            is_check_in=is_check_in
#                                        )
#
#                                        display_name = f"{first_name} {last_name}"
#                                        action = "Check-in" if is_check_in else "Check-out"
#
#                                        # Display name with camera info and action
#                                        display_name_on_frame(
#                                            obj_meta,
#                                            frame_meta,
#                                            batch_meta,
#                                            f"{display_name} ({action})"
#                                        )
#
#                                        # Print recognition with camera info and action
#                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                        print(f"[{timestamp}] {source_info}: {action} - Recognized {display_name} (ID: {best_match_id}, Score: {best_score:.2f})")
#                                    else:
#                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                        print(f"[{timestamp}] {source_info}: User ID {best_match_id} not found in database")
#                            elif save_feature:
#                                # For unknown faces being saved
#                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                print(f"[{timestamp}] {source_info}: Detected unknown face")
#                    except Exception as e:
#                        print(f"ERROR processing object meta: {str(e)}")
#                        traceback.print_exc()
#                    finally:
#                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
#            except Exception as e:
#                print(f"ERROR processing frame meta: {str(e)}")
#                traceback.print_exc()
#            finally:
#                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
#    except Exception as e:
#        print(f"CRITICAL ERROR in sgie_feature_extract_probe: {str(e)}")
#        traceback.print_exc()
#    return Gst.PadProbeReturn.OK



def match_faces(obj_meta, frame_meta, batch_meta, face_feature, loaded_faces):
    """Match faces against known faces and return the best matching ID"""
    best_match_id = None
    best_score = 0.0

    try:
        for user_id, known_feature in loaded_faces.items():
            try:
                if known_feature is None:
                    continue

                # Calculate similarity score and ensure it's a scalar float
                score = float(np.dot(face_feature, known_feature).item())

                # Update best match if this score is higher
                if score > best_score:
                    best_score = score
                    best_match_id = user_id

                # Debug output
                print(f"Frame {frame_meta.frame_num}, Face {obj_meta.object_id} vs User {user_id}: Score {score:.4f}")

            except Exception as e:
                print(f"ERROR matching face {user_id}: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"CRITICAL ERROR in match_faces: {str(e)}")
        traceback.print_exc()

    return best_match_id, best_score

def get_face_feature(obj_meta, frame_num, data):
    """Extract face features from tensor meta with dictionary support"""
    try:
        if not obj_meta or not hasattr(obj_meta, 'obj_user_meta_list'):
            return None

        #ADD FOLLOWING LINES:
        # Handle both dictionary and list formats for backward compatibility
        if isinstance(data, dict):
            save_feature = data.get('save_feature', False)
            save_path = data.get('save_path')
        else:
            # Legacy list format support
            save_feature = data[1] if data and len(data) > 1 else False
            save_path = data[2] if data and len(data) > 2 else None
        #*****************END OF CHANGE*************
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
                        break

                    output = []
                    for i in range(512):
                        try:
                            val = pyds.get_detections(layer.buffer, i)
                            if val is None:
                                break
                            output.append(val)
                        except Exception as e:
                            print(f"ERROR getting detection {i}: {str(e)}")
                            break

                    if len(output) == 512:
                        res = np.reshape(output, (1, -1))
                        norm = np.linalg.norm(res)
                        if norm > 0:
                            normal_array = res / norm
                            #REMOVE FOLLOWING LINES:
                            #if data and len(data) > 1 and data[1] and len(data) > 2 and data[2]:
                            #    try:
                            #        save_path = os.path.join(data[2], f"{obj_meta.object_id}-{frame_num}.npy")
                            #        np.save(save_path, normal_array)
                            #ADD FOLLWING LINE
                            if save_feature and save_path:
                                try:
                                    # Create directory if it doesn't exist
                                    os.makedirs(save_path, exist_ok=True)
                                    save_file = os.path.join(save_path, f"{obj_meta.object_id}-{frame_num}.npy")
                                    np.save(save_file, normal_array)
                                    print(f"Saved face feature to {save_file}")
                            #***************END OF CHANGE***************
                                except Exception as e:
                                    print(f"ERROR saving feature: {str(e)}")
                            return normal_array
            except Exception as e:
                print(f"ERROR processing user meta: {str(e)}")
                traceback.print_exc()
            finally:
                l_user_meta = l_user_meta.next if hasattr(l_user_meta, 'next') else None
    except Exception as e:
        print(f"CRITICAL ERROR in get_face_feature: {str(e)}")
        traceback.print_exc()
    return None
