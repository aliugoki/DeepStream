import os
import sys
import numpy as np
import traceback
from datetime import datetime
import logging # Import logging module

# Configure logging for probe.py
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

import pyds

# Import your attendance functions
# Ensure attendance.py is in the same directory or properly accessible via PYTHONPATH
try:
    from .attendance import log_attendance, get_user_info
except ImportError as e:
    logger.critical(f"Failed to import attendance functions: {e}. Make sure attendance.py is in the correct location and contains log_attendance and get_user_info.")
    sys.exit(1)


# Detection area configuration
DETECTION_AREA = {
    'x1': 0,  # Left boundary
    'y1': 150,  # Top boundary
    'x2': 2500,  # Right boundary
    'y2': 2000,  # Bottom boundary
}
AREA_COLOR = (0, 0, 0, 0.0)  # RGBA color
LINE_COLOR = (0, 255, 0, 0.8)  # RGBA color
TRACKED_OBJECTS = {}  # Track objects in detection area

# --- Global Data for Face Recognition (to be populated by load_faces) ---
# This will store {user_id: np.array(feature_vector)}
# We assume one feature vector per user_id for simplicity in matching.
# If a user has multiple known faces, their features should be averaged
# or handled during the pre-computation/loading phase.
GLOBAL_KNOWN_FACE_FEATURES = {}
FEATURE_MATCH_THRESHOLD = 0.6 # Cosine similarity threshold for recognition


# Global to keep track of already sent attendance for the day
# Stores (emp_id, date_str) for daily unique attendance logging
_attendance_sent_today = set()
_last_check_date = None # To reset the set daily

# --- Helper Functions ---

def load_faces(known_face_dir):
    """
    Loads known face features from a specified directory.
    Assumes that:
    1. `known_face_dir` contains subfolders, each representing a user.
    2. Each subfolder is named after the `user_id` (e.g., "101", "102").
    3. Inside each `user_id` folder, there's a pre-computed face feature `.npy` file.
       (e.g., "101/feature.npy" or "101/average_feature.npy").
       If there are multiple `.npy` files, it will load the first one found.
    
    Populates the GLOBAL_KNOWN_FACE_FEATURES dictionary.
    """
    global GLOBAL_KNOWN_FACE_FEATURES
    GLOBAL_KNOWN_FACE_FEATURES.clear() # Clear previous data

    if not os.path.exists(known_face_dir):
        logger.error(f"Known faces directory not found: {known_face_dir}")
        return

    logger.info(f"Loading known face features from: {known_face_dir}")

    for user_id_str in os.listdir(known_face_dir):
        user_dir = os.path.join(known_face_dir, user_id_str)
        if not os.path.isdir(user_dir):
            continue

        try:
            user_id = int(user_id_str) # Convert folder name to integer ID
        except ValueError:
            logger.warning(f"Skipping non-integer directory name in known faces: {user_id_str}")
            continue

        feature_found = False
        for filename in os.listdir(user_dir):
            if filename.lower().endswith('.npy'):
                feature_path = os.path.join(user_dir, filename)
                try:
                    feature_vector = np.load(feature_path).astype(np.float32)
                    if feature_vector.ndim > 1: # Ensure it's a 1D vector
                        feature_vector = feature_vector.flatten()
                    
                    # Normalize the feature vector if it's not already
                    norm = np.linalg.norm(feature_vector)
                    if norm > 0:
                        feature_vector = feature_vector / norm
                    else:
                        logger.warning(f"Zero norm feature vector for user {user_id} in {feature_path}. Skipping.")
                        continue

                    GLOBAL_KNOWN_FACE_FEATURES[user_id] = feature_vector
                    logger.debug(f"Loaded feature for User ID {user_id} from {feature_path}")
                    feature_found = True
                    break # Load only the first .npy file found per user
                except Exception as e:
                    logger.error(f"Error loading feature from {feature_path} for user {user_id}: {e}")
        
        if not feature_found:
            logger.warning(f"No .npy feature file found for User ID {user_id} in {user_dir}")

    logger.info(f"Finished loading known face features. Total unique users loaded: {len(GLOBAL_KNOWN_FACE_FEATURES)}")


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
            rect.border_width = 2
            rect.border_color.set(*LINE_COLOR)
            rect.has_bg_color = 1
            rect.bg_color.set(*AREA_COLOR)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error drawing detection area: {str(e)}")

def is_in_detection_area(obj_meta):
    """Check if object is within the detection area"""
    try:
        center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
        center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
        return (DETECTION_AREA['x1'] <= center_x <= DETECTION_AREA['x2'] and
                DETECTION_AREA['y1'] <= center_y <= DETECTION_AREA['y2'])
    except Exception as e:
        logger.error(f"Error checking detection area: {str(e)}")
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
            logger.warning("Failed to acquire display meta.")
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
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        return True
    except Exception as e:
        logger.error(f"Error displaying name on frame: {str(e)}\n{traceback.format_exc()}")
        return False

# --- GStreamer Probe Callbacks ---

def pgie_src_filter_probe(pad, info, u_data):
    """Filter face detections after primary inference (PGIE)"""
    try:
        if not pad or not info:
            logger.error("Invalid probe parameters for pgie_src_filter_probe")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("Unable to get GstBuffer in pgie_src_filter_probe")
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
                        if obj_meta and obj_meta.class_id == 0: # Assuming class_id 0 is 'person' for PGIE
                            # You might filter persons here based on confidence or other criteria
                            # obj_meta.confidence is typically from PGIE.
                            if hasattr(obj_meta, 'confidence') and obj_meta.confidence <= 0.6:
                                # For PGIE, typically remove low-confidence *person* detections if SGIE follows
                                # Or just skip further processing for low-confidence detections.
                                # The current code removes the object meta, which might be too aggressive
                                # if SGIE is expecting it.
                                # If SGIE's input is a cropped person, and SGIE does face detection,
                                # removing the person object here might prevent SGIE from running.
                                # It's more common to set object_meta.rect_params.width = 0 etc. or just ignore.
                                # Let's keep it as is based on your original code's intent to remove.
                                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                        # else: # If not a person, or other class_ids
                            # If PGIE detects faces directly, its class_id would be different.
                            # Adjust filtering based on your PGIE model's output.
                            pass

                    except Exception as e:
                        logger.error(f"Error processing object meta in pgie_src_filter_probe: {str(e)}\n{traceback.format_exc()}")
                    finally:
                        # This 'next' check handles both pyds.NvDsObjectMeta.next and standard Python iteration
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                logger.error(f"Error processing frame meta in pgie_src_filter_probe: {str(e)}\n{traceback.format_exc()}")
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in pgie_src_filter_probe: {str(e)}\n{traceback.format_exc()}")
    return Gst.PadProbeReturn.OK


def sgie_feature_extract_probe(pad, info, data):
    """Extract face features and match against known faces"""
    try:
        if not pad or not info or not data:
            logger.error("Invalid probe parameters for sgie_feature_extract_probe")
            return Gst.PadProbeReturn.OK

        # `data` parameter from main_webrtc.py's sgie_src_pad.add_probe
        # should pass GLOBAL_KNOWN_FACE_FEATURES directly, not as data[0].
        # For compatibility, we'll try to get it this way, but ideally,
        # it should just be `loaded_faces = data`
        # However, your `main_webrtc.py` passes `data = [known_face_features, save_feature, save_path]`.
        # So `loaded_faces` is indeed `data[0]`.
        # `save_feature` is `data[1]`, `save_path` is `data[2]`.
        
        # Access the GLOBAL_KNOWN_FACE_FEATURES directly, or through `data[0]` if passed this way
        # Since your main passes it as `data[0]`, we'll keep that.
        # But ensure your main.py calls `load_faces()` to populate GLOBAL_KNOWN_FACE_FEATURES first.
        # It's less prone to error if `sgie_feature_extract_probe` directly uses `GLOBAL_KNOWN_FACE_FEATURES`.
        
        # Let's assume `data[0]` is indeed the dict of loaded faces.
        loaded_faces_dict = data[0] if data and len(data) > 0 else None
        save_face_crops_enabled = data[1] if data and len(data) > 1 else False
        output_dir = data[2] if data and len(data) > 2 else None

        if not loaded_faces_dict:
            logger.warning("No loaded face features provided to sgie_feature_extract_probe. Face recognition will not work.")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("Unable to get GstBuffer in sgie_feature_extract_probe")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        global _attendance_sent_today, _last_check_date
        current_date_str = datetime.now().strftime("%Y-%m-%d")

        # Reset daily attendance tracking
        if _last_check_date is None or _last_check_date != current_date_str:
            logger.info(f"Resetting daily attendance tracking for {current_date_str}")
            _attendance_sent_today.clear()
            _last_check_date = current_date_str

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                if not frame_meta:
                    break

                draw_detection_area(frame_meta, batch_meta) # Draw detection area for each frame
                
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        # Ensure this is a face object (if SGIE detects faces)
                        # Or, if PGIE detects person and SGIE attaches face meta to person
                        # This part depends on your model architecture.
                        # Assuming SGIE's output is directly a face feature
                        
                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, data) # data is passed for saving feature
                        if face_feature is not None: # Face feature successfully extracted
                            best_match_id, best_score = match_faces(
                                obj_meta, frame_meta, batch_meta, face_feature, loaded_faces_dict
                            )

                            if best_match_id is not None and best_score >= FEATURE_MATCH_THRESHOLD: # Use the global threshold
                                in_area = check_detection_area(obj_meta, frame_meta, str(best_match_id))

                                if in_area:
                                    # Get user details from database
                                    first_name, last_name = get_user_info(best_match_id)
                                    if first_name is not None and last_name is not None:
                                        display_name = f"{first_name} {last_name}"
                                        display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)

                                        attendance_key = (best_match_id, current_date_str)
                                        if attendance_key not in _attendance_sent_today:
                                            # Log attendance
                                            # The save_face_crop function would generate the image_url
                                            # For now, pass None or a default if not saving actual crops
                                            # If you implement save_face_crop, ensure it returns the path
                                            
                                            # Dummy image_url for attendance if crop saving isn't implemented
                                            # or if we are using known_faces directory.
                                            image_url_for_attendance = None 
                                            if save_face_crops_enabled and output_dir:
                                                # TODO: Implement save_face_crop and get the actual path
                                                # For now, it's just a placeholder
                                                logger.warning("save_face_crop is enabled but not fully implemented to return path. "
                                                               "Attendance will use a default image if provided by FastAPI.")
                                                # If you save crops, this is where you'd call it and get the path:
                                                # image_url_for_attendance = save_face_crop(...) 
                                            
                                            log_attendance(
                                                emp_id=best_match_id,
                                                first_name=first_name,
                                                last_name=last_name,
                                                image_url=image_url_for_attendance # Pass the generated image path or None
                                            )
                                            _attendance_sent_today.add(attendance_key)
                                            logger.info(f"Attendance logged for ID {best_match_id} ({display_name}) with score {best_score:.2f}")
                                        else:
                                            logger.debug(f"Attendance for ID {best_match_id} ({display_name}) already logged today.")
                                    else:
                                        logger.warning(f"User ID {best_match_id} not found in database (or name incomplete). Cannot log attendance.")
                                        display_name = f"Unknown ID:{best_match_id} ({best_score:.2f})"
                                        display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)
                                else:
                                    display_name = f"ID:{best_match_id} (Outside Area)"
                                    display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)
                                    logger.debug(f"Recognized ID {best_match_id} but outside detection area.")
                            else:
                                display_name = f"Unknown ({obj_meta.object_id})"
                                display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)
                                logger.debug(f"Face {obj_meta.object_id} not recognized (best score: {best_score:.2f}).")
                        else:
                            logger.debug(f"No face feature extracted for object {obj_meta.object_id}.")

                    except Exception as e:
                        logger.error(f"ERROR processing object meta in sgie_feature_extract_probe loop: {str(e)}\n{traceback.format_exc()}")
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                logger.error(f"ERROR processing frame meta in sgie_feature_extract_probe loop: {str(e)}\n{traceback.format_exc()}")
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in sgie_feature_extract_probe: {str(e)}\n{traceback.format_exc()}")
    return Gst.PadProbeReturn.OK


def match_faces(obj_meta, frame_meta, batch_meta, face_feature, loaded_faces_dict):
    """Match faces against known faces and return the best matching ID and score."""
    best_match_id = None
    best_score = 0.0 # Initialize with 0.0, as scores are between -1 and 1

    try:
        # Normalize the input face_feature if not already
        norm_face_feature = face_feature / np.linalg.norm(face_feature) if np.linalg.norm(face_feature) > 0 else face_feature

        for user_id, known_feature in loaded_faces_dict.items():
            try:
                if known_feature is None or np.linalg.norm(known_feature) == 0:
                    logger.warning(f"Skipping empty or zero-norm known feature for user {user_id}")
                    continue

                # Ensure known_feature is also normalized (should be from load_faces)
                # Cosine similarity between two normalized vectors is simply their dot product.
                score = np.dot(norm_face_feature, known_feature)
                
                # Update best match if this score is higher
                if score > best_score:
                    best_score = score
                    best_match_id = user_id

                logger.debug(f"Frame {frame_meta.frame_num}, Face {obj_meta.object_id} vs User {user_id}: Score {score:.4f}")

            except Exception as e:
                logger.error(f"ERROR matching face with user {user_id}: {str(e)}\n{traceback.format_exc()}")

    except Exception as e:
        logger.critical(f"CRITICAL ERROR in match_faces: {str(e)}\n{traceback.format_exc()}")

    return best_match_id, best_score

def get_face_feature(obj_meta, frame_num, data):
    """Extract face features from tensor meta (from SGIE output)"""
    try:
        if not obj_meta or not hasattr(obj_meta, 'obj_user_meta_list'):
            return None

        l_user_meta = obj_meta.obj_user_meta_list
        while l_user_meta:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                if not user_meta:
                    break

                # Check for NVDSINFER_TENSOR_OUTPUT_META which contains SGIE's output (feature vector)
                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    if not tensor_meta:
                        break

                    # Assuming the feature vector is the first (and only) layer output
                    # Adjust layer index if your model outputs multiple layers
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0) 
                    if not layer or not layer.buffer:
                        logger.warning(f"Layer 0 or its buffer not found in tensor meta for object {obj_meta.object_id}")
                        break

                    # Directly access the buffer for feature vector
                    # Feature vector is typically a float array
                    feature_vector_ptr = hash(layer.buffer) + layer.offset
                    feature_vector = np.array(pyds.get_detections(feature_vector_ptr, layer.dims.numElements), dtype=np.float32)
                    
                    if feature_vector.size != layer.dims.numElements:
                        logger.error(f"Extracted feature vector size mismatch for object {obj_meta.object_id}. Expected {layer.dims.numElements}, got {feature_vector.size}")
                        return None
                    
                    # Normalize the feature vector (important for cosine similarity)
                    norm = np.linalg.norm(feature_vector)
                    if norm > 0:
                        normal_array = feature_vector / norm
                    else:
                        logger.warning(f"Zero norm feature vector extracted for object {obj_meta.object_id}. Skipping normalization.")
                        normal_array = feature_vector

                    # `data[1]` is `save_feature`, `data[2]` is `save_path` from main_webrtc.py
                    if data and len(data) > 1 and data[1] and len(data) > 2 and data[2]:
                        try:
                            # This part is for saving *detected* face features to disk (for debugging/re-training)
                            # not for loading known faces.
                            save_path_base = data[2] # This is the output directory configured in main.py
                            output_features_dir = os.path.join(save_path_base, "detected_features")
                            os.makedirs(output_features_dir, exist_ok=True)
                            save_feature_path = os.path.join(output_features_dir, f"{obj_meta.object_id}-{frame_num}-{datetime.now().strftime('%H%M%S')}.npy")
                            np.save(save_feature_path, normal_array)
                            logger.debug(f"Saved detected feature to {save_feature_path}")
                        except Exception as e:
                            logger.error(f"ERROR saving detected feature for object {obj_meta.object_id}: {str(e)}")
                    
                    return normal_array
            except Exception as e:
                logger.error(f"ERROR processing user meta (tensor output) for object {obj_meta.object_id}: {str(e)}\n{traceback.format_exc()}")
            finally:
                l_user_meta = l_user_meta.next if hasattr(l_user_meta, 'next') else None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in get_face_feature: {str(e)}\n{traceback.format_exc()}")
    return None

# --- If you decide to implement save_face_crop (requires OpenCV and NVMM buffer handling) ---
# def save_face_crop(gst_buffer, obj_meta, output_dir, filename_prefix):
#     """
#     Saves the cropped face image from the GstBuffer.
#     This requires complex handling of NVMM buffers and conversion to OpenCV image.
#     """
#     try:
#         # Acquire buffer surface from NvBufSurface
#         nvds_frame_surface = pyds.get_nvds_buf_surface(gst_buffer, 0) # Assuming stream 0
#         
#         # Convert NvBufSurface to OpenCV image (complex, might need specific DeepStream utility)
#         # Example (conceptual, actual implementation needs care with memory mapping):
#         # frame_image = np.array(nvds_frame_surface, copy=False) 
#         # frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR) # Or appropriate conversion
#         
#         # If you have the frame_image:
#         # x = int(obj_meta.rect_params.left)
#         # y = int(obj_meta.rect_params.top)
#         # w = int(obj_meta.rect_params.width)
#         # h = int(obj_meta.rect_params.height)
#         # cropped_face = frame_image[y:y+h, x:x+w]
#         
#         # os.makedirs(output_dir, exist_ok=True)
#         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         # filepath = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.jpg")
#         # cv2.imwrite(filepath, cropped_face)
#         # logger.info(f"Saved face crop to: {filepath}")
#         # return filepath # Return the path for attendance logging
#     except Exception as e:
#         logger.error(f"Error saving face crop: {e}\n{traceback.format_exc()}")
#     return None # Return None on failure