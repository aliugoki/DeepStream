import os
import sys
import numpy as np
import traceback
from datetime import datetime
import logging

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

# Detection area configuration
DETECTION_AREA = {
    'x1': 0,  # Left boundary
    'y1': 150,  # Top boundary
    'x2': 2500,  # Right boundary
    'y2': 2000,  # Bottom boundary
}
AREA_COLOR = (0.0, 0.0, 0.0, 0.0)  # RGBA color (Fully transparent black)
LINE_COLOR = (0.0, 1.0, 0.0, 0.8)  # RGBA color (Green with 80% opacity)
TRACKED_OBJECTS = {}  # Track objects in detection area

# Cooldown period for logging attendance (in seconds)
ATTENDANCE_COOLDOWN_SECONDS = 10 # Example: log attendance only once every 10 seconds per person

# Threshold for face recognition similarity score
# This value needs to be tuned based on your specific model's performance
# and your desired balance between false positives and false negatives.
# A common range is 0.6 to 0.75 for good quality models.
RECOGNITION_THRESHOLD = 0.3 # Increased from 0.5 for better accuracy

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
            # Use set() for color parameters as they expect individual floats
            rect.border_color.set(*LINE_COLOR)
            rect.has_bg_color = 1
            rect.bg_color.set(*AREA_COLOR)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        logger.error(f"Error drawing detection area: {str(e)}")

def is_in_detection_area(obj_meta):
    """Check if object is within the detection area"""
    try:
        # Ensure rect_params are valid
        if not obj_meta or not obj_meta.rect_params:
            logger.warning("Object metadata or rect_params are invalid for area check.")
            return False

        center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
        center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
        
        # Check if the center of the bounding box is within the defined area
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

    prev_data = TRACKED_OBJECTS.get(obj_id, {
        'in_area': False,
        'name': None,
        'last_logged_time': None # Track last attendance log time for this object
    })

    # Update the current state of the object in TRACKED_OBJECTS
    TRACKED_OBJECTS[obj_id] = {
        'in_area': current_in_area,
        'name': name,
        'last_logged_time': prev_data['last_logged_time'] # Preserve last logged time for decision
    }

    if current_in_area:
        # Check if it's the first time this object is detected in the area,
        # OR if enough time has passed since its last attendance log.
        if prev_data['last_logged_time'] is None or \
           (current_timestamp - prev_data['last_logged_time']).total_seconds() > ATTENDANCE_COOLDOWN_SECONDS:
            
            # Update the last logged time, as attendance is about to be logged
            TRACKED_OBJECTS[obj_id]['last_logged_time'] = current_timestamp
            return True # Signal for attendance logging
    
    return False # Not in area, or already logged recently

def display_name_on_frame(obj_meta, frame_meta, batch_meta, name):
    """Display the name on the frame"""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            logger.warning("Failed to acquire display meta.")
            return False

        # Max 16 labels can be added to one NvDsDisplayMeta object
        if display_meta.num_labels >= 16:
            # Release the acquired display_meta if it's full to prevent leaks
            pyds.nvds_release_display_meta(display_meta)
            logger.warning("Exceeded max labels for display meta. Skipping adding new label.")
            return False

        text_params = display_meta.text_params[display_meta.num_labels]
        text_params.display_text = name
        
        # Adjust y_offset to place text above the bounding box
        # Ensure text is not off-screen at the top (minimum 0)
        text_params.x_offset = int(obj_meta.rect_params.left)
        text_params.y_offset = int(max(0, obj_meta.rect_params.top - 20)) # Move up by 20 pixels
        
        text_params.font_params.font_name = "Serif"
        text_params.font_params.font_size = 15 # Slightly smaller font size for readability
        
        # Ensure color values are floats between 0.0 and 1.0
        text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0) # White color for text
        text_params.set_bg_clr = 1 # Enable background color for text
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6) # Semi-transparent black background
        
        display_meta.num_labels += 1 # Increment the count of labels added to this display meta
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta) # Add to the frame
        return True
    except Exception as e:
        logger.error(f"Error displaying name on frame for object {obj_meta.object_id if obj_meta else 'N/A'}: {str(e)}")
        traceback.print_exc()
        return False

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
                # Collect objects to remove to avoid issues with modifying the list while iterating.
                objects_to_remove = []
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if obj_meta and hasattr(obj_meta, 'confidence'):
                            # Confidence threshold for PGIE (e.g., general object detection)
                            # This threshold is separate from the face recognition threshold.
                            # It filters out weak initial detections.
                            PGIE_CONFIDENCE_THRESHOLD = 0.6 # Adjust as needed for your PGIE model

                            # Only consider removing if it's a 'face' class if your PGIE detects other objects.
                            # You might need to check obj_meta.class_id if your PGIE is multi-class.
                            # Example: if obj_meta.class_id == FACE_CLASS_ID and obj_meta.confidence <= PGIE_CONFIDENCE_THRESHOLD:
                            if obj_meta.confidence <= PGIE_CONFIDENCE_THRESHOLD:
                                objects_to_remove.append(obj_meta)
                    except Exception as e:
                        logger.error(f"Error processing object meta in pgie_src_filter_probe: {str(e)}")
                        traceback.print_exc()
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
                
                # Remove collected objects after iterating
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
    Handles attendance logging, name display, and face crop saving.
    """
    try:
        if not pad or not info or not data:
            logger.error("Invalid probe parameters in sgie_feature_extract_probe.")
            return Gst.PadProbeReturn.OK

        # Ensure data contains loaded_faces (data[0]) and feature_save_dir (data[2])
        if not isinstance(data, (list, tuple)) or len(data) < 3:
            logger.error("Invalid 'data' format for sgie_feature_extract_probe. Expected [loaded_faces, ..., feature_save_dir]")
            return Gst.PadProbeReturn.OK

        loaded_faces = data[0] if data and len(data) > 0 else None
        feature_save_dir = data[2] # This is where the feature will be saved if available

        if not loaded_faces:
            logger.warning("No loaded faces provided for feature extraction and matching.")
            # Still process frames and draw detection area, but no recognition will happen.
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

                draw_detection_area(frame_meta, batch_meta) # Draw area on each frame

                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        # It's good practice to ensure this object is indeed a 'face'
                        # if your PGIE detects other objects. Example:
                        # if obj_meta.class_id != YOUR_FACE_CLASS_ID:
                        #    l_obj = l_obj.next
                        #    continue

                        # Pass feature_save_dir to get_face_feature
                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, feature_save_dir)
                        
                        if face_feature is not None and loaded_faces:
                            # Proceed with face matching only if features are extracted and known faces exist
                            best_match_id, best_score = match_faces(face_feature, loaded_faces)

                            if best_match_id and best_score >= RECOGNITION_THRESHOLD:
                                # Import attendance and crop utility functions only when a positive match is found
                                # to potentially manage dependencies and avoid circular imports.
                                from .attendance import get_user_info, log_attendance
                                from .crop_utils import save_face_crop

                                first_name, last_name = get_user_info(best_match_id)
                                if first_name and last_name:
                                    display_name = f"{first_name} {last_name}"
                                    
                                    # check_detection_area now handles cooldown and returns True if attendance should be logged
                                    if check_detection_area(obj_meta, display_name):
                                        log_attendance(best_match_id)

                                        output_dir = "/workspace/face_final/output_faces"
                                        os.makedirs(output_dir, exist_ok=True)

                                        # Use a unique filename for face crops to prevent overwriting
                                        crop_filename = f"face_{best_match_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                        
                                        # Pass frame_meta here!
                                        save_face_crop(gst_buffer, frame_meta, obj_meta, output_dir, crop_filename) # <--- THIS IS THE CALL YOU NEED TO CHANGE

                                        logger.info(f"Recognized: ID {best_match_id} ({display_name}) with score {best_score:.2f}. Attendance logged.")
                                    else:
                                        logger.debug(f"Recognized ID {best_match_id} ({display_name}) with score {best_score:.2f}. Not logging attendance (cooldown/out of area).")
                                    
                                    # Always display name if recognized, regardless of attendance logging or area.
                                    display_name_on_frame(obj_meta, frame_meta, batch_meta, display_name)
                                else:
                                    # Recognized ID, but user details are missing in DB
                                    logger.warning(f"Recognized face with ID {best_match_id}, but user details not found in database or incomplete.")
                                    display_name_on_frame(obj_meta, frame_meta, batch_meta, f"ID:{best_match_id} (No Info)")
                            else:
                                # Face recognized but score too low, or no match found among loaded_faces
                                # Display "Unknown" or the ID with low score
                                if best_match_id: # If there was a closest match but score was too low
                                    display_name_on_frame(obj_meta, frame_meta, batch_meta, f"Unknown (ID:{best_match_id}, {best_score:.2f})")
                                    logger.info(f"Face matched ID {best_match_id} but score {best_score:.2f} is below threshold {RECOGNITION_THRESHOLD}.")
                                else: # No match found at all
                                    display_name_on_frame(obj_meta, frame_meta, batch_meta, "Unknown")
                                    logger.debug(f"No match found for object {obj_meta.object_id} (frame {frame_meta.frame_num}).")
                        elif face_feature is None:
                            logger.debug(f"No face feature extracted for object {obj_meta.object_id}.")
                            display_name_on_frame(obj_meta, frame_meta, batch_meta, "No Feature")
                        else: # loaded_faces is empty
                            logger.debug("No known faces to match against.")
                            display_name_on_frame(obj_meta, frame_meta, batch_meta, "No Known Faces")


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
    best_score = 0.0  # Initialize with a score lower than any possible valid score (cosine similarity is -1 to 1)

    try:
        # Ensure face_feature is a flat array for dot product
        face_feature_flat = face_feature.flatten()

        for user_id, known_feature in loaded_faces.items():
            if known_feature is None:
                continue

            # Ensure known_feature is also a flat array
            known_feature_flat = known_feature.flatten()

            # Ensure both feature vectors have the same dimension
            if face_feature_flat.shape != known_feature_flat.shape:
                logger.warning(f"Feature dimension mismatch for user {user_id}. Skipping match.")
                continue

            # Calculate cosine similarity using dot product (since vectors are assumed normalized)
            score = np.dot(face_feature_flat, known_feature_flat)

            # Update best match if this score is higher
            if score > best_score:
                best_score = score
                best_match_id = user_id

            logger.debug(f"Matching with user {user_id}: Score {score:.4f}")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in match_faces: {str(e)}")
        traceback.print_exc()

    return best_match_id, best_score

def get_face_feature(obj_meta, frame_num, feature_save_dir=None):
    """
    Extract face features (embedding) from SGIE's tensor output.
    Uses pyds.get_detections to access tensor data, compatible with older DeepStream versions.
    Optionally saves the extracted feature to a file.
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

                # Check if the user meta is indeed a tensor output meta from SGIE
                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    if not tensor_meta:
                        break

                    # Assuming the face embedding is the first layer (index 0)
                    # You MUST verify this with your SGIE model's output structure.
                    # Some models might have multiple output layers (e.g., bounding box + feature).
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    if not layer or not layer.buffer:
                        logger.warning("Tensor layer or buffer not found for feature extraction.")
                        break

                    # Define the expected size of the face embedding.
                    # This MUST match the output dimension of your face recognition model.
                    FEATURE_VECTOR_SIZE = 512 # Common for face embeddings (e.g., FaceNet)
                    
                    output_list = []
                    # Iterate to extract individual float values from the layer buffer
                    for i in range(FEATURE_VECTOR_SIZE):
                        try:
                            # pyds.get_detections can be used to extract scalar values
                            # from a flat buffer at a given index.
                            val = pyds.get_detections(layer.buffer, i)
                            if val is None:
                                # This can happen if the buffer is smaller than FEATURE_VECTOR_SIZE
                                logger.warning(f"Failed to get detection at index {i}. Buffer might be smaller than expected. Expected size: {FEATURE_VECTOR_SIZE}")
                                output_list = [] # Clear incomplete data
                                break # Exit loop
                            output_list.append(val)
                        except Exception as e:
                            logger.error(f"ERROR getting detection {i} from tensor buffer: {str(e)}")
                            traceback.print_exc()
                            output_list = [] # Clear incomplete data on error
                            break # Exit loop on error

                    if len(output_list) == FEATURE_VECTOR_SIZE:
                        # Convert list to NumPy array and reshape to (1, FEATURE_VECTOR_SIZE)
                        # Ensure data type is float32, which is typical for model outputs.
                        res = np.array(output_list, dtype=np.float32).reshape((1, FEATURE_VECTOR_SIZE))
                        
                        # Normalize the feature vector to unit length (L2 normalization)
                        # This is crucial for cosine similarity, which is often used with face embeddings.
                        norm = np.linalg.norm(res)
                        if norm > 0:
                            normal_array = res / norm # L2 normalization
                            
                            # Save the extracted and normalized feature if a directory is provided
                            if feature_save_dir:
                                try:
                                    os.makedirs(feature_save_dir, exist_ok=True)
                                    # Include timestamp in filename for uniqueness
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