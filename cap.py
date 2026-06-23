#!/usr/bin/env python3
"""
DeepStream Face Detection, Recognition and RTSP Streaming Pipeline
===================================================================
Fixed version with proper recognition tracking between probes
"""

import argparse
import sys
import os
import threading
import queue
import time
import traceback
import json
import numpy as np
import cv2
from datetime import datetime

# Disable proxy to prevent libproxy crash
os.environ['GIO_USE_VFS'] = 'local'
os.environ['GIO_USE_PROXY_RESOLVER'] = 'dummy'
os.environ['GIO_MODULE_DIR'] = '/nonexistent'
os.environ['no_proxy'] = '*'
os.environ['ALL_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['FTP_PROXY'] = ''

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
import pyds

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
perf_data = None
saved_count = {}
PGIE_CLASS_ID_FACE = 0
folder_name = "out_crops"

# Thread-safe queue for saving images
image_queue = queue.Queue(maxsize=100)

# Dictionaries for tracking recognition across probes
recognized_faces = {}  # For display probe (simple mapping)
object_recognition_status = {}  # For save probe (detailed info with timestamp)

# Constants
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 720
TILED_OUTPUT_HEIGHT = 576

# ============================================================================
# RECOGNITION TRACKING FUNCTIONS
# ============================================================================
def update_object_recognition(obj_id, name, confidence):
    """
    Update recognition status for an object with timestamp
    This allows save probe to know if an object was recognized
    """
    global object_recognition_status
    object_recognition_status[obj_id] = {
        "name": name,
        "confidence": confidence,
        "timestamp": time.time()
    }
    # Clean up old entries (older than 5 seconds)
    current_time = time.time()
    expired_keys = [k for k, v in object_recognition_status.items() 
                   if current_time - v["timestamp"] > 5.0]
    for k in expired_keys:
        del object_recognition_status[k]

def get_object_recognition(obj_id):
    """Get recognition status for an object"""
    global object_recognition_status
    return object_recognition_status.get(obj_id)

def is_face_recognized(obj_id):
    """Check if a face object ID has been recognized (for display)"""
    return obj_id in recognized_faces

def get_recognized_name(obj_id):
    """Get recognized name for a face object ID (for display)"""
    return recognized_faces.get(obj_id, None)

# ============================================================================
# WORKER THREAD FOR SAVING IMAGES
# ============================================================================
def save_image_worker():
    """Worker thread to save images from queue without blocking main pipeline"""
    while True:
        try:
            data = image_queue.get(timeout=1.0)
            if data is None:  # Exit signal
                print("Image saver thread exiting...")
                break

            img_data, img_path = data
            if img_data is not None and img_data.size > 0:
                try:
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, img_data)
                    print(f"Saved: {img_path}")
                except Exception as e:
                    print(f"Error saving image {img_path}: {e}")
                    traceback.print_exc()

            image_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in save worker: {e}")
            traceback.print_exc()

# Start worker thread
save_thread = threading.Thread(target=save_image_worker, daemon=True)
save_thread.start()

# ============================================================================
# FACE EMBEDDING LOADING FUNCTIONS
# ============================================================================
def load_face_embeddings(embeddings_dir):
    """Load all .npy face embeddings from directory"""
    embeddings = {}
    
    if not os.path.exists(embeddings_dir):
        print(f"Warning: Embeddings directory {embeddings_dir} not found")
        return embeddings
    
    for file in os.listdir(embeddings_dir):
        if file.endswith('.npy'):
            try:
                person_id = os.path.splitext(file)[0]
                embedding = np.load(os.path.join(embeddings_dir, file))
                
                if embedding.ndim == 2:
                    embedding = embedding.flatten()
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings[person_id] = embedding
                print(f"Loaded embedding for: {person_id}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(embeddings)} face embeddings")
    return embeddings

def determine_camera_type(uri, index):
    """Determine if camera is entrance or exit"""
    if "entrance" in uri.lower():
        return "entrance"
    elif "exit" in uri.lower():
        return "exit"
    else:
        return "entrance" if index % 2 == 0 else "exit"

# ============================================================================
# GSTREAMER PROBES
# ============================================================================
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    """Probe on tiler sink - displays recognition results"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        face_count = 0

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                if obj_meta.class_id == PGIE_CLASS_ID_FACE:
                    face_count += 1
                    obj_id = obj_meta.object_id
                    
                    if is_face_recognized(obj_id):
                        # Recognized face - show green border
                        obj_meta.rect_params.border_width = 3
                        obj_meta.rect_params.border_color.red = 0.0
                        obj_meta.rect_params.border_color.green = 1.0
                        obj_meta.rect_params.border_color.blue = 0.0
                        obj_meta.rect_params.border_color.alpha = 1.0
                        obj_meta.rect_params.has_bg_color = 0
                    else:
                        # Unknown face - blur it
                        obj_meta.rect_params.border_width = 0
                        obj_meta.rect_params.has_bg_color = 1
                        obj_meta.rect_params.bg_color.red = 0.0
                        obj_meta.rect_params.bg_color.green = 0.0
                        obj_meta.rect_params.bg_color.blue = 0.0
                        obj_meta.rect_params.bg_color.alpha = 0.7

            except StopIteration:
                break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        stream_id = frame_meta.pad_index
        if perf_data:
            perf_data.update_fps(f"stream{stream_id}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def save_probe_buffer_probe(pad, info, u_data):
    """Probe on CPU buffer branch - saves face crops with recognition info"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    result, map_info = gst_buffer.map(Gst.MapFlags.READ)
    if not result:
        return Gst.PadProbeReturn.OK

    try:
        caps = pad.get_current_caps()
        if not caps:
            gst_buffer.unmap(map_info)
            return Gst.PadProbeReturn.OK

        structure = caps.get_structure(0)
        width = structure.get_int("width").value
        height = structure.get_int("height").value

        # Create numpy array from buffer data
        array = np.ndarray(
            shape=(height, width, 4),
            dtype=np.uint8,
            buffer=map_info.data
        )

        # Convert BGRx to BGR
        bgr_array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

        # Get metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta:
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    stream_id = frame_meta.pad_index

                    if f"stream_{stream_id}" not in saved_count:
                        saved_count[f"stream_{stream_id}"] = 0

                    l_obj = frame_meta.obj_meta_list
                    while l_obj is not None:
                        try:
                            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                            if obj_meta.class_id == PGIE_CLASS_ID_FACE:
                                # Get object ID and check recognition status
                                obj_id = obj_meta.object_id
                                detection_confidence = obj_meta.confidence
                                
                                # Check if this object has been recognized
                                recognition_info = get_object_recognition(obj_id)
                                
                                # Get bounding box
                                rect = obj_meta.rect_params
                                top = int(rect.top)
                                left = int(rect.left)
                                crop_width = int(rect.width)
                                crop_height = int(rect.height)

                                # Add padding
                                padding = 10
                                top = max(0, top - padding)
                                left = max(0, left - padding)
                                bottom = min(height, top + crop_height + 2*padding)
                                right = min(width, left + crop_width + 2*padding)

                                crop_width = right - left
                                crop_height = bottom - top

                                if crop_width > 20 and crop_height > 20:
                                    face_crop = bgr_array[top:bottom, left:right]

                                    if face_crop.size > 0:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                        
                                        if recognition_info:
                                            # This face is RECOGNIZED
                                            recognized_name = recognition_info["name"]
                                            recognition_confidence = recognition_info["confidence"]
                                            img_path = f"{folder_name}/stream_{stream_id}/recognized/{recognized_name}_{timestamp}_detconf_{detection_confidence:.2f}_recogconf_{recognition_confidence:.2f}.jpg"
                                            print(f"✅ Saving RECOGNIZED face: {recognized_name}")
                                        else:
                                            # This face is UNKNOWN
                                            img_path = f"{folder_name}/stream_{stream_id}/unknown/unknown_{timestamp}_conf_{detection_confidence:.2f}.jpg"
                                            print(f"❓ Saving UNKNOWN face")

                                        try:
                                            image_queue.put((face_crop.copy(), img_path), block=False)
                                        except queue.Full:
                                            print(f"Queue full, skipping save")

                        except StopIteration:
                            break

                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break

                    saved_count[f"stream_{stream_id}"] += 1

                except StopIteration:
                    break

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

    except Exception as e:
        print(f"Error in save probe: {e}")
        traceback.print_exc()
    finally:
        gst_buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK

# ============================================================================
# FACE RECOGNITION PROBE (SGIE)
# ============================================================================
def sgie_feature_extract_probe(pad, info, data):
    """ArcFace feature extraction and recognition probe"""
    try:
        if not pad or not info or not data:
            return Gst.PadProbeReturn.OK

        known_face_features = data.get('known_face_features')
        save_feature = data.get('save_feature', False)
        save_path = data.get('save_path')
        sources = data.get('sources', {})

        if not known_face_features:
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
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
                source_key = f"source_{source_id}"
                source_info = sources.get(source_key, {})
                source_name = source_info.get("name", f"Camera {source_id}")
                camera_type = source_info.get("type", "unknown")

                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        if not obj_meta:
                            continue

                        face_feature = get_face_feature(obj_meta, frame_meta.frame_num, data)
                        if face_feature is not None and known_face_features:
                            best_match_id, best_score = match_faces(
                                obj_meta, frame_meta, batch_meta, face_feature, known_face_features
                            )

                            # Process recognition
                            if best_match_id and best_score > 0.3:
                                # IMPORTANT: Update recognition status for this object
                                obj_id = obj_meta.object_id
                                update_object_recognition(obj_id, best_match_id, best_score)
                                recognized_faces[obj_id] = best_match_id
                                
                                # Get current time for logging
                                now = datetime.now()
                                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Print recognition result
                                print(f"✅ [{timestamp}] {source_name}: RECOGNIZED {best_match_id} (Score: {best_score:.2f}, Object ID: {obj_id})")
                                
                                # Handle attendance based on camera type
                                if camera_type == 'entrance':
                                    print(f"   → ENTRANCE: {best_match_id} check-in recorded")
                                elif camera_type == 'exit':
                                    print(f"   → EXIT: {best_match_id} check-out recorded")
                                    
                            else:
                                # Unknown person
                                obj_id = obj_meta.object_id
                                print(f"❓ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {source_name}: Unknown face (Score: {best_score:.2f if best_score > 0 else 0.0})")
                                
                                # Optionally save unknown face features
                                if save_feature and save_path:
                                    save_unknown_face(face_feature, obj_id, frame_meta.frame_num, save_path)

                    except Exception as e:
                        print(f"ERROR in recognition: {str(e)}")
                    finally:
                        l_obj = l_obj.next if hasattr(l_obj, 'next') else None
            except Exception as e:
                print(f"ERROR processing frame: {str(e)}")
            finally:
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
    except Exception as e:
        print(f"CRITICAL ERROR in recognition probe: {str(e)}")
    return Gst.PadProbeReturn.OK

def get_face_feature(obj_meta, frame_num, data):
    """Extract face embedding from tensor metadata"""
    try:
        if not obj_meta or not hasattr(obj_meta, 'obj_user_meta_list'):
            return None

        save_feature = data.get('save_feature', False)
        save_path = data.get('save_path')

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
                            break

                    if len(output) == 512:
                        res = np.reshape(output, (1, -1))
                        norm = np.linalg.norm(res)
                        if norm > 0:
                            normal_array = res / norm
                            
                            if save_feature and save_path:
                                try:
                                    os.makedirs(save_path, exist_ok=True)
                                    save_file = os.path.join(save_path, f"{obj_meta.object_id}-{frame_num}.npy")
                                    np.save(save_file, normal_array)
                                except Exception as e:
                                    pass
                            
                            return normal_array
            except Exception as e:
                pass
            finally:
                l_user_meta = l_user_meta.next if hasattr(l_user_meta, 'next') else None
    except Exception as e:
        pass
    return None

def match_faces(obj_meta, frame_meta, batch_meta, face_feature, loaded_faces):
    """Match face embedding against known faces database"""
    best_match_id = None
    best_score = 0.0

    try:
        if not loaded_faces:
            return None, 0.0

        for user_id, known_feature in loaded_faces.items():
            try:
                if known_feature is None:
                    continue

                score = float(np.dot(face_feature, known_feature).item())

                if score > best_score:
                    best_score = score
                    best_match_id = user_id

            except Exception as e:
                pass

    except Exception as e:
        pass

    return best_match_id, best_score

def save_unknown_face(face_feature, object_id, frame_num, save_path):
    """Save unknown face features for later training"""
    try:
        unknown_dir = os.path.join(save_path, "unknown_faces")
        os.makedirs(unknown_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}_obj{object_id}_frame{frame_num}.npy"
        filepath = os.path.join(unknown_dir, filename)

        np.save(filepath, face_feature)
    except Exception as e:
        pass

# ============================================================================
# SOURCE BIN CREATION FUNCTIONS (keep your existing ones)
# ============================================================================
def create_rtsp_source_bin(index, uri):
    """Create source bin specifically for RTSP sources with optimized settings"""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(f"Unable to create source bin {bin_name}\n")
        return None

    # Create RTSP source element
    rtspsrc = Gst.ElementFactory.make("rtspsrc", f"rtspsrc-{index}")
    if not rtspsrc:
        sys.stderr.write("Unable to create rtspsrc\n")
        return None

    # Configure RTSP source for stability
    rtspsrc.set_property("location", uri)
    rtspsrc.set_property("latency", 0)
    rtspsrc.set_property("buffer-mode", 0)  # Auto
    rtspsrc.set_property("drop-on-latency", True)
    rtspsrc.set_property("do-rtsp-keep-alive", True)
    rtspsrc.set_property("protocols", "tcp")  # Force TCP for stability

    # Create decodebin for RTSP stream
    decodebin = Gst.ElementFactory.make("decodebin", f"decodebin-{index}")
    if not decodebin:
        sys.stderr.write("Unable to create decodebin\n")
        return None

    # Create queue for buffering
    queue_elem = Gst.ElementFactory.make("queue", f"queue-{index}")

    def on_pad_added(element, pad, data):
        """Callback when decodebin adds a new pad (stream)"""
        caps = pad.get_current_caps()
        if caps:
            struct = caps.get_structure(0)
            if struct and struct.get_name().find("video") != -1:
                ghost_pad = data.get_static_pad("src")
                if ghost_pad:
                    if not ghost_pad.set_target(pad):
                        sys.stderr.write("Failed to link pads\n")

    def on_rtspsrc_pad_added(element, pad, data):
        """Callback when rtspsrc adds a new pad"""
        sinkpad = data.get_static_pad("sink")
        if sinkpad:
            pad.link(sinkpad)

    # Connect callbacks
    decodebin.connect("pad-added", on_pad_added, nbin)
    rtspsrc.connect("pad-added", on_rtspsrc_pad_added, decodebin)

    # Add elements to bin
    Gst.Bin.add(nbin, rtspsrc)
    Gst.Bin.add(nbin, decodebin)
    if queue_elem:
        Gst.Bin.add(nbin, queue_elem)
        decodebin.link(queue_elem)

    # Create ghost pad (external interface of the bin)
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost_pad)

    # Set ghost pad target
    if queue_elem:
        srcpad = queue_elem.get_static_pad("src")
        if srcpad:
            ghost_pad.set_target(srcpad)

    return nbin

def create_file_source_bin(index, uri):
    """Create source bin for file sources"""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(f"Unable to create source bin {bin_name}\n")
        return None

    # Create URI decode bin for file sources
    decodebin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not decodebin:
        sys.stderr.write("Unable to create uridecodebin\n")
        return None

    decodebin.set_property("uri", uri)

    def on_pad_added(element, pad, data):
        """Callback when decodebin adds a video pad"""
        caps = pad.get_current_caps()
        if caps:
            struct = caps.get_structure(0)
            if struct and struct.get_name().find("video") != -1:
                ghost_pad = data.get_static_pad("src")
                if ghost_pad:
                    if not ghost_pad.set_target(pad):
                        sys.stderr.write("Failed to link pads\n")

    decodebin.connect("pad-added", on_pad_added, nbin)
    Gst.Bin.add(nbin, decodebin)

    # Create ghost pad
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost_pad)

    return nbin






# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main(uri_inputs, codec, bitrate):
    """Main pipeline function"""
    global perf_data, folder_name, recognized_faces, object_recognition_status

    number_sources = len(uri_inputs)

    # Clear recognition trackers
    recognized_faces.clear()
    object_recognition_status.clear()

    # Initialize performance data
    try:
        from common.FPS import PERF_DATA
        perf_data = PERF_DATA(number_sources)
    except:
        perf_data = None

    # Create output directory structure
    if os.path.exists(folder_name):
        import shutil
        try:
            shutil.rmtree(folder_name)
        except:
            pass
    
    os.makedirs(folder_name, exist_ok=True)
    
    for i in range(number_sources):
        os.makedirs(f"{folder_name}/stream_{i}/recognized", exist_ok=True)
        os.makedirs(f"{folder_name}/stream_{i}/unknown", exist_ok=True)
        saved_count[f"stream_{i}"] = 0

    # Load face embeddings
    embeddings_dir = "/workspace/face_copy/web/data/1"
    known_face_features = load_face_embeddings(embeddings_dir)
    
    # Prepare camera sources information
    sources_info = {}
    for i, uri in enumerate(uri_inputs):
        camera_type = determine_camera_type(uri, i)
        sources_info[f"source_{i}"] = {
            "name": f"Camera_{i}",
            "type": camera_type,
            "uri": uri
        }
        print(f"Camera {i}: {uri} -> Type: {camera_type}")

    # Initialize GStreamer
    Gst.init(None)

    # Create pipeline
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return 1

    print("\n=== Pipeline Setup ===")

    # Create streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")
        return 1
    pipeline.add(streammux)

    # Create source bins
    for i in range(number_sources):
        uri = uri_inputs[i]

        if uri.startswith("rtsp://"):
            source_bin = create_rtsp_source_bin(i, uri)
        else:
            source_bin = create_file_source_bin(i, uri)

        if source_bin:
            pipeline.add(source_bin)
            sinkpad = streammux.request_pad_simple(f"sink_{i}")
            if sinkpad:
                srcpad = source_bin.get_static_pad("src")
                if srcpad:
                    srcpad.link(sinkpad)
                    print(f"Linked source {i} to streammux")

    # Configure streammux
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property('live-source', 1 if any(u.startswith('rtsp://') for u in uri_inputs) else 0)

    # Create PGIE (Face detection)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")
        return 1
    pgie.set_property('config-file-path', "/workspace/config/config_yolo.txt")
    pgie.set_property("batch-size", number_sources)

    # Create SGIE (Face recognition)
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    if not sgie:
        sys.stderr.write("Unable to create sgie\n")
        return 1
    sgie.set_property('config-file-path', "/workspace/config/config_arcface.txt")
    sgie.set_property("batch-size", number_sources)

    # Prepare recognition data
    recognition_data = {
        'known_face_features': known_face_features,
        'save_feature': True,
        'save_path': folder_name,
        'sources': sources_info
    }

    # Create other elements (keep your existing code)
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if filter1:
        filter1.set_property("caps", caps1)

    tee = Gst.ElementFactory.make("tee", "tee")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if tiler:
        tiler.set_property("rows", 1)
        tiler.set_property("columns", 1)
        tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")

    # Save branch
    queue_save = Gst.ElementFactory.make("queue", "queue_save")
    nvvidconv_save = Gst.ElementFactory.make("nvvideoconvert", "convertor_save")
    caps_save = Gst.Caps.from_string("video/x-raw, format=BGRx")
    filter_save = Gst.ElementFactory.make("capsfilter", "filter_save")
    if filter_save:
        filter_save.set_property("caps", caps_save)

    fakesink_save = Gst.ElementFactory.make("fakesink", "fakesink_save")
    if fakesink_save:
        fakesink_save.set_property("sync", False)
        fakesink_save.set_property("async", False)
        fakesink_save.set_property("qos", False)

    # RTSP elements
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    else:
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")

    if encoder:
        encoder.set_property('bitrate', bitrate)

    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if sink:
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', 5400)
        sink.set_property('async', False)
        sink.set_property('sync', False)

    # Add elements to pipeline
    elements = [pgie, sgie, nvvidconv1, filter1, tee, queue_save, nvvidconv_save,
                filter_save, fakesink_save, tiler, nvvidconv, nvosd,
                nvvidconv_postosd, encoder, rtppay, sink]

    for elem in elements:
        if elem:
            pipeline.add(elem)

    # Link pipeline
    try:
        streammux.link(pgie)
        pgie.link(sgie)
        sgie.link(nvvidconv1)
        nvvidconv1.link(filter1)
        filter1.link(tee)

        tee.link(queue_save)
        queue_save.link(nvvidconv_save)
        nvvidconv_save.link(filter_save)
        filter_save.link(fakesink_save)

        tee_src_pad_template = tee.get_pad_template("src_%u")
        tee_src_pad = tee.request_pad(tee_src_pad_template, None, None)
        tiler_sink_pad = tiler.get_static_pad("sink")
        
        if tee_src_pad and tiler_sink_pad:
            tee_src_pad.link(tiler_sink_pad)

        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(nvvidconv_postosd)
        nvvidconv_postosd.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink)

        print("Pipeline linked successfully")
    except Exception as e:
        print(f"Error linking pipeline: {e}")
        return 1

    # Add probes
    sgie_src_pad = sgie.get_static_pad("src")
    if sgie_src_pad:
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, recognition_data)
        print("Added SGIE recognition probe")

    if tiler:
        sink_pad = tiler.get_static_pad("sink")
        if sink_pad:
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
            print("Added tiler display probe")

    if filter_save:
        sink_pad = filter_save.get_static_pad("sink")
        if sink_pad:
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, save_probe_buffer_probe, recognition_data)
            print("Added save probe")

    # Setup RTSP server
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "8554"
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(f'( udpsrc name=pay0 port=5400 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name={codec}, payload=96" )')
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(f"\n=== System Ready ===")
    print(f"RTSP stream: rtsp://localhost:8554/ds-test")
    print(f"Face crops: {folder_name}/")
    print(f"Known faces: {len(known_face_features)}")
    print("===================\n")

    # Start pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message, loop):
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True

    bus.connect("message", on_message, loop)

    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        return 1

    if perf_data:
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            image_queue.put(None, block=False)
        except:
            pass
        pipeline.set_state(Gst.State.NULL)
        save_thread.join(timeout=2.0)

        print("\n=== Summary ===")
        print(f"Recognition status entries: {len(object_recognition_status)}")
        print("================")

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection & Recognition with RTSP')
    parser.add_argument("-i", "--uri_inputs", nargs='+', required=True,
                       help='Input URIs (file:// or rtsp://)')
    parser.add_argument("-c", "--codec", default="H264",
                       choices=['H264', 'H265'],
                       help="Output codec")
    parser.add_argument("-b", "--bitrate", type=int, default=4000000,
                       help="Output bitrate")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.uri_inputs, args.codec, args.bitrate))
