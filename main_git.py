# main.py
import sys
import os
import gi
import math
import traceback
import platform
import multiprocessing

# --- GStreamer and GObject Initialization ---
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GLib, GstRtspServer

# --- Import Utility Functions from your project structure ---
try:
    from utils.probe import pgie_src_filter_probe, attach_sgie_probe, attendance_worker
    from utils.parser_cfg import parse_args, set_property, set_tracker_properties, load_faces
    from utils.bus_call import bus_call
except ImportError as e:
    sys.stderr.write(f"ERROR: Failed to import utility modules. Ensure 'utils' directory and its contents (probe.py, parser_cfg.py, bus_call.py) are present. Error: {e}\n")
    sys.exit(1)

# --- Extra stdlib imports for hot-reload ---
import time
import threading
import hashlib
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# --- Hot-reload helpers (no watchdog) ---
# ==============================================================================
def get_dir_signature(directory):
    """
    Return a hash based on file names + modified times in the directory.
    This changes whenever a file is added, removed, or updated.
    """
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
                hash_md5.update(fname.encode())                 # file name
                hash_md5.update(str(stat.st_mtime).encode())    # modified time
                hash_md5.update(str(stat.st_size).encode())     # size helps catch some FS edge cases
            except FileNotFoundError:
                continue
            except PermissionError:
                continue
    return hash_md5.hexdigest()

def reload_known_faces_if_changed(known_face_dir, loaded_faces, last_signature):
    """
    Reload faces only if directory content has changed.
    Updates the passed-in dict *in place* so any consumer (e.g., SGIE probe) sees new data.
    Also logs a summary of added/removed identities.
    """
    try:
        current_signature = get_dir_signature(known_face_dir)
        if current_signature != last_signature[0]:
            logger.info("[Hot Reload] Change detected in known_face_dir. Reloading...")

            # Keep a copy of the current keys for delta summary
            prev_keys = set(loaded_faces.keys())

            # Re-load using your existing utility
            new_faces = load_faces(known_face_dir) or {}

            # Compute simple deltas (by keys)
            new_keys = set(new_faces.keys())
            added = sorted(list(new_keys - prev_keys))
            removed = sorted(list(prev_keys - new_keys))
            kept = len(new_keys & prev_keys)

            # Update the shared dict IN PLACE so references stay valid
            loaded_faces.clear()
            loaded_faces.update(new_faces)

            last_signature[0] = current_signature

            # Log a clear summary
            logger.info(
                "[Hot Reload] Done. Total in memory: %d | Added: %d%s | Removed: %d%s | Unchanged: %d",
                len(new_faces),
                len(added), f" ({', '.join(added)})" if added else "",
                len(removed), f" ({', '.join(removed)})" if removed else "",
                kept,
            )
    except Exception as e:
        logger.error(f"[Hot Reload] Error reloading faces: {e}\n{traceback.format_exc()}")

def start_face_reloader(known_face_dir, loaded_faces, interval=5):
    """
    Starts a daemon thread that checks the directory every `interval` seconds
    and triggers a reload when it detects a change.
    """
    last_signature = [get_dir_signature(known_face_dir)]  # mutable holder

    def reloader():
        while True:
            reload_known_faces_if_changed(known_face_dir, loaded_faces, last_signature)
            time.sleep(interval)

    t = threading.Thread(target=reloader, daemon=True, name="FaceReloader")
    t.start()
    return t

# ==============================================================================
# --- Logging Functions ---
# ==============================================================================
def log_error(message):
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stdout.flush()

def log_info(message):
    print(f"INFO: {message}")
    sys.stdout.flush()

# ==============================================================================
# --- RTSP Server Classes (unchanged) ---
# ==============================================================================
class MyRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, framerate=25, codec="H265", udpsink_port=5400, **properties):
        super().__init__(**properties)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.codec = codec
        self.udpsink_port = udpsink_port
        self.set_shared(True)

    def do_create_element(self, url):
        log_info(f"RTSPMediaFactory: Creating new element for URL: {url.get_request_uri()}")
        if self.codec == "H264":
            rtp_caps = "application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96"
            pipeline_str = (f"udpsrc name=mysource port={self.udpsink_port} buffer-size=524288 caps=\"{rtp_caps}\" ! "
                            f"rtph264depay ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1")
        else:
            rtp_caps = "application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H265, payload=96"
            pipeline_str = (f"udpsrc name=mysource port={self.udpsink_port} buffer-size=524288 caps=\"{rtp_caps}\" ! "
                            f"rtph265depay ! h265parse ! rtph265pay name=pay0 pt=96 config-interval=1")
        rtsp_pipeline_bin = Gst.parse_launch(pipeline_str)
        if not rtsp_pipeline_bin:
            log_error("Failed to parse RTSP media pipeline launch string.")
            return None
        log_info("RTSP Media Factory element created successfully.")
        return rtsp_pipeline_bin

class MyRTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super().__init__(**properties)
        self.is_reconnecting = False

    def check_and_reconnect(self):
        if self.is_reconnecting:
            return True
        is_attached = self.get_session_pool() is not None
        if not is_attached:
            log_error("RTSP server not attached to the main loop. Attempting to restart...")
            self.is_reconnecting = True
            try:
                self.attach(None)
                log_info("RTSP server successfully re-attached to the main loop.")
            except Exception as e:
                log_error(f"Failed to re-attach RTSP server: {e}\n{traceback.format_exc()}")
            finally:
                self.is_reconnecting = False
        return True

# ==============================================================================
# --- Source bin helpers (unchanged) ---
# ==============================================================================
def cb_newpad(decodebin, decoder_src_pad, data):
    try:
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            caps = decoder_src_pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)
        if "video" in gstname.lower():
            if features.contains("memory:NVMM"):
                log_info(f"Linking 'uridecodebin' pad with NVMM-capable caps: {caps.to_string()}")
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad:
                    log_error("Failed to get source bin ghost pad")
                    return
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    log_error("Failed to link decoder src pad to source bin ghost pad")
            else:
                log_error("Decodebin did not pick nvidia decoder plugin - no NVMM memory feature.")
    except Exception as e:
        log_error(f"Exception in cb_newpad: {str(e)}\n{traceback.format_exc()}")

def decodebin_child_added(child_proxy, obj, name, user_data):
    if "decodebin" in name:
        obj.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        obj.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    try:
        log_info(f"Creating source bin {index} for URI: {uri}")
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        if not uri_decode_bin or not nbin:
            log_error(f"Unable to create uridecodebin or source bin {index}")
            return None
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            log_error("Failed to add ghost pad in source bin")
            return None
        return nbin
    except Exception as e:
        log_error(f"Exception in create_source_bin: {str(e)}\n{traceback.format_exc()}")
        return None

# ==============================================================================
# --- Main Application ---
# ==============================================================================
def main(cfg):
    # Ensure multiprocessing start method is set before creating processes
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # already set
        pass

    GObject.threads_init()
    Gst.init(None)

    log_info("DeepStream Face Recognition and RTSP Streaming Pipeline")

    # --- Load known faces safely ---
    try:
        known_face_dir = cfg['pipeline']['known_face_dir']
        log_info(f"Checking configured known faces directory: '{known_face_dir}'")
        if not os.path.isdir(known_face_dir):
            log_error(f"The configured known_face_dir '{known_face_dir}' does not exist.")
            return 1
        if not os.access(known_face_dir, os.R_OK):
            log_error(f"The configured known_face_dir '{known_face_dir}' exists but is not readable.")
            return 1
        file_count = len(os.listdir(known_face_dir))
        log_info(f"Found {file_count} files/directories in '{known_face_dir}'.")
        if file_count == 0:
            log_error(f"The configured known_face_dir '{known_face_dir}' is empty. Please add face data.")
            # continue running but recognition will be disabled

        # Initial load
        known_face_features = load_faces(known_face_dir) or {}
        log_info(f"Loaded {len(known_face_features)} known face features.")

    except Exception as e:
        log_error(f"Failed to load known faces: {e}\n{traceback.format_exc()}")
        return 1

    save_feature = cfg['pipeline'].get('save_feature', 0)
    save_path = cfg['pipeline'].get('save_feature_path') if save_feature else None

    # Start attendance worker process (multiprocessing.Queue)
    attendance_queue = multiprocessing.Queue()
    attendance_proc = multiprocessing.Process(target=attendance_worker, args=(attendance_queue,))
    attendance_proc.start()
    log_info(f"Attendance worker process started (pid={attendance_proc.pid})")

    # --- Create GStreamer Elements ---
    pipeline = Gst.Pipeline.new("deepstream-pipeline")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvvidconv_sgie = Gst.ElementFactory.make("nvvideoconvert", "sgie-vid-converter")
    capsfilter_sgie = Gst.ElementFactory.make("capsfilter", "sgie-capsfilter")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv_main = Gst.ElementFactory.make("nvvideoconvert", "main-vid-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    tee = Gst.ElementFactory.make("tee", "output-tee")

    elements = [pipeline, streammux, pgie, tracker, nvvidconv_sgie, capsfilter_sgie, sgie, tiler, nvvidconv_main, nvosd, tee]
    if any(e is None for e in elements):
        log_error("Failed to create one or more essential GStreamer elements. Exiting.")
        # shutdown attendance worker
        attendance_queue.put(None); attendance_proc.join(timeout=5)
        return 1

    # --- Add sources and streammux config ---
    pipeline.add(streammux)
    try:
        sources = cfg['sources']
    except KeyError:
        log_error("Configuration file missing 'sources'.")
        attendance_queue.put(None); attendance_proc.join(timeout=5)
        return 1

    is_live = False
    for i, source_dict in enumerate(sources):
        uri = source_dict.get('uri')
        if not uri:
            log_error(f"Source {i} missing URI. Skipping.")
            continue
        if "rtsp://" in uri or "rtspt://" in uri or "rtmp://" in uri:
            is_live = True
        source_bin = create_source_bin(i, uri)
        capsfilter_src = Gst.ElementFactory.make("capsfilter", f"capsfilter_src_{i}")
        nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv_src_{i}")
        if not all([source_bin, nvvidconv_src, capsfilter_src]):
            log_error(f"Failed to create source elements for {i}. Skipping.")
            continue
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        capsfilter_src.set_property("caps", caps)
        pipeline.add(source_bin); pipeline.add(capsfilter_src); pipeline.add(nvvidconv_src)
        srcpad = source_bin.get_static_pad("src")
        sinkpad = nvvidconv_src.get_static_pad("sink")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            log_error(f"Failed to link source_bin {i} to nvvidconv_src.")
            continue
        if not nvvidconv_src.link(capsfilter_src):
            log_error(f"Failed to link nvvidconv_src to capsfilter_src for source {i}.")
            continue
        sinkpad_mux = streammux.get_request_pad(f"sink_{i}")
        srcpad_caps = capsfilter_src.get_static_pad("src")
        if srcpad_caps.link(sinkpad_mux) != Gst.PadLinkReturn.OK:
            log_error(f"Failed to link capsfilter_src to streammux for source {i}.")
            continue
        log_info(f"Source {i} ({uri}) added & linked to streammux.")

    streammux.set_property('live-source', is_live)
    set_property(cfg, streammux, "streammux")

    # Configure elements
    set_property(cfg, pgie, "pgie")
    set_property(cfg, sgie, "sgie")
    set_property(cfg, tiler, "tiler")
    set_property(cfg, nvosd, "nvosd")
    set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

    main_elements = [pgie, tracker, nvvidconv_sgie, capsfilter_sgie, sgie, tiler, nvvidconv_main, nvosd, tee]
    for el in main_elements:
        pipeline.add(el)

    # Link main pipeline
    if not streammux.link(pgie): log_error("Failed to link streammux->pgie"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not pgie.link(tracker): log_error("Failed to link pgie->tracker"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not tracker.link(nvvidconv_sgie): log_error("Failed to link tracker->nvvidconv_sgie"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not nvvidconv_sgie.link(capsfilter_sgie): log_error("Failed to link nvvidconv_sgie->capsfilter_sgie"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not capsfilter_sgie.link(sgie): log_error("Failed to link capsfilter_sgie->sgie"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not sgie.link(tiler): log_error("Failed to link sgie->tiler"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not tiler.link(nvvidconv_main): log_error("Failed to link tiler->nvvidconv_main"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not nvvidconv_main.link(nvosd): log_error("Failed to link nvvidconv_main->nvosd"); attendance_queue.put(None); attendance_proc.join(); return 1
    if not nvosd.link(tee): log_error("Failed to link nvosd->tee"); attendance_queue.put(None); attendance_proc.join(); return 1

    # Display / fakesink branch
    display_queue = Gst.ElementFactory.make("queue", "display_queue")
    pipeline.add(display_queue)
    tee_display_pad = tee.get_request_pad("src_%u")
    if not tee_display_pad:
        log_error("Failed to get tee src pad for display branch"); attendance_queue.put(None); attendance_proc.join(); return 1
    if tee_display_pad.link(display_queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        log_error("Failed to link tee to display_queue"); attendance_queue.put(None); attendance_proc.join(); return 1
    if cfg['pipeline']['display']:
        display_sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        display_sink = Gst.ElementFactory.make("fakesink", "fakesink")
    display_sink.set_property("sync", False)
    pipeline.add(display_sink)
    if not display_queue.link(display_sink):
        log_error("Failed to link display_queue to sink"); attendance_queue.put(None); attendance_proc.join(); return 1

    # RTSP branch (if enabled)
    if cfg['pipeline']['enable_rtsp_streaming']:
        rtsp_queue = Gst.ElementFactory.make("queue", "rtsp_queue")
        rtsp_queue.set_property("leaky", 2)
        rtsp_queue.set_property("max-size-buffers", 50)
        rtsp_queue.set_property("max-size-bytes", 0)
        rtsp_queue.set_property("max-size-time", 0)
        rtsp_conv = Gst.ElementFactory.make("nvvideoconvert", "rtsp-converter")
        codec = cfg['rtsp_server'].get('codec', 'H265')
        if codec == "H264":
            rtsp_encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
            rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        else:
            rtsp_encoder = Gst.ElementFactory.make("nvv4l2h265enc", "h265-encoder")
            rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        updsink_port_num = cfg['rtsp_server'].get('udpsink-port', 5400)
        udpsink_host = cfg['rtsp_server'].get('udpsink-host', "127.0.0.1")
        udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
        for el in [rtsp_queue, rtsp_conv, rtsp_encoder, rtppay, udpsink]:
            pipeline.add(el)
        rtppay.set_property("pt",96)
        udpsink.set_property("host", udpsink_host)
        udpsink.set_property("port", updsink_port_num)
        udpsink.set_property("async", False)
        udpsink.set_property("sync", False)
        tee_rtsp_pad = tee.get_request_pad("src_%u")
        if not tee_rtsp_pad:
            log_error("Failed to get tee src pad for RTSP branch"); attendance_queue.put(None); attendance_proc.join(); return 1
        if tee_rtsp_pad.link(rtsp_queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            log_error("Failed to link tee to rtsp_queue"); attendance_queue.put(None); attendance_proc.join(); return 1
        if not rtsp_queue.link(rtsp_conv): log_error("Failed to link rtsp_queue to rtsp_conv"); attendance_queue.put(None); attendance_proc.join(); return 1
        if not rtsp_conv.link(rtsp_encoder): log_error("Failed to link rtsp_conv to rtsp_encoder"); attendance_queue.put(None); attendance_proc.join(); return 1
        if not rtsp_encoder.link(rtppay): log_error("Failed to link rtsp_encoder to rtppay"); attendance_queue.put(None); attendance_proc.join(); return 1
        if not rtppay.link(udpsink): log_error("Failed to link rtppay to udpsink"); attendance_queue.put(None); attendance_proc.join(); return 1

        server = MyRTSPServer()
        rtsp_server_port = str(cfg['rtsp_server'].get('port', 8555))
        server.set_service(rtsp_server_port)
        mount_points = server.get_mount_points()
        factory = MyRTSPMediaFactory(
            width=cfg['rtsp_server'].get('width', 1280),
            height=cfg['rtsp_server'].get('height', 720),
            framerate=cfg['rtsp_server'].get('framerate', 25),
            codec=codec,
            udpsink_port=updsink_port_num
        )
        factory.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)
        mount_point_path = cfg['rtsp_server'].get('mount-point', '/mystream')
        mount_points.add_factory(mount_point_path, factory)
        server.attach(None)
        GLib.timeout_add_seconds(5, server.check_and_reconnect)
        log_info(f"RTSP stream available at rtsp://<host-ip>:{rtsp_server_port}{mount_point_path}")

    # --- Attach probes ---
    log_info("Attaching buffer probes for custom logic...")
    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)
        log_info("PGIE src pad probe attached.")
    else:
        log_error("Could not get pgie src pad for probe. PGIE probe not attached.")

    # Attach SGIE probe using helper that bundles user_data safely
    attach_sgie_probe(sgie, known_face_features, attendance_queue, sources)

    # --- Start hot reloader (non-blocking) ---
    reload_interval = int(cfg['pipeline'].get('reload_interval', 5))  # seconds
    start_face_reloader(known_face_dir, known_face_features, interval=reload_interval)
    log_info(f"Hot reloader started (interval={reload_interval}s). Changes to '{known_face_dir}' will be picked up live.")

    # --- Start main loop ---
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    log_info("Pipeline set to PLAYING.")

    try:
        loop.run()
    except KeyboardInterrupt:
        log_info("Keyboard interrupt received. Stopping pipeline.")
    except Exception as e:
        log_error(f"Exception in main loop: {e}\n{traceback.format_exc()}")
    finally:
        log_info("Stopping pipeline and cleaning up...")
        pipeline.set_state(Gst.State.NULL)

        # cleanly stop attendance worker
        try:
            attendance_queue.put(None)
            attendance_proc.join(timeout=5)
            if attendance_proc.is_alive():
                log_info("Attendance worker didn't exit in time; terminating.")
                attendance_proc.terminate()
                attendance_proc.join(timeout=2)
        except Exception:
            traceback.print_exc()

    return 0

if __name__ == '__main__':
    try:
        cfg = parse_args(cfg_path="config/config_pipeline.toml")
        sys.exit(main(cfg))
    except Exception as e:
        log_error(f"Fatal error in __main__: {e}\n{traceback.format_exc()}")
        sys.exit(1)
