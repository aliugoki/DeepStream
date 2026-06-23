import sys
import os
import gi
import math
import traceback
import platform

# --- GStreamer and GObject Initialization ---
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GLib, GstRtspServer

# --- Import Utility Functions from your project structure ---
# Note: These files (probe.py, parser_cfg.py, bus_call.py) must exist in a 'utils' directory.
from utils.probe import pgie_src_filter_probe, sgie_feature_extract_probe
from utils.parser_cfg import parse_args, set_property, set_tracker_properties, load_faces
from utils.bus_call import bus_call

# ==============================================================================
# --- Logging Functions ---
# ==============================================================================

def log_error(message):
    """Logs an error message to stderr."""
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stderr.flush()

def log_info(message):
    """Logs an informational message to stdout."""
    print(f"INFO: {message}")
    sys.stdout.flush()

# ==============================================================================
# --- RTSP Server Components (Merged from original script) ---
# ==============================================================================

# Global variables for FPS calculation in the RTSP branch
last_appsink_timestamp = 0
frame_count_appsink = 0

class MyRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    """Custom RTSP Media Factory to handle pushing buffers from the main pipeline."""
    def __init__(self, width, height, framerate=30, **properties):
        super().__init__(**properties)
        self.rtsp_appsrc = None
        self.width = width
        self.height = height
        self.framerate = framerate
        self.set_shared(True)

    def do_create_element(self, url):
        """Creates the GStreamer element that consumes buffers and serves them via RTSP."""
        log_info(f"RTSPMediaFactory: Creating new element for URL: {url.get_request_uri()}")
        # Pipeline for the RTSP client. It starts with an appsrc to receive H.265 buffers.
        pipeline_str = (            
            f"appsrc name=mysource is-live=true block=true format=time max-bytes=0 ! video/x-h265,width={self.width},height={self.height},framerate={self.framerate}/1,stream-format=byte-stream,alignment=au ! queue ! h265parse ! rtph265pay name=pay0 pt=96 config-interval=1"            
        )
        rtsp_pipeline_bin = Gst.parse_launch(pipeline_str)
        if not rtsp_pipeline_bin:
            log_error("Failed to parse RTSP media pipeline launch string.")
            return None

        appsrc_element = rtsp_pipeline_bin.get_by_name("mysource")
        if not appsrc_element:
            log_error("Failed to find 'mysource' appsrc in RTSP media pipeline.")
            return None

        # Store a reference to the appsrc so we can push buffers to it.
        self.rtsp_appsrc = appsrc_element
        log_info("RTSP Media Factory element created successfully.")
        return rtsp_pipeline_bin

def appsink_new_sample_probe(sink, factory):
    """Callback for the 'new-sample' signal from the appsink in the RTSP branch."""
    global last_appsink_timestamp, frame_count_appsink

    sample = sink.emit("pull-sample")
    if sample:
        gst_buffer = sample.get_buffer()
        if not gst_buffer:
            return Gst.FlowReturn.OK

        # --- FPS Calculation for the encoded stream ---
        current_time_ns = GLib.get_monotonic_time() * 1000
        if last_appsink_timestamp == 0:
            last_appsink_timestamp = current_time_ns

        frame_count_appsink += 1
        elapsed_ns = current_time_ns - last_appsink_timestamp
        if elapsed_ns >= 1_000_000_000: # Every 1 second
            fps = frame_count_appsink / (elapsed_ns / 1_000_000_000)
            log_info(f"RTSP APPSINK (Encoded) FPS: {fps:.2f}")
            last_appsink_timestamp = current_time_ns
            frame_count_appsink = 0                       

        # Push the buffer to the RTSP server's appsrc if it exists
        if factory and factory.rtsp_appsrc:
            ret = factory.rtsp_appsrc.emit("push-buffer", gst_buffer)
            if ret != Gst.FlowReturn.OK:
                log_error(f"Failed to push buffer to RTSP appsrc: {ret.value_name}")

        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR


# ==============================================================================
# --- Source Bin Creation (From new script, using uridecodebin) ---
# ==============================================================================

def cb_newpad(decodebin, decoder_src_pad, data):
    """Callback to link `uridecodebin`'s newly created source pad to the source bin's ghost pad."""
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
                log_error("Decodebin did not pick nvidia decoder plugin - no NVMM memory feature")
    except Exception as e:
        log_error(f"Exception in cb_newpad: {str(e)}\n{traceback.format_exc()}")

def decodebin_child_added(child_proxy, obj, name, user_data):
    """Set properties on child elements of uridecodebin."""
    if "decodebin" in name:
        obj.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        # For RTSP sources, dropping frames on latency helps prevent the pipeline from getting stuck.
        obj.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    """Creates a Gst.Bin containing a uridecodebin for a given URI."""
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
    """Main function to build and run the GStreamer pipeline."""
    # It's crucial to initialize GObject threads for server applications
    GObject.threads_init()
    Gst.init(None)

    log_info("DeepStream Face Recognition and RTSP Streaming Pipeline")

    # --- Load Application-Specific Data ---
    try:
        faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recognized_faces")
        os.makedirs(faces_dir, exist_ok=True)
        known_face_features = load_faces(cfg['pipeline']['known_face_dir'])
        log_info(f"Loaded {len(known_face_features)} known face features")
    except Exception as e:
        log_error(f"Failed to load known faces: {e}\n{traceback.format_exc()}")
        return

    save_feature = cfg['pipeline'].get('save_feature', 0)
    save_path = cfg['pipeline'].get('save_feature_path') if save_feature else None

    # --- Create GStreamer Elements ---
    pipeline = Gst.Pipeline.new("deepstream-face-reco-pipeline")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "main-vid-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    tee = Gst.ElementFactory.make("tee", "output-tee")

    elements = [pipeline, streammux, pgie, tracker, sgie, tiler, nvvidconv, nvosd, tee]
    if any(e is None for e in elements):
        log_error("Failed to create one or more essential GStreamer elements.")
        return

    # --- Add Sources and Configure StreamMux ---
    pipeline.add(streammux)
    sources = cfg['source']
    is_live = False
    for i, (k, v) in enumerate(sources.items()):
        if "rtsp://" in v: is_live = True
        source_bin = create_source_bin(i, v)
        if source_bin:
            pipeline.add(source_bin)
            sinkpad = streammux.get_request_pad(f"sink_{i}")
            srcpad = source_bin.get_static_pad("src")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                log_error(f"Failed to link source bin {i} to streammux.")

    streammux.set_property('live-source', is_live)
    set_property(cfg, streammux, "streammux")

    # --- Configure Elements ---
    set_property(cfg, pgie, "pgie")
    set_property(cfg, sgie, "sgie")
    set_property(cfg, tiler, "tiler")
    set_property(cfg, nvosd, "nvosd")
    set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

    # --- Add and Link Main Pipeline Elements ---
    main_elements = [pgie, tracker, sgie, tiler, nvvidconv, nvosd, tee]
    for el in main_elements:
        pipeline.add(el)

    log_info("Linking main pipeline elements: streammux -> pgie -> tracker -> ... -> tee")
    if not streammux.link(pgie): log_error("Failed to link streammux to pgie"); return
    if not pgie.link(tracker): log_error("Failed to link pgie to tracker"); return
    if not tracker.link(sgie): log_error("Failed to link tracker to sgie"); return
    if not sgie.link(tiler): log_error("Failed to link sgie to tiler"); return
    if not tiler.link(nvvidconv): log_error("Failed to link tiler to nvvidconv"); return
    if not nvvidconv.link(nvosd): log_error("Failed to link nvvidconv to nvosd"); return
    if not nvosd.link(tee): log_error("Failed to link nvosd to tee"); return

    # --- Create Output Branches from Tee ---
    # Branch 1: Display or Fake Sink
    log_info("Creating display/fakesink branch...")
    display_queue = Gst.ElementFactory.make("queue", "display_queue")
    pipeline.add(display_queue)
    if not tee.link(display_queue): log_error("Failed to link tee to display_queue"); return

    if cfg['pipeline']['display']:
        log_info("Creating display sink (nveglglessink)...")
        display_sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        log_info("Creating fakesink...")
        display_sink = Gst.ElementFactory.make("fakesink", "fakesink")

    if not display_sink: log_error("Failed to create display/fake sink"); return
    display_sink.set_property("sync", False)
    pipeline.add(display_sink)
    if not display_queue.link(display_sink): log_error("Failed to link display_queue to sink"); return

    # Branch 2: RTSP Streaming
    if cfg['pipeline']['enable_rtsp_streaming']:
        log_info("Setting up RTSP streaming branch...")
        rtsp_queue = Gst.ElementFactory.make("queue", "rtsp_queue")
        rtsp_conv = Gst.ElementFactory.make("nvvideoconvert", "rtsp-converter")
        rtsp_caps = Gst.ElementFactory.make("capsfilter", "rtsp-caps")
        rtsp_encoder = Gst.ElementFactory.make("nvv4l2h265enc", "h265-encoder")
        rtsp_parser = Gst.ElementFactory.make("h265parse", "rtsp-parser")
        rtsp_parser.set_property("config-interval", 1)
        rtsp_appsink = Gst.ElementFactory.make("appsink", "rtsp-appsink")

        rtsp_elements = [rtsp_queue, rtsp_conv, rtsp_caps, rtsp_encoder, rtsp_appsink]
        if any(e is None for e in rtsp_elements): log_error("Failed to create RTSP branch elements"); return
        
        # Configure RTSP elements
        
        rtsp_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12, width=1280, height=720, framerate=25/1"
            )
        )

        rtsp_encoder.set_property("bitrate", cfg['rtsp_server'].get('bitrate', 4000000))
        rtsp_encoder.set_property("tuning-info-id", 1) # 1: HighQualityPreset (from default 2: LowLatencyPreset)
        rtsp_encoder.set_property("control-rate", 0) # 0: variable_bitrate (from default 1: constant_bitrate)
        rtsp_encoder.set_property("cq", 25) # Target quality for VBR. Experiment, lower is higher quality (e.g., 20-30).
        rtsp_encoder.set_property("maxbitrate", rtsp_encoder.get_property("bitrate") * 2)
        rtsp_encoder.set_property("profile", 0) # Main profile
        rtsp_encoder.set_property("idrinterval", 25) 
        #rtsp_encoder.set_property("preset-level", 1) 
        #rtsp_encoder.set_property("insert-sps-pps", 1)         
        rtsp_encoder.set_property("preset-id", 4)
        rtsp_encoder.set_property("aq", 10) # Spatial AQ strength (0-15, 0=auto). Experiment.
        rtsp_encoder.set_property("temporalaq", True) 
        rtsp_encoder.set_property("iframeinterval", 30) # 1 keyframe per second for 25 FPS                        
        rtsp_encoder.set_property("vbvbufsize", rtsp_encoder.get_property("bitrate") * 1.2)
        rtsp_encoder.set_property("vbvinit", rtsp_encoder.get_property("vbvbufsize") * 0.9)
        rtsp_appsink.set_property("emit-signals", True)
        rtsp_appsink.set_property("sync", False)
        rtsp_appsink.set_property("max-buffers", 1)
        rtsp_appsink.set_property("drop", True)
        

        
        for el in rtsp_elements: pipeline.add(el)
        
        log_info("Linking RTSP branch: tee -> queue -> ... -> appsink")
        if not tee.link(rtsp_queue): log_error("Failed to link tee to rtsp_queue"); return
        if not rtsp_queue.link(rtsp_conv): log_error("Failed to link rtsp_queue to rtsp_conv"); return
        if not rtsp_conv.link(rtsp_caps): log_error("Failed to link rtsp_conv to rtsp_caps"); return
        if not rtsp_caps.link(rtsp_encoder): log_error("Failed to link rtsp_caps to rtsp_encoder"); return
        if not rtsp_encoder.link(rtsp_appsink): log_error("Failed to link rtsp_encoder to rtsp_appsink"); return

        # Setup and start the RTSP Server
        server = GstRtspServer.RTSPServer.new()
        port = str(cfg['rtsp_server'].get('port', 8555))
        server.set_service(port)
        
        mount_points = server.get_mount_points()
        factory = MyRTSPMediaFactory(
            width=cfg['streammux'].get('width', 1920), 
            height=cfg['streammux'].get('height', 1080)
        )
        
        factory.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)        
        
        mount_points.add_factory(cfg['rtsp_server'].get('mount-point', '/mystream'), factory)
        server.attach(None)
        
        rtsp_appsink.connect("new-sample", appsink_new_sample_probe, factory)
        log_info(f"RTSP stream available at rtsp://<host-ip>:{port}{cfg['rtsp_server'].get('mount-point', '/mystream')}")


    # --- Attach Probes for Custom Logic ---
    log_info("Attaching buffer probes for custom logic...")
    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)
    else: log_error("Could not get pgie src pad for probe")

    sgie_src_pad = sgie.get_static_pad("src")
    if sgie_src_pad:
        probe_data = [known_face_features, save_feature, save_path]
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, probe_data)
    else: log_error("Could not get sgie src pad for probe")

    # --- Start Pipeline and Run Main Loop ---
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    log_info("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        log_info("Keyboard interrupt received.")
    except Exception as e:
        log_error(f"Exception in main loop: {e}\n{traceback.format_exc()}")
    finally:
        log_info("Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        log_info("Pipeline stopped.")


if __name__ == '__main__':
    try:
        # Assumes config file is at 'config/config_pipeline.toml' relative to the script
        cfg = parse_args(cfg_path="config/config_pipeline.toml")
        sys.exit(main(cfg))
    except Exception as e:
        log_error(f"Fatal error in __main__: {e}\n{traceback.format_exc()}")
        sys.exit(1)