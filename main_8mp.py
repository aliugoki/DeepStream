import sys
import os
import gi
import math
import traceback
import platform

# --- GStreamer and GObject Initialization ---
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# --- Import Utility Functions from your project structure ---
from utils.probe import pgie_src_filter_probe, sgie_feature_extract_probe
from utils.parser_cfg import parse_args, set_property, set_tracker_properties, load_faces
from utils.bus_call import bus_call

# --- Logging Functions ---
def log_error(message):
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stderr.flush()

def log_info(message):
    print(f"INFO: {message}")
    sys.stdout.flush()

# --- RTSP Server Components ---
last_appsink_timestamp = 0
frame_count_appsink = 0

class MyRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, framerate=30, **properties):
        super().__init__(**properties)
        self.rtsp_appsrc = None
        self.width = width
        self.height = height
        self.framerate = framerate
        self.set_shared(True)

    def do_create_element(self, url):
        log_info(f"RTSPMediaFactory: Creating new element for URL: {url.get_request_uri()}")
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

        self.rtsp_appsrc = appsrc_element
        log_info("RTSP Media Factory element created successfully.")
        return rtsp_pipeline_bin

def appsink_new_sample_probe(sink, factory):
    global last_appsink_timestamp, frame_count_appsink

    sample = sink.emit("pull-sample")
    if sample:
        gst_buffer = sample.get_buffer()
        if not gst_buffer:
            return Gst.FlowReturn.OK

        current_time_ns = GLib.get_monotonic_time() * 1000
        if last_appsink_timestamp == 0:
            last_appsink_timestamp = current_time_ns

        frame_count_appsink += 1
        elapsed_ns = current_time_ns - last_appsink_timestamp
        if elapsed_ns >= 1_000_000_000:
            fps = frame_count_appsink / (elapsed_ns / 1_000_000_000)
            log_info(f"RTSP APPSINK (Encoded) FPS: {fps:.2f}")
            last_appsink_timestamp = current_time_ns
            frame_count_appsink = 0

        if factory and factory.rtsp_appsrc:
            ret = factory.rtsp_appsrc.emit("push-buffer", gst_buffer)
            if ret != Gst.FlowReturn.OK:
                log_error(f"Failed to push buffer to RTSP appsrc: {ret.value_name}")

        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR

# --- Source Bin Creation ---
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
                log_error("Decodebin did not pick nvidia decoder plugin - no NVMM memory feature")
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

# --- Main Application ---
def main(cfg):
    Gst.init(None)
    log_info("DeepStream Face Recognition and RTSP Streaming Pipeline")

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

    pipeline = Gst.Pipeline.new("deepstream-face-reco-pipeline")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "main-vid-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    tee = Gst.ElementFactory.make("tee", "output-tee")
    tracker_sgie_queue = Gst.ElementFactory.make("queue", "tracker_sgie_queue")

    elements = [streammux, pgie, tracker, tracker_sgie_queue, sgie, tiler, nvvidconv, nvosd, tee]
    if any(e is None for e in elements):
        log_error("Failed to create one or more essential GStreamer elements.")
        return

    for e in elements:
        pipeline.add(e)

    sources = cfg['source']
    is_live = any("rtsp://" in v for v in sources.values())
    for i, (k, v) in enumerate(sources.items()):
        source_bin = create_source_bin(i, v)
        if source_bin:
            pipeline.add(source_bin)
            sinkpad = streammux.get_request_pad(f"sink_{i}")
            srcpad = source_bin.get_static_pad("src")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                log_error(f"Failed to link source bin {i} to streammux.")

    streammux.set_property('live-source', is_live)
    set_property(cfg, streammux, "streammux")
    set_property(cfg, pgie, "pgie")
    set_property(cfg, sgie, "sgie")
    set_property(cfg, tiler, "tiler")
    set_property(cfg, nvosd, "nvosd")
    set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

    log_info("Linking main pipeline elements: streammux -> pgie -> tracker -> queue -> sgie -> ... -> tee")
    if not streammux.link(pgie): log_error("Failed to link streammux to pgie"); return
    if not pgie.link(tracker): log_error("Failed to link pgie to tracker"); return
    if not tracker.link(tracker_sgie_queue): log_error("Failed to link tracker to queue"); return
    if not tracker_sgie_queue.link(sgie): log_error("Failed to link queue to sgie"); return
    if not sgie.link(tiler): log_error("Failed to link sgie to tiler"); return
    if not tiler.link(nvvidconv): log_error("Failed to link tiler to nvvidconv"); return
    if not nvvidconv.link(nvosd): log_error("Failed to link nvvidconv to nvosd"); return
    if not nvosd.link(tee): log_error("Failed to link nvosd to tee"); return

    # --- Display branch ---
    display_queue = Gst.ElementFactory.make("queue", "display_queue")
    display_sink = Gst.ElementFactory.make("fakesink", "fakesink")
    display_sink.set_property("sync", False)
    pipeline.add(display_queue)
    pipeline.add(display_sink)
    tee.link(display_queue)
    display_queue.link(display_sink)

    # --- RTSP branch ---
    rtsp_queue = Gst.ElementFactory.make("queue", "rtsp_queue")
    rtsp_convert = Gst.ElementFactory.make("nvvideoconvert", "rtsp_convert")
    rtsp_caps = Gst.ElementFactory.make("capsfilter", "rtsp_caps")
    rtsp_encoder = Gst.ElementFactory.make("nvv4l2h265enc", "rtsp_encoder")
    rtsp_appsink = Gst.ElementFactory.make("appsink", "rtsp_appsink")
    
    rtsp_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, width=1280, height=720, framerate=25/1"))
    rtsp_encoder.set_property("bitrate", 4000000)
    rtsp_appsink.set_property("emit-signals", True)
    rtsp_appsink.set_property("sync", False)
    
    for el in [rtsp_queue, rtsp_convert, rtsp_caps, rtsp_encoder, rtsp_appsink]:
        pipeline.add(el)

    tee.link(rtsp_queue)
    rtsp_queue.link(rtsp_convert)
    rtsp_convert.link(rtsp_caps)
    rtsp_caps.link(rtsp_encoder)
    rtsp_encoder.link(rtsp_appsink)

    factory = MyRTSPMediaFactory(width=1280, height=720)
    rtsp_appsink.connect("new-sample", appsink_new_sample_probe, factory)
    server = GstRtspServer.RTSPServer.new()
    server.set_service("8554")
    mount_points = server.get_mount_points()
    mount_points.add_factory("/ds-stream", factory)
    server.attach(None)
    log_info("RTSP stream available at rtsp://<host-ip>:8554/ds-stream")

    # --- Probes ---
    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)

    sgie_src_pad = sgie.get_static_pad("src")
    if sgie_src_pad:
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, [known_face_features, save_feature, save_path])

    # --- Main Loop ---
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
    finally:
        pipeline.set_state(Gst.State.NULL)
        log_info("Pipeline stopped.")

if __name__ == '__main__':
    try:
        cfg = parse_args(cfg_path="config/config_pipeline.toml")
        sys.exit(main(cfg))
    except Exception as e:
        log_error(f"Fatal error in __main__: {e}\n{traceback.format_exc()}")
        sys.exit(1)
