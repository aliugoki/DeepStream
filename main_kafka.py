# main_modular.py

import sys
import os
import time
import threading
import hashlib
import logging
import gi

# ---------------- ENV FIXES ----------------
os.environ['GIO_USE_VFS'] = 'local'
os.environ['GIO_USE_PROXY_RESOLVER'] = 'dummy'
os.environ['GIO_MODULE_DIR'] = '/nonexistent'
os.environ['no_proxy'] = '*'
os.environ['ALL_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['FTP_PROXY'] = ''

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# ---------------- APP IMPORTS ----------------
from utils.probe import pgie_src_filter_probe
from utils.probe.attach_sgie_probe import attach_sgie_probe
from utils.parser_cfg import parse_args, set_property, set_tracker_properties, load_faces
from utils.bus_call import bus_call

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("DeepStream")


# ===== Hot reload for faces =====
def get_dir_signature(directory):
    h = hashlib.md5()
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
                h.update(f.encode())
                h.update(str(st.st_mtime).encode())
                h.update(str(st.st_size).encode())
            except Exception:
                continue
    return h.hexdigest()


def start_face_reloader(path, shared_faces, interval=5):
    sig = [get_dir_signature(path)]

    def loop():
        while True:
            try:
                new_sig = get_dir_signature(path)
                if new_sig != sig[0]:
                    logger.info("Hot reload: faces updated")
                    new_faces = load_faces(path) or {}
                    shared_faces.clear()
                    shared_faces.update(new_faces)
                    sig[0] = new_sig
            except Exception:
                logger.exception("Face reload error")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


# ===== Source bin helpers =====
def cb_newpad(decodebin, decoder_src_pad, data):
    try:
        source_bin = data
        queue = source_bin.get_by_name("queue")
        if not queue:
            return
        sink_pad = queue.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        decoder_src_pad.link(sink_pad)
        logger.info("Linked decodebin → queue")
    except Exception:
        logger.exception("cb_newpad error")


def decodebin_child_added(child_proxy, obj, name, user_data):
    if "decodebin" in name:
        obj.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        obj.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    bin = Gst.Bin.new(f"source-bin-{index}")

    decodebin = Gst.ElementFactory.make("uridecodebin", f"decodebin-{index}")
    decodebin.set_property("uri", uri)
    decodebin.connect("pad-added", cb_newpad, bin)
    decodebin.connect("child-added", decodebin_child_added, bin)

    queue = Gst.ElementFactory.make("queue", "queue")
    queue.set_property("leaky", 2)
    queue.set_property("max-size-buffers", 30)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert")
    capsfilter = Gst.ElementFactory.make("capsfilter")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    )

    for e in [decodebin, queue, nvvidconv, capsfilter]:
        bin.add(e)

    queue.link(nvvidconv)
    nvvidconv.link(capsfilter)

    ghost_pad = Gst.GhostPad.new("src", capsfilter.get_static_pad("src"))
    bin.add_pad(ghost_pad)

    return bin


# ===== Main Pipeline =====
def main(cfg):
    Gst.init(None)

    known_dir = cfg["pipeline"]["known_face_dir"]
    known_faces = load_faces(known_dir) or {}

    pipeline = Gst.Pipeline.new("ds-pipeline")

    # Core elements
    streammux = Gst.ElementFactory.make("nvstreammux")
    pgie = Gst.ElementFactory.make("nvinfer")
    tracker = Gst.ElementFactory.make("nvtracker")
    nvvidconv_sgie = Gst.ElementFactory.make("nvvideoconvert")
    caps_sgie = Gst.ElementFactory.make("capsfilter")
    sgie = Gst.ElementFactory.make("nvinfer")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert")
    nvosd = Gst.ElementFactory.make("nvdsosd")

    # Kafka branch
    msgconv = Gst.ElementFactory.make("nvmsgconv", "msgconv")
    msgbroker = Gst.ElementFactory.make("nvmsgbroker", "msgbroker")
    msgconv.set_property("config", "config/msgconv_config.txt")
    msgconv.set_property("payload-type", 0)
    msgbroker.set_property("config", "config/kafka_broker.cfg")
    msgbroker.set_property(
        "proto-lib", "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
    )
    msgbroker.set_property("conn-str", "kafka:9092")
    msgbroker.set_property("topic", "attendance.events")
    msgbroker.set_property("sync", False)

    for e in [
        streammux, pgie, tracker, nvvidconv_sgie, caps_sgie, sgie,
        tiler, nvvidconv, nvosd, msgconv, msgbroker
    ]:
        pipeline.add(e)

    # ===== Sources =====
    is_live = False
    for i, src in enumerate(cfg["sources"]):
        uri = src["uri"]
        if uri.startswith(("rtsp://", "rtspt://")):
            is_live = True
        sb = create_source_bin(i, uri)
        pipeline.add(sb)
        streammux.get_request_pad(f"sink_{i}").link(sb.get_static_pad("src"))

    streammux.set_property("live-source", is_live)
    set_property(cfg, streammux, "streammux")
    set_property(cfg, pgie, "pgie")
    set_property(cfg, sgie, "sgie")
    set_property(cfg, tiler, "tiler")
    set_property(cfg, nvosd, "nvosd")
    set_tracker_properties(tracker, cfg["tracker"]["config-file-path"])

    # ===== Link pipeline =====
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv_sgie)
    nvvidconv_sgie.link(caps_sgie)
    caps_sgie.link(sgie)
    sgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)

    # Kafka branch
    nvosd.link(msgconv)
    msgconv.link(msgbroker)

    # ===== Probes =====
    attach_sgie_probe(
        sgie,
        known_faces=known_faces,
        sources=cfg["sources"],
        threshold=0.55
    )

    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        pgie_src_filter_probe,
        None
    )

    start_face_reloader(
        known_dir,
        known_faces,
        int(cfg["pipeline"].get("reload_interval", 5))
    )

    # ===== Run Loop =====
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    logger.info("Pipeline PLAYING")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    return 0


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    cfg = parse_args("config/config.toml")
    sys.exit(main(cfg))
