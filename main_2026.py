# main_2026.py
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

import os
import logging
from datetime import date
from utils.probe import attach_sgie_probe, attendance_worker
from utils.probe.encoder import get_obj_encoder, save_face_crop_probe

# Initialize GStreamer
Gst.init(None)

# Config loader
import configparser
import ast

def load_config(config_path):
    parser = configparser.ConfigParser(strict=False)
    parser.read(config_path)
    cfg = {}

    for section in parser.sections():
        if section.startswith("sources"):
            cfg.setdefault("sources", [])
            d = {}
            for k, v in parser.items(section):
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    try:
                        v = ast.literal_eval(v)
                    except:
                        pass
                elif v.isdigit():
                    v = int(v)
                d[k] = v
            cfg["sources"].append(d)
        else:
            d = {}
            for k, v in parser.items(section):
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    try:
                        v = ast.literal_eval(v)
                    except:
                        pass
                elif v.isdigit():
                    v = int(v)
                elif v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                d[k] = v
            cfg[section] = d
    return cfg


# Load configuration
cfg = load_config("/workspace/config/config.ini")
sources = cfg["sources"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepStream2026")


# ---------------------------
# Helper: create_source_bin
# ---------------------------
def create_source_bin(index, uri):
    bin_name = f"source-bin-{index}"
    source_bin = Gst.Bin.new(bin_name)

    # Decodebin for RTSP
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"decode-bin-{index}")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", decodebin_pad_added, source_bin)

    source_bin.add(uri_decode_bin)
    return source_bin


# Called when decodebin adds pad
def decodebin_pad_added(decodebin, pad, target_bin):
    caps = pad.get_current_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    if name.startswith("video/"):
        queue = Gst.ElementFactory.make("queue", None)
        conv = Gst.ElementFactory.make("nvvideoconvert", None)
        sink = Gst.ElementFactory.make("nvvideosink", None)
        target_bin.add(queue)
        target_bin.add(conv)
        target_bin.add(sink)
        queue.sync_state_with_parent()
        conv.sync_state_with_parent()
        sink.sync_state_with_parent()
        pad.link(queue.get_static_pad("sink"))
        queue.link(conv)
        conv.link(sink)


# ---------------------------
# Build pipeline
# ---------------------------
def main(cfg):
    pipeline = Gst.Pipeline.new("ds-pipeline")

    # Streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    sm_cfg = cfg["streammux"]
    streammux.set_property("width", sm_cfg["width"])
    streammux.set_property("height", sm_cfg["height"])
    streammux.set_property("batch-size", sm_cfg["batch-size"])
    streammux.set_property("batched-push-timeout", sm_cfg["batched-push-timeout"])
    pipeline.add(streammux)

    # PGIE (YOLO face detector)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", cfg["pgie"]["config-file-path"])
    pipeline.add(pgie)

    # SGIE (ArcFace feature extractor)
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", cfg["sgie"]["config-file-path"])
    pipeline.add(sgie)

    # OSD
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    pipeline.add(nvosd)

    # Tiler
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    tiler.set_property("width", cfg["tiler"]["width"])
    tiler.set_property("height", cfg["tiler"]["height"])
    pipeline.add(tiler)

    # Tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker_config_path = cfg["tracker"]["config-file-path"]
    tracker.set_property("tracker-width", 640)  # optional
    tracker.set_property("tracker-height", 480)
    pipeline.add(tracker)

    # ---------------------------
    # Add sources
    # ---------------------------
    for i, src in enumerate(sources):
        sb = create_source_bin(i, src["uri"])
        pipeline.add(sb)
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        src_pad = sb.get_static_pad("src")
        if not sinkpad or not src_pad:
            logger.error(f"Source {i} pads not available")
            continue
        src_pad.link(sinkpad)

    # ---------------------------
    # Attach SGIE probe for feature extraction
    # ---------------------------
    user_data = {
        "loaded_faces": {},  # TODO: load known features
        "attendance_queue": [],
        "sources": sources
    }
    sgie_src_pad = sgie.get_static_pad("src")
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, attach_sgie_probe, user_data)

    # ---------------------------
    # Start pipeline
    # ---------------------------
    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()

    while True:
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"ERROR: {err}, {debug}")
                break
            elif msg.type == Gst.MessageType.EOS:
                logger.info("End of stream")
                break

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main(cfg)
