# merged_main.py
import sys
import os
import gi
import time
import cv2
import math
import queue
import threading
import traceback
import multiprocessing
import numpy as np
from datetime import datetime

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GLib, GstRtspServer

# ------------------------------------------------------------------------------
# Imports from existing utils (UNCHANGED)
# ------------------------------------------------------------------------------
from utils.probe import (
    pgie_src_filter_probe,
    attach_sgie_probe,
    attendance_worker
)
from utils.parser_cfg import (
    parse_args,
    set_property,
    set_tracker_properties,
    load_faces
)
from utils.bus_call import bus_call

# ------------------------------------------------------------------------------
# GLOBAL RECOGNITION STATE (UPDATED BY SGIE PROBE)
# ------------------------------------------------------------------------------
# object_id → dict(name, confidence, bbox, embedding, last_seen)
RECOGNITION_STATE = {}

# ------------------------------------------------------------------------------
# FACE CROP MANAGER (FULL LOGIC)
# ------------------------------------------------------------------------------
class FaceCropManager:
    def __init__(
        self,
        base_dir,
        enable=True,
        save_known=True,
        save_unknown=True,
        rate_limit_sec=5,
        min_face_area=2500,
        confidence_threshold=0.75
    ):
        self.enable = enable
        self.save_known = save_known
        self.save_unknown = save_unknown
        self.rate_limit = rate_limit_sec
        self.min_face_area = min_face_area
        self.conf_threshold = confidence_threshold

        self.base_dir = base_dir
        self.last_saved = {}     # object_id → timestamp
        self.q = queue.Queue(maxsize=300)

        os.makedirs(base_dir, exist_ok=True)

        self.worker = threading.Thread(
            target=self._writer,
            daemon=True,
            name="FaceCropWriter"
        )
        self.worker.start()

    # --------------------------------------------------------------------------
    def _should_save(self, object_id):
        now = time.time()
        last = self.last_saved.get(object_id, 0)
        if now - last >= self.rate_limit:
            self.last_saved[object_id] = now
            return True
        return False

    # --------------------------------------------------------------------------
    def submit(self, frame, obj):
        """
        obj:
          {
            object_id,
            name,
            confidence,
            bbox
          }
        """
        if not self.enable:
            return

        object_id = obj["object_id"]
        name = obj["name"]
        conf = obj["confidence"]
        x1, y1, x2, y2 = obj["bbox"]

        if not self._should_save(object_id):
            return

        # Clamp bbox
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return

        area = (x2 - x1) * (y2 - y1)
        if area < self.min_face_area:
            return

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return

        is_known = conf >= self.conf_threshold and name != "unknown"

        if is_known and not self.save_known:
            return
        if not is_known and not self.save_unknown:
            return

        label = name if is_known else "unknown"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        out_dir = os.path.join(self.base_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        fname = f"{label}_{object_id}_{ts}.jpg"
        self.q.put((os.path.join(out_dir, fname), crop))

    # --------------------------------------------------------------------------
    def _writer(self):
        while True:
            try:
                path, img = self.q.get()
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"[FaceCropManager] write error: {e}")

# ------------------------------------------------------------------------------
# APPSINK CALLBACK
# ------------------------------------------------------------------------------
def make_appsink_callback(crop_manager):
    def on_new_sample(sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=mapinfo.data
        )

        now = time.time()
        for obj_id, info in list(RECOGNITION_STATE.items()):
            if now - info["last_seen"] > 2:
                continue

            crop_manager.submit(frame, {
                "object_id": obj_id,
                "name": info["name"],
                "confidence": info["confidence"],
                "bbox": info["bbox"]
            })

        buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    return on_new_sample

# ------------------------------------------------------------------------------
# MAIN (UNCHANGED + CROP ATTACHMENT)
# ------------------------------------------------------------------------------
def main(cfg):
    multiprocessing.set_start_method('spawn', force=True)

    GObject.threads_init()
    Gst.init(None)

    # ------------------ Load known faces ------------------
    known_face_dir = cfg['pipeline']['known_face_dir']
    known_face_features = load_faces(known_face_dir) or {}

    # ------------------ Attendance worker -----------------
    attendance_queue = multiprocessing.Queue()
    attendance_proc = multiprocessing.Process(
        target=attendance_worker,
        args=(attendance_queue,)
    )
    attendance_proc.start()

    # ------------------ Pipeline elements -----------------
    pipeline = Gst.Pipeline.new("merged-pipeline")

    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvvidconv_sgie = Gst.ElementFactory.make("nvvideoconvert", "conv_sgie")
    capsfilter_sgie = Gst.ElementFactory.make("capsfilter", "caps_sgie")
    sgie = Gst.ElementFactory.make("nvinfer", "sgie")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "conv_main")
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
    tee = Gst.ElementFactory.make("tee", "tee")


    for el in [
        streammux, pgie, tracker, nvvidconv_sgie,
        capsfilter_sgie, sgie, tiler, nvvidconv,
        nvosd, tee
    ]:
        pipeline.add(el)

    # ------------------ Linking (unchanged) ------------------
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv_sgie)
    nvvidconv_sgie.link(capsfilter_sgie)
    capsfilter_sgie.link(sgie)
    sgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)

    # ------------------ Display branch ------------------
    q_disp = Gst.ElementFactory.make("queue", "q_disp")
    sink_disp = Gst.ElementFactory.make("nveglglessink", "disp")

    pipeline.add(q_disp)
    pipeline.add(sink_disp)

    tee.get_request_pad("src_%u").link(q_disp.get_static_pad("sink"))
    q_disp.link(sink_disp)

    # ------------------ FACE CROP BRANCH ------------------
    crop_cfg = cfg["pipeline"].get("face_crops", {})

    crop_manager = FaceCropManager(
        base_dir=crop_cfg.get("path", "/data/face_crops"),
        enable=crop_cfg.get("enabled", True),
        save_known=crop_cfg.get("save_known", True),
        save_unknown=crop_cfg.get("save_unknown", True),
        rate_limit_sec=crop_cfg.get("rate_limit_sec", 5),
        confidence_threshold=crop_cfg.get("confidence_threshold", 0.75)
    )

    q_cpu = Gst.ElementFactory.make("queue", "q_cpu")
    conv_cpu = Gst.ElementFactory.make("nvvideoconvert", "conv_cpu")
    caps_cpu = Gst.ElementFactory.make("capsfilter", "caps_cpu")
    appsink = Gst.ElementFactory.make("appsink", "appsink")

    caps_cpu.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw, format=BGR")
    )
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)

    for el in [q_cpu, conv_cpu, caps_cpu, appsink]:
        pipeline.add(el)

    tee.get_request_pad("src_%u").link(q_cpu.get_static_pad("sink"))
    q_cpu.link(conv_cpu)
    conv_cpu.link(caps_cpu)
    caps_cpu.link(appsink)

    appsink.connect(
        "new-sample",
        make_appsink_callback(crop_manager)
    )

    # ------------------ Probes ------------------
    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        pgie_src_filter_probe,
        0
    )

    attach_sgie_probe(
        sgie,
        known_face_features,
        attendance_queue,
        None,
        cfg["sources"],
        recognition_state=RECOGNITION_STATE
    )

    # ------------------ Run ------------------
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        attendance_queue.put(None)
        attendance_proc.join()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_args("config/config_pipeline.toml")
    sys.exit(main(cfg))
