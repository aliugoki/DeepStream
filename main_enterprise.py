"""
Enterprise face-recognition pipeline (aligned ArcFace + stabilized identity).

Runs alongside the legacy main_udp.py without modifying it. Key differences:
  * No ArcFace SGIE in the graph. Embeddings are produced in a probe on
    5-point-aligned 112x112 chips by a standalone ArcFace TensorRT engine, so
    recognition accuracy no longer suffers from unaligned crops.
  * Recognition threshold + top-1/top-2 margin come from config (the legacy
    code hardcoded 0.2 and ignored rec_threshold).
  * Identity is stabilized per track with TTL eviction (bounded memory).
  * Gallery hot-reload is an atomic swap (no clear()+update() race).

Prerequisite: the gallery MUST be (re-)enrolled with tools/enroll.py so its
vectors come from the same aligned embedder (see gallery_meta.json marker).

Graph: source(s) -> nvstreammux -> PGIE(YOLO-face) -> nvtracker
       -> nvvideoconvert(RGBA, unified mem) -> [recognition probe]
       -> nvmultistreamtiler -> nvvideoconvert -> nvdsosd -> tee
       -> {display/fakesink, UDP+RTSP out}
"""
import sys
import os
import time
import json
import logging
import threading
import traceback
import multiprocessing

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib

# Reuse the RTSP-output helper from the legacy module.
from main_udp import add_udp_rtsp_branches, get_dir_signature
from utils.parser_cfg import (parse_args, set_property, set_tracker_properties,
                              load_faces)
from utils.reliability import (create_resilient_source_bin, make_resilient_bus_call,
                               HealthState, start_health_server, start_watchdog)
from utils.recognition import Gallery, TrackIdentityManager
from utils.visiontrack_publisher import from_config as vt_from_config
from utils.arcface_embedder import make_embedder
from utils.probe_enterprise import (EnterpriseRecognizer, attach_enterprise_probe,
                                    attendance_worker)
from utils.probe_git import pgie_src_filter_probe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepStream-Enterprise")

# NvBufSurfaceMemType enum (nvbuf-memory-type): device=2, UNIFIED=3. Unified is
# CPU+GPU-addressable, which the in-probe np.array(get_nvds_buf_surface(...)) read
# requires on dGPU — value 2 is device-only memory and segfaults the probe.
NVBUF_MEM_CUDA_UNIFIED = 3  # dGPU; see deepstream-imagedata-multistream sample
# The nvv4l2 decoder's cudadec-memtype is a DIFFERENT enum: device=0, pinned=1,
# unified=2. Keep it separate so we don't reuse the NvBufSurface value (3) here.
CUDADEC_MEMTYPE_UNIFIED = 2


def start_gallery_reloader(path, gallery: Gallery, interval=5):
    """Hot-reload the gallery on directory change via an atomic swap."""
    sig = [get_dir_signature(path)]

    def loop():
        while True:
            try:
                new = get_dir_signature(path)
                if new != sig[0]:
                    logger.info("Hot reload: gallery updated")
                    gallery.replace(load_faces(path) or {})
                    sig[0] = new
            except Exception:
                traceback.print_exc()
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


def check_gallery_consistency(known_dir):
    """Warn loudly if the gallery was not enrolled with the aligned embedder."""
    meta_path = os.path.join(known_dir, "gallery_meta.json")
    if not os.path.exists(meta_path):
        logger.warning("No gallery_meta.json in %s -- the gallery may be legacy "
                       "(unaligned). Run tools/enroll.py --all before trusting "
                       "matches.", known_dir)
        return
    try:
        meta = json.load(open(meta_path))
        if not meta.get("aligned"):
            logger.warning("gallery_meta.json reports a non-aligned gallery.")
        else:
            logger.info("Gallery: %s (%s faces).", meta.get("embedder"), meta.get("count"))
    except Exception as e:
        logger.warning("Could not read gallery_meta.json: %s", e)


def main(cfg):
    multiprocessing.set_start_method("spawn", force=True)
    Gst.init(None)

    pcfg = cfg["pipeline"]
    known_dir = pcfg["known_face_dir"]
    check_gallery_consistency(known_dir)

    gallery = Gallery()
    gallery.replace(load_faces(known_dir) or {})
    logger.info("Loaded gallery with %d faces.", len(gallery))

    track_mgr = TrackIdentityManager(
        threshold=float(pcfg.get("rec_threshold", 0.35)),
        min_margin=float(pcfg.get("rec_margin", 0.05)),
        min_votes=int(pcfg.get("track_min_votes", 3)),
        ttl_seconds=float(pcfg.get("track_ttl_sec", 30.0)),
    )

    embedder = make_embedder(
        pcfg.get("embedder", "trt"),
        pcfg.get("arcface_engine", "/workspace/models/arcface/arc1.engine")
        if pcfg.get("embedder", "trt") == "trt"
        else pcfg.get("arcface_onnx", "/workspace/models/arcface/arcface.onnx"),
    )

    attendance_q = multiprocessing.Queue()
    attendance_p = multiprocessing.Process(target=attendance_worker,
                                            args=(attendance_q,), daemon=True)
    attendance_p.start()

    pipeline = Gst.Pipeline.new("ds-enterprise")
    streammux = Gst.ElementFactory.make("nvstreammux")
    pgie = Gst.ElementFactory.make("nvinfer")
    tracker = Gst.ElementFactory.make("nvtracker")
    nvvidconv_rgba = Gst.ElementFactory.make("nvvideoconvert")
    caps_rgba = Gst.ElementFactory.make("capsfilter")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert")
    nvosd = Gst.ElementFactory.make("nvdsosd")
    tee = Gst.ElementFactory.make("tee")

    for e in [streammux, pgie, tracker, nvvidconv_rgba, caps_rgba,
              tiler, nvvidconv, nvosd, tee]:
        pipeline.add(e)

    caps_rgba.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))

    gpu_id = int(cfg.get("streammux", {}).get("gpu_id", 0))
    is_live = False
    for i, src in enumerate(cfg["sources"]):
        if src["uri"].startswith(("rtsp://", "rtspt://")):
            is_live = True
        # nvurisrcbin with native RTSP reconnect (see utils/reliability.py).
        sb = create_resilient_source_bin(i, src, gpu_id=gpu_id,
                                         cudadec_memtype=CUDADEC_MEMTYPE_UNIFIED)
        pipeline.add(sb)
        sb.get_static_pad("src").link(streammux.get_request_pad(f"sink_{i}"))

    streammux.set_property("live-source", is_live)
    set_property(cfg, streammux, "streammux")
    set_property(cfg, pgie, "pgie")
    set_property(cfg, tiler, "tiler")
    set_property(cfg, nvosd, "nvosd")
    set_tracker_properties(tracker, cfg["tracker"]["config-file-path"])

    # CPU/GPU-addressable surfaces for in-probe alignment (dGPU unified mem).
    streammux.set_property("nvbuf-memory-type", NVBUF_MEM_CUDA_UNIFIED)
    nvvidconv_rgba.set_property("nvbuf-memory-type", NVBUF_MEM_CUDA_UNIFIED)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv_rgba)
    nvvidconv_rgba.link(caps_rgba)
    caps_rgba.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)

    # Display branch.
    q_disp = Gst.ElementFactory.make("queue")
    # Leaky (drop oldest) so a slow display can't build latency or back-pressure
    # the tee — the RTSP/recognition branches must never stall for the local window.
    q_disp.set_property("leaky", 2)            # 2 = downstream (drop old buffers)
    q_disp.set_property("max-size-buffers", 3)
    sink_disp = Gst.ElementFactory.make(
        "nveglglessink" if pcfg.get("display") else "fakesink")
    pipeline.add(q_disp); pipeline.add(sink_disp)
    if pcfg.get("display"):
        # Live source + nveglglessink with clock sync floods the log with
        # "A lot of buffers are being dropped / is_too_late". Render frames as they
        # arrive instead of dropping for lateness; the local window is best-effort.
        sink_disp.set_property("sync", False)
        sink_disp.set_property("qos", False)
    tee.get_request_pad("src_%u").link(q_disp.get_static_pad("sink"))
    q_disp.link(sink_disp)

    # Health/metrics state shared across probe, bus handler, and watchdog.
    health = HealthState(cfg["sources"],
                         stale_after_sec=float(pcfg.get("stale_after_sec", 20)))
    start_health_server(health, port=int(pcfg.get("health_port", 9108)))

    # Optional: publish recognized identities to VisionTrack (dedicated stream).
    vt_publisher = vt_from_config(cfg)
    if vt_publisher:
        logger.info("VisionTrack identity publishing enabled.")

    # Recognition probe (after tracker, before tiler -> per-source frames).
    recognizer = EnterpriseRecognizer(embedder, gallery, track_mgr,
                                      cfg["sources"], attendance_q, health=health,
                                      vt_publisher=vt_publisher)
    attach_enterprise_probe(caps_rgba, recognizer)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                         pgie_src_filter_probe, None)
    start_gallery_reloader(known_dir, gallery, int(pcfg.get("reload_interval", 5)))

    add_udp_rtsp_branches(pipeline, tee, cfg)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", make_resilient_bus_call(loop, health), loop)
    start_watchdog(health, interval_sec=int(pcfg.get("watchdog_interval_sec", 10)))

    pipeline.set_state(Gst.State.PLAYING)
    logger.info("Enterprise pipeline PLAYING")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        attendance_q.put(None)
        attendance_p.join(3)
    return 0


if __name__ == "__main__":
    cfg = parse_args("config/config_pipeline.toml")
    sys.exit(main(cfg))
