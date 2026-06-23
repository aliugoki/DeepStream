"""
Enterprise recognition probe: aligned ArcFace + stabilized identity.

Differs from the legacy utils/probe_git.py SGIE probe:
  * faces are 5-point aligned (Umeyama -> 112x112 ArcFace template) using the
    landmarks the YOLO-face PGIE already emits in obj_meta.mask_params, instead
    of feeding raw axis-aligned crops to the SGIE;
  * embeddings are computed in-probe by a standalone ArcFace TensorRT engine
    (utils.arcface_embedder), so this probe runs WITHOUT an ArcFace SGIE in the
    graph and produces vectors identical to the offline enrollment tool;
  * matching uses the configurable threshold + top-1/top-2 margin gate and a
    single vectorized matmul (utils.recognition.Gallery);
  * identity is stabilized per track (recognize-once-per-track) with TTL
    eviction, so a noisy frame can't mislabel a person and committed tracks are
    not re-embedded -- bounding both error and GPU cost.

Attach on the src pad of the RGBA capsfilter that sits AFTER the tracker and
BEFORE the tiler, so landmark coordinates are in per-source frame space and the
frame surface is CPU/GPU-addressable (NVBUF_MEM_CUDA_UNIFIED).

NOTE: the live frame-surface + TensorRT path requires a GPU/DeepStream runtime
and must be validated on-device; the recognition/alignment/embedder pieces it
builds on are independently unit-tested offline.
"""
import os
import logging
from datetime import datetime, date

import numpy as np
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from .face_align import align_chip, inverse_letterbox_points, parse_landmarks
from .recognition import Gallery, TrackIdentityManager
# Reuse the proven display / line-crossing helpers from the legacy probe...
from .probe_git import (
    draw_detection_area, draw_detection_line, check_line_crossing,
    is_in_detection_area, display_name_on_frame, display_global_status,
    display_daily_counts, GLOBAL_STATUS_DISPLAY_COOLDOWN_SECONDS,
)
# ...but use the hardened, pooled, env-driven DB/webhook layer for persistence.
from .db_service import get_user_info, log_attendance, attendance_worker, COMPANY_ID

log = logging.getLogger("probe_enterprise")

MIN_LANDMARK_SCORE = 0.30
CAMERA_STATE = {}


def _read_rgba_frame(gst_buffer, frame_meta):
    """Return an HxWx3 BGR copy of the frame surface, or None."""
    try:
        surf = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame = np.array(surf, copy=True, order='C')  # RGBA
        try:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except Exception:
            pass  # dGPU unified memory does not require unmap
        # RGBA -> BGR for the embedder (which converts BGR->RGB internally).
        return frame[:, :, [2, 1, 0]].copy()
    except Exception as e:
        log.error("surface map failed: %s", e)
        return None


class EnterpriseRecognizer:
    """Holds the gallery, embedder, and per-track identity state."""

    def __init__(self, embedder, gallery: Gallery, track_mgr: TrackIdentityManager,
                 sources, attendance_queue, save_unknown_dir=None, health=None,
                 vt_publisher=None):
        self.embedder = embedder
        self.gallery = gallery
        self.tracks = track_mgr
        self.sources = sources or []
        self.attendance_queue = attendance_queue
        self.save_unknown_dir = save_unknown_dir
        self.health = health
        self.vt = vt_publisher  # optional VisionTrack identity publisher

    # -- main probe entrypoint -------------------------------------------------
    def probe(self, pad, info, _u):
        try:
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                if frame_meta:
                    self._process_frame(buf, batch_meta, frame_meta)
                l_frame = l_frame.next if hasattr(l_frame, 'next') else None
        except Exception:
            log.exception("probe failure")
        return Gst.PadProbeReturn.OK

    def _process_frame(self, buf, batch_meta, frame_meta):
        source_id = frame_meta.source_id
        if self.health is not None:
            self.health.mark_frame(source_id)
        camera_name, camera_type = f"Camera {source_id}", "unknown"
        if 0 <= source_id < len(self.sources):
            si = self.sources[source_id]
            camera_name = si.get("id", camera_name)
            camera_type = si.get("type", camera_type)

        st = CAMERA_STATE.setdefault(source_id, {
            'in_count': 0, 'out_count': 0, 'last_reset_date': date.today(),
            'last_global_status': {'status': 'Standby', 'timestamp': datetime.now()}})

        now, today = datetime.now(), date.today()
        if today != st['last_reset_date']:
            st.update(in_count=0, out_count=0, last_reset_date=today,
                      last_global_status={'status': 'Standby', 'timestamp': now})

        draw_detection_area(frame_meta, batch_meta)
        draw_detection_line(frame_meta, batch_meta)

        # Collect faces needing recognition (skip already-committed tracks).
        frame_bgr = None
        chips, pending = [], []
        objs = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            if obj:
                objs.append(obj)
                if not self.tracks.is_committed(obj.object_id):
                    pts, scores = self._landmarks(obj, frame_meta)
                    if pts is not None and scores.mean() >= MIN_LANDMARK_SCORE:
                        if frame_bgr is None:
                            frame_bgr = _read_rgba_frame(buf, frame_meta)
                        if frame_bgr is not None:
                            chips.append(align_chip(frame_bgr, pts))
                            pending.append(obj)
            l_obj = l_obj.next if hasattr(l_obj, 'next') else None

        # One batched embedding call per frame for all pending faces.
        embs = self.embedder.embed(np.stack(chips)) if chips else None

        for obj in objs:
            committed = None
            if embs is not None and obj in pending:
                feat = embs[pending.index(obj)]
                bid, score, margin = self.gallery.match(feat)
                committed = self.tracks.observe(obj.object_id, bid, score, margin)
            elif self.tracks.is_committed(obj.object_id):
                committed = self.tracks.observe(obj.object_id, None, -1.0, -1.0)

            self._handle_object(obj, frame_meta, batch_meta, now, today,
                                 camera_name, camera_type, st, committed)

        elapsed = (now - st['last_global_status']['timestamp']).total_seconds()
        if elapsed > GLOBAL_STATUS_DISPLAY_COOLDOWN_SECONDS:
            st['last_global_status']['status'] = 'Standby'
        display_global_status(frame_meta, batch_meta, st['last_global_status']['status'])
        display_daily_counts(frame_meta, batch_meta, st['in_count'], st['out_count'], camera_type)

    def _handle_object(self, obj, frame_meta, batch_meta, now, today,
                       camera_name, camera_type, st, committed_id):
        label = "Unknown"
        direction = check_line_crossing(obj, obj.object_id, now, frame_meta, batch_meta, label)

        person_name = ""
        if committed_id is not None:
            label = f"ID:{committed_id}"
            if is_in_detection_area(obj):
                emp_id = str(committed_id).strip()
                first, last, db_emp, image = get_user_info(emp_id, COMPANY_ID)
                if first or last or db_emp:
                    person_name = f"{first} {last}".strip()
                    label = f"{first} {last} (Emp ID: {db_emp})"
                    check_type = {'entrance': 'in', 'exit': 'out'}.get(camera_type)
                    if check_type and hasattr(self.attendance_queue, 'put'):
                        self.attendance_queue.put({
                            "company_id": COMPANY_ID, "emp_id": emp_id,
                            "first_name": first, "last_name": last, "image_url": image,
                            "attendance_date": today.strftime("%d-%m-%Y"),
                            "attendance_time": now.strftime("%H:%M:%S"),
                            "check_type": check_type, "camera_name": camera_name})
                else:
                    label = f"ID:{committed_id} (No Info)"

            # Feed the recognized identity to VisionTrack (dedicated stream).
            if self.vt is not None:
                r = obj.rect_params
                self.vt.publish(frame_meta.source_id, committed_id, person_name,
                                1.0, (r.left, r.top, r.width, r.height))

        display_name_on_frame(obj, frame_meta, batch_meta, label)

        if direction == "IN" and camera_type == 'entrance':
            st['in_count'] += 1
            st['last_global_status'] = {'status': 'IN', 'timestamp': now}
        elif direction == "OUT" and camera_type == 'exit':
            st['out_count'] += 1
            st['last_global_status'] = {'status': 'OUT', 'timestamp': now}

    @staticmethod
    def _landmarks(obj, frame_meta):
        """Extract 5 landmarks (frame pixels) from obj_meta.mask_params."""
        try:
            mp = obj.mask_params
            if not mp or mp.size == 0:
                return None, None
            data = mp.get_mask_array() if hasattr(mp, "get_mask_array") else None
            if data is None:
                return None, None
            pts, scores = parse_landmarks(np.array(data, copy=True))
            if pts is None:
                return None, None
            # PGIE landmarks are in 640x640 net space; map to frame pixels.
            fw = frame_meta.source_frame_width or 1280
            fh = frame_meta.source_frame_height or 720
            return inverse_letterbox_points(pts, fw, fh), scores
        except Exception:
            return None, None


def attach_enterprise_probe(pad_element, recognizer):
    """Attach the recognizer to the given element's src pad."""
    src = pad_element.get_static_pad("src")
    if not src:
        log.error("could not get src pad for enterprise probe")
        return False
    src.add_probe(Gst.PadProbeType.BUFFER, recognizer.probe, None)
    log.info("enterprise recognition probe attached")
    return True
