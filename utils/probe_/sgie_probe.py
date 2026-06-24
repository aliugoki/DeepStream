import os
import time
import logging
from datetime import datetime, date
from gi.repository import Gst
import pyds

from .constants import *
from .state import CAMERA_STATE, FACE_SAVE_STATE, FACE_SAVE_LOCK
from .geometry import is_in_area, process_line_crossing
from .face_features import get_face_feature, match_faces
from .overlays import add_label, draw_static_overlays

from ..posgres_service import get_user_info, log_attendance, COMPANY_ID


logger = logging.getLogger("SGIE")

_OBJ_ENC_CTX = None
def get_obj_encoder(gpu_id=0):
    global _OBJ_ENC_CTX
    if not _OBJ_ENC_CTX:
        _OBJ_ENC_CTX = pyds.nvds_obj_enc_create_context(gpu_id)
    return _OBJ_ENC_CTX

def should_save_face(src, emp):
    key = (src, emp, date.today())
    with FACE_SAVE_LOCK:
        if key in FACE_SAVE_STATE:
            return False
        FACE_SAVE_STATE[key] = time.time()
        return True

def sgie_feature_extract_probe(pad, info, user_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    loaded_faces = user_data["loaded_faces"]
    q = user_data["attendance_queue"]

    l_frame = batch.frame_meta_list
    while l_frame:
        frame = pyds.NvDsFrameMeta.cast(l_frame.data)
        src = frame.source_id
        now = datetime.now()

        cam = CAMERA_STATE.setdefault(src, {
            "in": 0, "out": 0,
            "status": "Standby",
            "status_ts": now,
            "last_date": date.today()
        })

        if cam["last_date"] != date.today():
            cam.update({"in": 0, "out": 0, "last_date": date.today()})

        draw_static_overlays(frame, batch)

        l_obj = frame.obj_meta_list
        while l_obj:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            direction = process_line_crossing(src, obj, now)

            if direction:
                cam[direction.lower()] += 1
                cam["status"] = direction
                cam["status_ts"] = now

            if not is_in_area(obj):
                l_obj = l_obj.next
                continue

            feat = get_face_feature(obj)
            label = "Unknown"

            if feat is not None:
                uid, score = match_faces(feat, loaded_faces)
                if uid and score >= RECOGNITION_THRESHOLD and direction:
                    fname, lname, _, img = get_user_info(uid, COMPANY_ID)
                    label = f"{fname} {lname}"

                    if should_save_face(src, uid):
                        save_dir = f"/workspace/data/attendance_faces/Cam_{src}/{date.today()}"
                        os.makedirs(save_dir, exist_ok=True)
                        path = f"{save_dir}/{uid}_{now:%H%M%S}.jpg"

                        args = pyds.NvDsObjEncUsrArgs()
                        args.saveImg = 1
                        args.fileNameImg = path
                        args.quality = 90
                        pyds.nvds_obj_enc_process(
                            get_obj_encoder(), args, hash(buf), obj, frame
                        )

                        q.put({
                            "company_id": COMPANY_ID,
                            "emp_id": uid,
                            "first_name": fname,
                            "last_name": lname,
                            "image_url": img,
                            "attendance_date": now.strftime("%d-%m-%Y"),
                            "attendance_time": now.strftime("%H:%M:%S"),
                            "check_type": direction,
                            "camera_name": f"Camera_{src}"
                        })

            add_label(frame, batch, label,
                      obj.rect_params.left,
                      obj.rect_params.top - 20)

            l_obj = l_obj.next

        if (now - cam["status_ts"]).total_seconds() > GLOBAL_STATUS_COOLDOWN:
            cam["status"] = "Standby"

        add_label(frame, batch,
                  f"STATUS:{cam['status']} IN:{cam['in']} OUT:{cam['out']}",
                  10, 10, color=(1,1,0,1), bg=(0,0,0,0.8))

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def attach_sgie_probe(sgie, loaded_faces, attendance_queue, sources):
    """
    Attaches SGIE src pad probe with required user data.
    """
    pad = sgie.get_static_pad("src")
    if not pad:
        raise RuntimeError("Failed to get SGIE src pad")

    # Correctly pass user_data as a dictionary
    user_data = {
        "loaded_faces": loaded_faces,
        "attendance_queue": attendance_queue,
        "sources": sources   # <-- include sources if needed
    }

    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        sgie_feature_extract_probe,
        user_data
    )

