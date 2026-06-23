# utils/sgie_probe_callback.py
import os
import logging
from datetime import datetime, date
import pyds
from .posgres_redis import get_user_info
from .probe_helpers import (
    get_obj_encoder, cleanup_face_save_state, should_save_face,
    draw_static_overlays, add_label, is_in_area, process_line_crossing,
    get_face_feature, match_faces
)

logger = logging.getLogger("SGIEProbe")

def sgie_feature_extract_probe(pad, info, user_data):
    """
    SGIE probe for face recognition, line crossing, and attendance queuing.
    Only enqueues tasks; worker handles FSM + DB.
    """
    cleanup_face_save_state()
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return pad

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    loaded_faces = user_data.get('loaded_faces', {})
    q = user_data.get('attendance_queue')
    sources = user_data.get('sources', [])

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        src_id = frame_meta.source_id
        now = datetime.now()

        draw_static_overlays(frame_meta, batch_meta)

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

            # Line crossing logic
            direction = process_line_crossing(obj_meta, obj_meta.object_id, now)

            # Face recognition
            feat = get_face_feature(obj_meta)
            name_label = "Unknown"
            if feat is not None:
                uid, score = match_faces(feat, loaded_faces)
                if uid and score >= 0.4:  # RECOGNITION_THRESHOLD
                    fname, lname, emp_id, _ = get_user_info(uid, user_data.get("company_id"))

                    name_label = f"{fname} {lname}"

                    if is_in_area(obj_meta) and should_save_face(src_id, uid, obj_meta.object_id):
                        # Save cropped face
                        save_dir = os.path.join("/workspace/data/attendance_faces", f"Cam_{src_id}", str(date.today()))
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{uid}_{now.strftime('%H%M%S')}.jpg")

                        enc_args = pyds.NvDsObjEncUsrArgs()
                        enc_args.saveImg, enc_args.fileNameImg, enc_args.quality = 1, save_path, 90
                        pyds.nvds_obj_enc_process(get_obj_encoder(), enc_args, hash(gst_buffer), obj_meta, frame_meta)

                        # Enqueue task
                        try:
                            q.put_nowait({
                                "company_id": user_data.get("company_id"),
                                "emp_id": uid,
                                "first_name": fname,
                                "last_name": lname,
                                "image_url": save_path,  # ✅ use actual saved image
                                "attendance_date": now.date(),  # ✅ date object
                                "attendance_time": now.time(),  # ✅ time object
                                "check_type": direction or "IN",
                                "camera_name": f"Camera_{src_id}"
                            })
                        except Exception:
                            logger.warning(f"Queue full, skipped emp_id={uid} cam={src_id}")

            add_label(frame_meta, batch_meta, name_label, obj_meta.rect_params.left, obj_meta.rect_params.top - 20)
            l_obj = l_obj.next

        l_frame = l_frame.next

    return pad
