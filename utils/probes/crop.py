# utils/probe/crop.py

import numpy as np
import cv2
import pyds


def extract_face_image(gst_buffer, frame_meta, obj_meta):
    """
    Extract face crop as BGR NumPy image.
    """
    try:
        # Map GPU buffer
        n_frame = pyds.get_nvds_buf_surface(
            hash(gst_buffer),
            frame_meta.batch_id
        )

        frame = np.array(n_frame, copy=False, order="C")

        # Convert RGBA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        rect = obj_meta.rect_params
        x = int(rect.left)
        y = int(rect.top)
        w = int(rect.width)
        h = int(rect.height)

        # Clamp to frame boundaries
        h_frame, w_frame, _ = frame.shape
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        face = frame[y:y + h, x:x + w]

        if face.size == 0:
            return None

        return face

    except Exception:
        return None
