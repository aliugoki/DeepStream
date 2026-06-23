# utils/probe/display.py

import pyds
from .constants import MAX_DISPLAY_LABELS

def draw_name(frame_meta, batch_meta, obj_meta, text):
    meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if not meta or meta.num_labels >= MAX_DISPLAY_LABELS:
        return

    t = meta.text_params[meta.num_labels]
    t.display_text = text
    t.x_offset = int(obj_meta.rect_params.left)
    t.y_offset = int(obj_meta.rect_params.top - 20)
    t.font_params.font_size = 15
    t.set_bg_clr = 1
    meta.num_labels += 1

    pyds.nvds_add_display_meta_to_frame(frame_meta, meta)
