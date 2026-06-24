import pyds
from .constants import MAX_DISPLAY_LABELS, MAX_DISPLAY_RECTS, MAX_DISPLAY_LINES, DETECTION_AREA, DETECTION_LINE

def add_label(frame_meta, batch_meta, text, x, y, color=(1,1,1,1), bg=(0,0,0,0.6)):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if display_meta.num_labels < MAX_DISPLAY_LABELS:
        txt = display_meta.text_params[display_meta.num_labels]
        txt.display_text = text
        txt.x_offset, txt.y_offset = int(x), int(max(0, y))
        txt.font_params.font_name = "Serif"
        txt.font_params.font_size = 15
        txt.font_params.font_color.set(*color)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(*bg)
        display_meta.num_labels += 1
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    else:
        pyds.nvds_release_display_meta(display_meta)

def draw_static_overlays(frame_meta, batch_meta):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    
    # Detection Rect
    if display_meta.num_rects < MAX_DISPLAY_RECTS:
        rect = display_meta.rect_params[display_meta.num_rects]
        rect.left, rect.top = DETECTION_AREA['x1'], DETECTION_AREA['y1']
        rect.width = DETECTION_AREA['x2'] - DETECTION_AREA['x1']
        rect.height = DETECTION_AREA['y2'] - DETECTION_AREA['y1']
        rect.border_width = 2
        rect.border_color.set(0,1,0,0.5)
        display_meta.num_rects += 1
        
    # Crossing Line
    if display_meta.num_lines < MAX_DISPLAY_LINES:
        line = display_meta.line_params[display_meta.num_lines]
        line.x1, line.y1 = DETECTION_LINE['x1'], DETECTION_LINE['y1']
        line.x2, line.y2 = DETECTION_LINE['x2'], DETECTION_LINE['y2']
        line.line_width = 3
        line.line_color.set(1,0,0,0.8)
        display_meta.num_lines += 1
    
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
