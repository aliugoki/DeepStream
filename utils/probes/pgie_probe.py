import pyds
from gi.repository import Gst
from .constants import PGIE_CONFIDENCE_THRESHOLD

def pgie_src_filter_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        to_remove = []
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            if obj_meta.confidence < PGIE_CONFIDENCE_THRESHOLD:
                to_remove.append(obj_meta)
            l_obj = l_obj.next
        for o in to_remove:
            pyds.nvds_remove_obj_meta_from_frame(frame_meta, o)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK
