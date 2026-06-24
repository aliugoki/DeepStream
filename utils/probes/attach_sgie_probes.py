import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import logging
from .sgie_probe import sgie_feature_extract_probe

logger = logging.getLogger(__name__)

def attach_sgie_probe(sgie, loaded_faces, attendance_queue, sources):
    pad = sgie.get_static_pad("src")
    if not pad:
        raise RuntimeError("Failed to get SGIE src pad")

    user_data = {
        "loaded_faces": loaded_faces,
        "attendance_queue": attendance_queue,
        "sources": sources
    }

    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        sgie_feature_extract_probe,
        user_data
    )
