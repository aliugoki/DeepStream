import logging
import numpy as np
import datetime
import pyds
from .attendance_worker import attendance_q

logger = logging.getLogger("DeepStream.AttachSGIEProbe")


def sgie_feature_extract_probe(pad, info, udata):
    """
    SGIE probe callback: extracts embeddings from tensor meta,
    performs FAISS search, and pushes attendance dicts to the worker queue.
    """
    try:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        # Get batch meta (pass raw pointer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buf.__gpointer__)
        if not batch_meta:
            logger.warning("No batch meta in buffer")
            return Gst.PadProbeReturn.OK

        if "faiss_index" not in udata:
            return Gst.PadProbeReturn.OK  # FAISS not initialized

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                # Only process faces (update class_id if needed)
                if obj_meta.class_id != udata.get("face_class_id", 0):
                    l_obj = l_obj.next
                    continue

                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta is not None:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)

                    # Tensor output meta
                    if getattr(user_meta, "base_meta", None):
                        if getattr(user_meta.base_meta, "meta_type", None) == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                            embedding = None
                            if hasattr(tensor_meta, "output_layers") and tensor_meta.output_layers:
                                outputs = [np.array(layer.buffer, copy=True) for layer in tensor_meta.output_layers]
                                embedding = np.concatenate(outputs).astype(np.float32)
                            elif hasattr(tensor_meta, "output") and tensor_meta.output:
                                embedding = np.array(tensor_meta.output[0], dtype=np.float32)

                            if embedding is not None:
                                # FAISS search
                                D, I = udata["faiss_index"].search(np.expand_dims(embedding, axis=0), 1)
                                matched_id = int(I[0][0])

                                # Push attendance item
                                attendance_item = {
                                    "employee_id": matched_id,
                                    "embedding": embedding.tolist(),
                                    "timestamp": datetime.datetime.utcnow().isoformat()
                                }
                                attendance_q.put(attendance_item)
                            else:
                                logger.warning(f"Tensor meta empty for obj_id {obj_meta.object_id}")

                    l_user_meta = l_user_meta.next
                l_obj = l_obj.next
            l_frame = l_frame.next

    except Exception as e:
        logger.error(f"SGIE probe exception: {e}", exc_info=True)

    return Gst.PadProbeReturn.OK


def attach_sgie_probe(sgie, known_faces, attendance_q, sources):
    """
    Attaches the SGIE probe to the SGIE element.
    udata contains FAISS index, face class ID, and optionally other info.
    """
    try:
        udata = {
            "faiss_index": known_faces.get("faiss_index"),
            "face_class_id": 0,  # update if your PGIE class ID for faces differs
        }
        sgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            sgie_feature_extract_probe,
            udata
        )
        logger.info("SGIE probe attached successfully")
    except Exception as e:
        logger.error(f"Failed to attach SGIE probe: {e}", exc_info=True)
