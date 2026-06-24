import logging
import numpy as np
import datetime
import pyds
from .attendance_worker import attendance_q

logger = logging.getLogger("DeepStream.SGIEProbe")

def sgie_feature_extract_probe(pad, info, user_data):
    try:
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
        if not batch_meta:
            logger.warning("No batch meta")
            return Gst.PadProbeReturn.OK

        if "faiss_index" not in user_data:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if obj_meta.class_id != user_data.get("face_class_id", 0):
                    l_obj = l_obj.next
                    continue

                # Iterate user metadata
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)

                    if getattr(user_meta, "base_meta", None):
                        if getattr(user_meta.base_meta, "meta_type", None) == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                            if hasattr(tensor_meta, "output_layers") and tensor_meta.output_layers:
                                outputs = [np.array(layer.buffer, copy=True) for layer in tensor_meta.output_layers]
                                embedding = np.concatenate(outputs).astype(np.float32)
                            elif hasattr(tensor_meta, "output") and tensor_meta.output:
                                embedding = np.array(tensor_meta.output[0], dtype=np.float32)
                            else:
                                logger.warning(f"Tensor meta empty for obj_id {obj_meta.object_id}")
                                l_user_meta = l_user_meta.next
                                continue

                            # FAISS search
                            D, I = user_data["faiss_index"].search(np.expand_dims(embedding, axis=0), 1)
                            matched_id = int(I[0][0])

                            # Send to attendance queue
                            attendance_item = {
                                "employee_id": matched_id,
                                "embedding": embedding.tolist(),
                                "timestamp": datetime.datetime.utcnow().isoformat()
                            }
                            attendance_q.put(attendance_item)

                    l_user_meta = l_user_meta.next
                l_obj = l_obj.next
            l_frame = l_frame.next

    except Exception as e:
        logger.error(f"SGIE probe exception: {e}")

    return Gst.PadProbeReturn.OK
