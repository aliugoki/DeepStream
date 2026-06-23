# utils/probe/face_features.py

import numpy as np
import pyds
from .constants import FEATURE_VECTOR_SIZE

def get_face_feature(obj_meta):
    l_user = obj_meta.obj_user_meta_list
    while l_user:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            layer = pyds.get_nvds_LayerInfo(tensor, 0)
            features = [pyds.get_detections(layer.buffer, i) for i in range(FEATURE_VECTOR_SIZE)]
            res = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(res)
            return (res / norm).reshape((1, FEATURE_VECTOR_SIZE)) if norm > 0 else None
        l_user = l_user.next
    return None

def match_faces(feature, loaded_faces):
    best_id, best_score = None, -1.0
    feat_flat = feature.flatten()
    for uid, known in loaded_faces.items():
        score = np.dot(feat_flat, known.flatten())
        if score > best_score:
            best_id, best_score = uid, score
    return best_id, best_score
