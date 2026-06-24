# utils/probe/encoder.py

import pyds

_ctx = None

def get_obj_encoder(gpu=0):
    global _ctx
    if _ctx is None:
        _ctx = pyds.nvds_obj_enc_create_context(gpu)
    return _ctx


def save_face_crop_probe(ctx, gst_buffer, frame_meta, obj_meta, path):
    args = pyds.NvDsObjEncUsrArgs()
    args.saveImg = 1
    args.quality = 90
    args.fileNameImg = path

    pyds.nvds_obj_enc_process(
        ctx,
        args,
        hash(gst_buffer),
        obj_meta,
        frame_meta
    )
