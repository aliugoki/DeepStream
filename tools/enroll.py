#!/usr/bin/env python3
"""
Offline gallery (re-)enrollment for the aligned ArcFace pipeline.

Why this exists: enabling face alignment changes the embedding space. The
legacy gallery (*.npy) was produced by the *unaligned* SGIE, so it is invalid
once alignment is on. This tool regenerates the gallery from the source face
images using the SAME alignment + embedder the live pipeline uses, so live
(aligned) probes and gallery vectors are directly comparable.

Pipeline per image  <known_faces>/<emp_id>.<png|jpg>:
    detect (YOLOv8n-face ONNX) -> pick face -> inverse-letterbox landmarks
    -> Umeyama align to 112x112 -> ArcFace embed -> L2-normalize
    -> atomic write <emp_id>.npy   (shape (512,1), matching load_faces)

Runs fully on CPU (onnxruntime) -- no GPU/DeepStream required -- so the whole
alignment+embedding path is verifiable offline before any on-device work.

Usage:
    python3 tools/enroll.py --all
    python3 tools/enroll.py --image data/known_faces/101.png --dry-run
"""
import os
import sys
import json
import argparse
import logging

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_align import (align_chip, inverse_letterbox_points,
                              NUM_LANDMARKS)
from utils.arcface_embedder import make_embedder, l2norm, EMBED_DIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("enroll")

NET = 640
SCORE_THRESH = 0.25          # matches config_yolo.txt pre-cluster-threshold
NMS_IOU = 0.45               # matches nvdsparseface_Yolo.cpp NMS_THRESH
MIN_LANDMARK_SCORE = 0.30    # reject faces with poor landmark confidence
AMBIGUITY_AREA_RATIO = 0.60  # warn if a 2nd face is >=60% of the largest area
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
EMBEDDER_TAG = "aligned_arcface_v1"


def letterbox(img, net=NET):
    h, w = img.shape[:2]
    scale = min(net / w, net / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((net, net, 3), 114, np.uint8)
    px, py = (net - nw) // 2, (net - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def nms(boxes_xyxy, scores, iou_thresh):
    """Standard greedy NMS; boxes as (N,4) xyxy. Returns kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


class YoloFace:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(onnx_path, providers=prov or ["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def detect(self, img_bgr):
        """Return list of dicts {box_xyxy(net), score, landmarks(5x2 net)}."""
        blob = letterbox(img_bgr)
        x = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        boxes, scores, lmks = self.sess.run(None, {self.in_name: x})
        boxes, scores, lmks = boxes[0], scores[0].reshape(-1), lmks[0]
        keep_mask = scores >= SCORE_THRESH
        boxes, scores, lmks = boxes[keep_mask], scores[keep_mask], lmks[keep_mask]
        if len(boxes) == 0:
            return []
        # boxes are xc,yc,w,h in net space -> xyxy.
        xc, yc, bw, bh = boxes.T
        xyxy = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
        kept = nms(xyxy, scores, NMS_IOU)
        out = []
        for i in kept:
            pts = lmks[i].reshape(NUM_LANDMARKS, 3)
            out.append({"box": xyxy[i], "score": float(scores[i]),
                        "landmarks": pts[:, :2], "lmk_score": float(pts[:, 2].mean()),
                        "area": float((xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]))})
        return out


def select_face(dets, emp_id):
    """Pick the enrollment face; flag ambiguity rather than silently guessing."""
    if not dets:
        return None, "no_face"
    dets = sorted(dets, key=lambda d: d["area"], reverse=True)
    primary = dets[0]
    if len(dets) > 1:
        ratio = dets[1]["area"] / max(primary["area"], 1e-9)
        if ratio >= AMBIGUITY_AREA_RATIO:
            log.warning("[%s] %d faces, 2nd is %.0f%% of largest -- AMBIGUOUS, "
                        "review manually.", emp_id, len(dets), ratio * 100)
            return primary, "ambiguous"
    if primary["lmk_score"] < MIN_LANDMARK_SCORE:
        return primary, "low_landmark_score"
    return primary, "ok"


def atomic_save_npy(path, array):
    tmp = path + ".tmp"
    np.save(tmp, array)
    # np.save appends .npy if missing; normalize.
    if not os.path.exists(tmp) and os.path.exists(tmp + ".npy"):
        tmp = tmp + ".npy"
    os.replace(tmp, path)


def enroll_image(img_path, detector, embedder, out_dir, dry_run=False):
    emp_id = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        log.error("[%s] cannot read image %s", emp_id, img_path)
        return False
    dets = detector.detect(img)
    face, status = select_face(dets, emp_id)
    if face is None or status in ("no_face", "low_landmark_score"):
        log.error("[%s] skipped (%s, %d detections)", emp_id, status, len(dets))
        return False

    h, w = img.shape[:2]
    pts = inverse_letterbox_points(face["landmarks"], w, h, NET, NET)
    chip = align_chip(img, pts)                       # BGR 112x112
    emb = embedder.embed(chip)                        # (1, 512) L2-normalized
    vec = emb.reshape(EMBED_DIM, 1).astype(np.float32)

    if dry_run:
        log.info("[%s] OK (det_score=%.3f lmk=%.3f status=%s) -> would write %s.npy",
                 emp_id, face["score"], face["lmk_score"], status,
                 os.path.join(out_dir, emp_id))
        return True
    atomic_save_npy(os.path.join(out_dir, f"{emp_id}.npy"), vec)
    log.info("[%s] enrolled (det_score=%.3f lmk=%.3f status=%s)",
             emp_id, face["score"], face["lmk_score"], status)
    return True


def main():
    ap = argparse.ArgumentParser(description="Aligned ArcFace gallery enrollment")
    ap.add_argument("--known-dir", default="data/known_faces")
    ap.add_argument("--yolo", default="models/yolov8n_face/myyolo.onnx")
    ap.add_argument("--arcface", default="models/arcface/arcface.onnx")
    ap.add_argument("--engine", default="models/arcface/arc1.engine",
                    help="TensorRT engine (used when --embedder trt)")
    ap.add_argument("--embedder", choices=["onnx", "trt"], default="onnx")
    ap.add_argument("--image", help="enroll a single image")
    ap.add_argument("--all", action="store_true", help="enroll every image in --known-dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    detector = YoloFace(args.yolo)
    model_path = args.engine if args.embedder == "trt" else args.arcface
    embedder = make_embedder(args.embedder, model_path)

    if args.image:
        targets = [args.image]
    elif args.all:
        targets = sorted(os.path.join(args.known_dir, f)
                         for f in os.listdir(args.known_dir)
                         if f.lower().endswith(IMG_EXTS))
    else:
        ap.error("specify --image <path> or --all")

    ok = sum(enroll_image(t, detector, embedder, args.known_dir, args.dry_run)
             for t in targets)
    log.info("Enrolled %d/%d images.", ok, len(targets))

    if args.all and not args.dry_run:
        meta = {"embedder": EMBEDDER_TAG, "aligned": True,
                "count": ok, "dim": EMBED_DIM}
        with open(os.path.join(args.known_dir, "gallery_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Wrote gallery_meta.json (%s)", EMBEDDER_TAG)


if __name__ == "__main__":
    main()
