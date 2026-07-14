"""Face anti-spoofing (liveness) — MiniFASNetV2, in-probe classifier.

Rejects presentation attacks (a phone/printed photo of an enrolled employee held
up to a camera) so attendance is only marked for a *live* person.

Model: Silent-Face-Anti-Spoofing `2.7_80x80_MiniFASNetV2static.onnx` — a 3-class
classifier over an 80x80 crop; class **1 = real/live**, classes 0 & 2 = spoof.
Preprocessing mirrors the model's training + the team's DeepStream `config/spoof.txt`
(RGB, raw 0..255, the 2.7x-expanded face crop).

Runs on the aligned face's *original* bbox (not the ArcFace chip). Fail-open: if the
model is missing/unloadable, liveness is disabled and everyone passes (a warning is
logged once) so anti-spoof can never take attendance down.
"""
import logging
import os

import cv2
import numpy as np

log = logging.getLogger("antispoof")


class AntiSpoof:
    def __init__(self, model_path, scale=2.7, size=80, min_live=0.5):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.scale, self.size, self.min_live = float(scale), int(size), float(min_live)

    @staticmethod
    def _scaled_box(src_w, src_h, box, scale):
        """Silent-Face crop: expand the face box by `scale` about its centre, clamped
        to the frame. Matches the model's training-time preprocessing."""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return 0, 0, min(int(src_w), 1), min(int(src_h), 1)
        scale = min((src_h - 1) / h, min((src_w - 1) / w, scale))
        nw, nh = w * scale, h * scale
        cx, cy = x + w / 2.0, y + h / 2.0
        lx, ly = cx - nw / 2.0, cy - nh / 2.0
        rx, ry = cx + nw / 2.0, cy + nh / 2.0
        if lx < 0:
            rx -= lx; lx = 0
        if ly < 0:
            ry -= ly; ly = 0
        if rx > src_w - 1:
            lx -= (rx - src_w + 1); rx = src_w - 1
        if ry > src_h - 1:
            ly -= (ry - src_h + 1); ry = src_h - 1
        return int(lx), int(ly), int(rx), int(ry)

    def score(self, frame_bgr, box):
        """box = (x, y, w, h) face bbox in frame pixels.
        Returns (is_live: bool, live_prob: float in 0..1)."""
        h, w = frame_bgr.shape[:2]
        lx, ly, rx, ry = self._scaled_box(w, h, box, self.scale)
        crop = frame_bgr[ly:ry, lx:rx]
        if crop.size == 0:
            return True, 1.0  # nothing to check -> fail-open
        chip = cv2.resize(crop, (self.size, self.size))
        rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB).astype(np.float32)   # RGB, raw 0..255
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])           # 1x3x80x80
        out = self.sess.run([self.out_name], {self.in_name: x})[0].ravel()
        ex = np.exp(out - out.max())
        sm = ex / ex.sum()
        live = float(sm[1])                                             # class 1 = real
        return live >= self.min_live, live


def load(model_path, scale=2.7, size=80, min_live=0.5):
    """Build an AntiSpoof, or None (fail-open) if the model is absent/unloadable."""
    if not model_path or not os.path.exists(model_path):
        log.warning("anti-spoof model not found (%s) -- liveness check DISABLED (fail-open)", model_path)
        return None
    try:
        a = AntiSpoof(model_path, scale=scale, size=size, min_live=min_live)
        log.info("anti-spoof loaded: %s (min_live=%.2f)", model_path, min_live)
        return a
    except Exception as e:
        log.error("anti-spoof load failed (%s) -- liveness DISABLED: %s", model_path, e)
        return None
