"""
Face alignment for ArcFace recognition.

The YOLOv8n-face PGIE emits 5 facial landmarks per detection
((x, y, score) x 5) in the 640x640 *network* coordinate space. ArcFace was
trained on faces aligned to a canonical 112x112 template, so feeding raw,
axis-aligned bbox crops (as the legacy pipeline does) costs significant
recognition accuracy. This module:

  1. inverse-letterboxes landmarks from network space back to frame pixels,
  2. estimates a similarity transform (Umeyama) onto the canonical template,
  3. warps the frame crop to an aligned 112x112 chip.

Pure numpy/cv2 (scikit-image is intentionally not a dependency). The exact
same alignment is used for offline gallery enrollment and live inference, so
embeddings are directly comparable.
"""
import numpy as np
import cv2

# Canonical 5-point ArcFace template (insightface standard) for a 112x112 chip.
# Order: left-eye, right-eye, nose, left-mouth-corner, right-mouth-corner.
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

ARCFACE_CHIP_SIZE = 112
NUM_LANDMARKS = 5


def umeyama_similarity(src, dst):
    """
    Estimate the 2x3 similarity transform (uniform scale + rotation +
    translation, no shear/reflection) mapping ``src`` points onto ``dst``.

    Implements Umeyama (1991) via SVD. ``src`` and ``dst`` are (N, 2) arrays.
    Returns a (2, 3) float32 affine matrix M such that
    ``dst ~= M @ [src, 1]``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Covariance of dst vs src.
    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)

    # Reflection guard: ensure a proper rotation (det = +1).
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    # Uniform scale.
    var_src = src_c.var(axis=0).sum()
    scale = 1.0 if var_src < 1e-12 else (D * np.diag(S)).sum() / var_src

    t = dst_mean - scale * (R @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def inverse_letterbox_points(points, frame_w, frame_h, net_w=640, net_h=640):
    """
    Map landmark points from letterboxed network space (``net_w`` x ``net_h``,
    aspect-ratio preserved with symmetric padding -- matching the YOLO config's
    ``maintain-aspect-ratio=1`` + ``symmetric-padding=1``) back to frame pixels.

    ``points`` is an (N, 2) array in network coordinates.
    """
    points = np.asarray(points, dtype=np.float32)
    scale = min(net_w / frame_w, net_h / frame_h)
    pad_x = (net_w - scale * frame_w) / 2.0
    pad_y = (net_h - scale * frame_h) / 2.0
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] - pad_x) / scale
    out[:, 1] = (points[:, 1] - pad_y) / scale
    return out


def parse_landmarks(mask_flat):
    """
    Parse the flat landmark buffer from ``obj_meta.mask_params`` into a
    (5, 2) array of (x, y) points and a (5,) array of per-point scores.

    The YOLO-face parser packs 5 points as (x, y, score) -> 15 floats in
    network (640x640) space (see nvdsparseface_Yolo.cpp::addFaceProposal).
    """
    arr = np.asarray(mask_flat, dtype=np.float32).reshape(-1)
    if arr.size < NUM_LANDMARKS * 3:
        return None, None
    arr = arr[:NUM_LANDMARKS * 3].reshape(NUM_LANDMARKS, 3)
    return arr[:, :2].copy(), arr[:, 2].copy()


def align_chip(frame_bgr_or_rgb, dst_points, size=ARCFACE_CHIP_SIZE):
    """
    Warp ``frame`` so the detected ``dst_points`` (5x2, in frame pixels) land
    on the canonical template, producing a ``size`` x ``size`` aligned chip.

    The image is warped as-is (no color conversion here -- the caller controls
    BGR/RGB ordering). Returns an HxWx3 array of the same dtype as input.
    """
    # We want frame -> template, so estimate the transform that sends the
    # detected points to the template positions.
    M = umeyama_similarity(np.asarray(dst_points, dtype=np.float32),
                           ARCFACE_TEMPLATE_112)
    chip = cv2.warpAffine(frame_bgr_or_rgb, M, (size, size),
                          flags=cv2.INTER_LINEAR, borderValue=0.0)
    return chip


def _self_test():
    """Sanity check: warped landmarks must land on the template within ~1px."""
    rng = np.random.default_rng(0)
    # Build a synthetic affine (scale+rotation+translation) and apply it to the
    # template to fabricate "detected" points, then confirm we recover them.
    theta = 0.3
    s = 1.7
    R = s * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    t = np.array([40.0, 25.0])
    detected = (ARCFACE_TEMPLATE_112 @ R.T) + t

    M = umeyama_similarity(detected.astype(np.float32), ARCFACE_TEMPLATE_112)
    ones = np.ones((NUM_LANDMARKS, 1), dtype=np.float32)
    mapped = (np.hstack([detected, ones]) @ M.T)
    err = np.linalg.norm(mapped - ARCFACE_TEMPLATE_112, axis=1).max()
    assert err < 1e-3, f"Umeyama round-trip error too large: {err}"

    # Inverse-letterbox round trip.
    fw, fh = 1280, 720
    scale = min(640 / fw, 640 / fh)
    pad_x = (640 - scale * fw) / 2.0
    pad_y = (640 - scale * fh) / 2.0
    frame_pts = np.array([[100.0, 200.0], [900.0, 600.0]], dtype=np.float32)
    net_pts = frame_pts * scale + np.array([pad_x, pad_y], dtype=np.float32)
    back = inverse_letterbox_points(net_pts, fw, fh)
    assert np.allclose(back, frame_pts, atol=1e-3), back
    print("face_align self-test OK (umeyama err=%.2e)" % err)


if __name__ == "__main__":
    _self_test()
