# utils/probe/quality.py

import cv2
import numpy as np


def sharpness_score(face_img):
    """
    Laplacian variance based sharpness.
    Typical values:
      blurry   < 80
      usable   80–120
      sharp    > 120
    """
    if face_img is None or face_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_area(obj_meta):
    """
    Normalized face area score (0–1)
    """
    rect = obj_meta.rect_params
    area = rect.width * rect.height

    # clamp (prevents domination by very large faces)
    return min(area / (300 * 300), 1.0)


def face_aspect_score(obj_meta):
    """
    Frontal face heuristic using aspect ratio.
    Ideal frontal ≈ square bounding box.
    """
    rect = obj_meta.rect_params
    if rect.height <= 0:
        return 0.0

    aspect = rect.width / rect.height

    # ideal ~ 0.9–1.1
    deviation = abs(aspect - 1.0)
    score = max(0.0, 1.0 - deviation)

    return min(score, 1.0)
