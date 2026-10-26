
import cv2
import numpy as np
from PIL import Image

REFERENCE_5PTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_112(pil_img, landmarks_5pt):
    src = np.asarray(landmarks_5pt, dtype=np.float32)
    arr = np.asarray(pil_img)  # RGB, HxWx3
    M, _ = cv2.estimateAffinePartial2D(src, REFERENCE_5PTS_112, method=cv2.LMEDS)
    if M is None:
        return None
    warped = cv2.warpAffine(arr, M, (112, 112), borderValue=0.0)
    return Image.fromarray(warped)


def detect_and_align(mtcnn, pil_img):
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    return align_112(pil_img, landmarks[idx])
