import cv2
import numpy as np
import math

def generate_angular_tension_labels(
    axial_img,
    num_sectors=12,
    center=None,
    aggregation="percentile"
):
    """
    axial_img: RGB axial map image (H,W,3)
    returns: (num_sectors,) normalized vector
    """

    gray = cv2.cvtColor(axial_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape

    if center is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = center

    sector_angle = 360.0 / num_sectors
    sector_vals = [[] for _ in range(num_sectors)]

    for y in range(H):
        for x in range(W):
            dx = x - cx
            dy = cy - y
            r = math.sqrt(dx*dx + dy*dy)
            if r < 8:
                continue

            angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            idx = int(angle // sector_angle)
            sector_vals[idx].append(gray[y, x])

    tension = np.zeros(num_sectors, dtype=np.float32)
    for i in range(num_sectors):
        if len(sector_vals[i]) == 0:
            tension[i] = 0
        else:
            if aggregation == "mean":
                tension[i] = np.mean(sector_vals[i])
            else:
                tension[i] = np.percentile(sector_vals[i], 90)

    # normalize 0–1
    tension -= tension.min()
    if tension.max() > 0:
        tension /= tension.max()

    return tension
