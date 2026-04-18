import cv2
import numpy as np

def difference_heatmap(img1, img2, save_path):
    """
    img1, img2: uint8 RGB images
    """
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)

    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)
