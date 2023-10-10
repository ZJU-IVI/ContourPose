# Compute gaussian kernel
import numpy as np
import cv2


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def generate_heatmap(c, img_height=480, img_width=640, sigma=(25, 25)):
    gaussian_map = np.zeros((c.shape[0], img_height, img_width))
    for i in range(c.shape[0]):
        x = int(c[i, 0])
        y = int(c[i, 1])
        if x >= 640 or y >= 480:
            gaussian_map[i, 0, 0] = 0
        else:
            gaussian_map[i, y, x] = 1
            gaussian_map[i] = cv2.GaussianBlur(gaussian_map[i], sigma, 0)
            am = np.amax(gaussian_map[i])
            gaussian_map[i] /= am / 255
    return gaussian_map / 255


