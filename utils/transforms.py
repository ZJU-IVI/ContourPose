import numpy as np
import torch
import cv2


class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.x = data.x / 255.0
        data.x -= self.mean
        data.x /= self.std
        data.x = torch.tensor(data.x, dtype=torch.float32)
        return data


def rotate_img(img, mask, gt_contour, position_3d, K, rot_ang_min, rot_ang_max):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    rotation_axis = position_3d / np.linalg.norm(position_3d)
    R_norm = cv2.Rodrigues(rotation_axis * degree / 180.0 * np.pi)
    K = np.array(K, dtype=np.float32)
    K_inv = np.linalg.inv(K)
    H = np.dot(np.dot(K, R_norm[0].T), K_inv)

    img = cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpPerspective(
        mask,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    gt_contour = cv2.warpPerspective(
        gt_contour,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img, mask, gt_contour, R_norm[0].T
