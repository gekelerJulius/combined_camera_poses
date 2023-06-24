from typing import Optional

import numpy as np
from numpy import ndarray
import cv2 as cv


def estimate_extrinsic(
    points1_img: ndarray,
    points2_img: ndarray,
    K1: ndarray,
    K2: ndarray,
) -> Optional[ndarray]:
    assert points1_img.shape == points2_img.shape
    assert points1_img.shape[1] == 2
    if points1_img.shape[0] < 8:
        return None

    points1_img = np.array(points1_img)
    points2_img = np.array(points2_img)
    E = np.zeros((3, 3))
    R = np.zeros((3, 3))
    t = np.zeros((3, 1))
    mask = np.zeros((len(points1_img), 1))

    cv.recoverPose(
        points1=points1_img,
        points2=points2_img,
        cameraMatrix1=K1,
        cameraMatrix2=K2,
        distCoeffs1=None,
        distCoeffs2=None,
        method=cv.RANSAC,
        prob=0.9999,
        threshold=2,
        E=E,
        R=R,
        t=t,
        mask=mask,
    )
    return np.hstack((R, t))
