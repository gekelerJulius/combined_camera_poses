import numpy as np
from numpy import ndarray
import cv2 as cv

from functions.funcs import rotation_matrix_to_angles


def estimate_extrinsic(
        points1_img: ndarray,
        points2_img: ndarray,
        K1: ndarray,
        K2: ndarray,
) -> ndarray:
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
        prob=0.999,
        threshold=3,
        E=E,
        R=R,
        t=t,
        mask=mask,
    )
    return np.hstack((R, t))
