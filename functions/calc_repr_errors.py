from typing import Tuple

import numpy as np
from numpy import ndarray
import cv2 as cv

from classes.camera_data import CameraData


def triangulate_3d_points(
        points1_img: ndarray,
        points2_img: ndarray,
        proj_matr1: ndarray,
        proj_matr2: ndarray,
) -> ndarray:
    assert points1_img.shape == points2_img.shape == (len(points1_img), 2)
    points4d = cv.triangulatePoints(
        projMatr1=proj_matr1,
        projMatr2=proj_matr2,
        projPoints1=points1_img.T,
        projPoints2=points2_img.T,
    )
    points3d = points4d[:3, :] / points4d[3, :]
    points3d = points3d.T
    return points3d


def calc_reprojection_errors(
        points1_img: ndarray,
        points2_img: ndarray,
        cam_data1: CameraData,
        cam_data2: CameraData,
) -> Tuple[float, float]:
    assert points1_img.shape == points2_img.shape == (len(points1_img), 2)
    proj_matr1 = cam_data1.get_projection_matrix()
    proj_matr2 = cam_data2.get_projection_matrix()
    points3d = triangulate_3d_points(points1_img, points2_img, proj_matr1, proj_matr2)
    points_cam1 = cam_data1.points_from_world_to_camera(points3d)
    points_cam1_img = cam_data1.points_from_camera_to_image(points_cam1)
    points_cam2 = cam_data2.points_from_world_to_camera(points3d)
    points_cam2_img = cam_data2.points_from_camera_to_image(points_cam2)

    repr_err1 = np.linalg.norm(points1_img - points_cam1_img, axis=1)
    repr_err2 = np.linalg.norm(points2_img - points_cam2_img, axis=1)
    mean1 = float(np.mean(repr_err1))
    mean2 = float(np.mean(repr_err2))
    return mean1, mean2
