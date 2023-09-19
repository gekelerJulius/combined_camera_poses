from typing import List, Optional

import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from numpy import ndarray
from scipy.spatial.transform import Slerp, Rotation

from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import refine_extrinsic_estimation

from classes.camera_data import CameraData
from classes.person import Person
from classes.person_recorder import PersonRecorder
from functions.funcs import (
    normalize_points,
    get_dominant_color_patch,
    colors_diff,
)
from functions.icp import do_icp_correl


def compare_persons(
        p1: Person,
        p2: Person,
        img1,
        img2,
        recorder1: PersonRecorder,
        recorder2: PersonRecorder,
        frame_count: int,
        cam_data1: CameraData,
        cam_data2: CameraData,
        estimated_extr: Optional[np.ndarray] = None,
        estimation_confidence: Optional[float] = None,
) -> float:
    crs = PersonRecorder.get_all_corresponding_frame_recordings(
        p1,
        p2,
        recorder1,
        recorder2,
        (frame_count - 6, frame_count),
        visibility_threshold=0.8,
    )
    lmks1: List[Landmark] = crs[0]
    lmks2: List[Landmark] = crs[1]
    assert len(lmks1) == len(lmks2)

    points1 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks1], dtype=np.float32)
    points2 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks2], dtype=np.float32)
    pts1_normalized, T1 = normalize_points(points1)
    pts2_normalized, T2 = normalize_points(points2)

    icp_transformation, rmse = do_icp_correl(pts1_normalized, pts2_normalized)
    distances: List[float] = []
    for lmk1, lmk2 in zip(lmks1, lmks2):
        if (
                lmk1.x is None
                or lmk1.y is None
                or lmk2.x is None
                or lmk2.y is None
                or lmk1.x < 0
                or lmk1.y < 0
                or lmk2.x < 0
                or lmk2.y < 0
                or img1 is None
                or img2 is None
        ):
            col_diff = 1
        else:
            dom_color_1: ndarray = get_dominant_color_patch(img1, lmk1.x, lmk1.y, 2)
            dom_color_2: ndarray = get_dominant_color_patch(img2, lmk2.x, lmk2.y, 2)
            if np.isnan(dom_color_1).any() or np.isnan(dom_color_2).any():
                col_diff = 1
            else:
                col_diff = colors_diff(np.array([dom_color_1]), np.array([dom_color_2]))

        assert col_diff >= 0 and lmk1.visibility >= 0 and lmk2.visibility >= 0
        factored_dist = col_diff * (1 - lmk1.visibility) * (1 - lmk2.visibility)
        assert (
                factored_dist is not None
                and factored_dist >= 0
                and not np.isnan(factored_dist)
        )
        distances.append(factored_dist)
    distances_np = np.array(distances)
    distances_np = distances_np / np.max(distances_np)
    assert len(distances_np) == len(lmks1) == len(lmks2)

    points1_img = points1[:, :2]
    points2_img = points2[:, :2]

    repr_err = None
    if estimated_extr is not None:
        fake_cam1 = CameraData.from_matrices(
            cam_data1.intrinsic_matrix,
            np.eye(3),
            np.zeros(3),
        )
        fake_cam2 = CameraData.from_matrices(
            cam_data2.intrinsic_matrix,
            estimated_extr[:3, :3],
            estimated_extr[:3, 3],
        )
        err1, err2 = calc_reprojection_errors(
            points1_img,
            points2_img,
            fake_cam1,
            fake_cam2,
        )
        repr_err = (err1 + err2) / 2

    mean_color_dist = float(np.mean(distances_np))
    res = (mean_color_dist ** 2) * (rmse ** 2)
    if repr_err is not None:
        res *= repr_err
    return res
