from typing import List, Optional

import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from numpy import ndarray
from scipy.spatial.transform import Slerp, Rotation

from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import estimate_extrinsic

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
    lmk_indices: List[int] = crs[2]
    assert len(lmks1) == len(lmks2)

    use_colors = False
    points1 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks1], dtype=np.float32)
    points2 = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks2], dtype=np.float32)
    pts1_normalized, T1 = normalize_points(points1)
    pts2_normalized, T2 = normalize_points(points2)

    if use_colors:
        colors1 = np.array(
            [get_dominant_color_patch(img1, lmk.x, lmk.y, 3) for lmk in lmks1],
            dtype=np.float32,
        )
        colors2 = np.array(
            [get_dominant_color_patch(img2, lmk.x, lmk.y, 3) for lmk in lmks2],
            dtype=np.float32,
        )

        nan_indexes = []
        for i, (c1, c2) in enumerate(zip(colors1, colors2)):
            if np.isnan(c1).any() or np.isnan(c2).any():
                nan_indexes.append(i)

        colors1 = np.delete(colors1, nan_indexes, axis=0)
        colors2 = np.delete(colors2, nan_indexes, axis=0)
        pts1_normalized = np.delete(pts1_normalized, nan_indexes, axis=0)
        pts2_normalized = np.delete(pts2_normalized, nan_indexes, axis=0)

        colors1 = colors1 / 255
        colors2 = colors2 / 255
        pts1_normalized = np.hstack((pts1_normalized, colors1))
        pts2_normalized = np.hstack((pts2_normalized, colors2))

    icp_translation, rmse = do_icp_correl(pts1_normalized, pts2_normalized, True)
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
    extr = estimate_extrinsic(
        points1_img,
        points2_img,
        K1=cam_data1.intrinsic_matrix,
        K2=cam_data2.intrinsic_matrix,
    )
    if estimated_extr is not None:
        # Slerp rotation, linear translation
        R1, t1 = extr[:3, :3], extr[:3, 3]
        R2, t2 = estimated_extr[:3, :3], estimated_extr[:3, 3]
        rotations = [Rotation.from_matrix(R1), Rotation.from_matrix(R2)]
        rotations_quat = Rotation.from_quat([r.as_quat() for r in rotations])
        slerp = Slerp([0, 1], rotations_quat)

        if estimation_confidence is None:
            estimation_confidence = 0.5
        R = slerp([estimation_confidence]).as_matrix()[0]
        t = (t1 + t2) / 2
        extr = np.eye(4)
        extr[:3, :3] = R
        extr[:3, 3] = t

    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix
    R1, t1 = np.eye(3), np.zeros((3, 1))
    R2, t2 = extr[:3, :3], extr[:3, 3]
    est_cam_data1 = CameraData.from_matrices(K1, R1, t1)
    est_cam_data2 = CameraData.from_matrices(K2, R2, t2)
    err1, err2 = calc_reprojection_errors(
        points1_img, points2_img, est_cam_data1, est_cam_data2
    )
    mean_err = (err1 + err2) / 2
    mean_color_dist = float(np.mean(distances_np))
    return (mean_color_dist**2) * (mean_err**2) * (rmse**2)