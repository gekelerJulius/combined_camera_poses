from typing import List, Tuple, Optional

import numpy as np
import cv2 as cv
from mediapipe.tasks.python.components.containers.landmark import Landmark
from numpy import ndarray
from scipy.spatial.transform import Slerp, Rotation

from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import estimate_extrinsic

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
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


# def get_person_pairs(
#         a: List[Person],
#         b: List[Person],
#         img1,
#         img2,
#         cam_data1: CameraData,
#         cam_data2: CameraData,
# ):
#     """
#     Returns a list of tuples of Person objects, where each tuple contains two Person objects
#     that are the same person in different images.
#     """
#     # record = set()
#
#     # for p1 in a:
#     #     for p2 in b:
#     # sim1 = p1.get_landmark_sim(p2, img1, img2)
#     # sim2 = p2.get_landmark_sim(p1, img2, img1)
#     #
#     # if sim1 is None or sim2 is None:
#     #     continue
#     # record.add((p1, p2, sim1, sim2))
#
#     # Sort the record by the smallest difference between people in a and b
#     # def sort_key(record_item):
#     #     return record_item[2] + record_item[3]
#
#     # sorted_record = sorted(record, key=sort_key)
#
#     cost_matrix = np.zeros((len(a), len(b)))
#     for i, p1 in enumerate(a):
#         for j, p2 in enumerate(b):
#             cost_matrix[i, j] = test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)
#
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     pairs = [(a[i], b[j]) for i, j in zip(row_ind, col_ind)]
#     return pairs


# def get_person_pairs_simple_distance(
#         a: List[Person],
#         b: List[Person],
#         img1,
#         img2,
#         cam_data1: CameraData,
#         cam_data2: CameraData,
# ):
#     """
#     Returns a list of tuples of Person objects, where each tuple contains two Person objects
#     that are the same person in different images.
#     """
#     record = set()
#     for p1 in a:
#         for p2 in b:
#             pose_diff = p1.pose_distance(p2)
#             dom_col1 = get_dominant_color_bbox(img1, p1.bounding_box)
#             dom_col2 = get_dominant_color_bbox(img2, p2.bounding_box)
#             color_diff = np.linalg.norm(dom_col1 - dom_col2)
#             color_diff_norm = color_diff / 255
#
#             if pose_diff is None or color_diff_norm is None:
#                 continue
#
#             cost = pose_diff * color_diff_norm
#             record.add((p1, p2, cost))
#
#     # Sort the record by the smallest cost
#     def sort_key(record_item):
#         return record_item[2]
#
#     sorted_record = sorted(record, key=sort_key)
#     pairs = []
#     for p1, p2, p_diff, c_diff in sorted_record:
#         if p1 in [pair[0] for pair in pairs] or p2 in [pair[1] for pair in pairs]:
#             continue
#         pairs.append((p1, p2))
#
#     # pair_copy = pairs.copy()
#     # for p1, p2 in pair_copy:
#     #     error = test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)
#     #     if error > 5:
#     #         Logger.log(f"Pairing error: {error}", LoggingLevel.DEBUG)
#     #         pairs.remove((p1, p2))
#     return pairs


def is_rotation_matrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def test_pairs(
    pairs: List[Tuple[Person, Person]],
    recorder1: PersonRecorder,
    recorder2: PersonRecorder,
    img1,
    img2,
    cam_data1: CameraData,
    cam_data2: CameraData,
) -> List[Tuple[Person, Person]]:
    if len(pairs) == 0:
        return []

    # res = []
    # for cur_pair in [p for p in pairs]:
    #     p1, p2 = cur_pair
    #     try:
    #         error = test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)
    #         with Divider("Test Pairing"):
    #             Logger.log(f"Pairing error: {error}", LoggingLevel.DEBUG)
    #         show_pair((p1, p2), img1, img2)
    #         res.append(cur_pair)
    #     except AssertionError as e:
    #         Logger.log(f"Assertion error: {e}", LoggingLevel.DEBUG)
    # return res

    # K1: ndarray = cam_data1.intrinsic_matrix
    # K2: ndarray = cam_data2.intrinsic_matrix
    points1 = []
    points2 = []

    for i, (p1, p2) in enumerate(pairs):
        a, b = get_points_for_pair((p1, p2), recorder1, recorder2)
        points1.extend(a)
        points2.extend(b)

    points1 = np.array(points1)
    points2 = np.array(points2)

    # F, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 5, 0.95, 30000)
    # Only keep inlier points
    # points1 = points1[mask.ravel() == 1]
    # points2 = points2[mask.ravel() == 1]
    # test_fundamental_matrix(F, points1, points2)

    # Compute the Essential matrix
    # E = K2.T @ F @ K1
    # # points_1_hom = np.array([[p[0], p[1], 1] for p in points1])
    # # points_2_hom = np.array([[p[0], p[1], 1] for p in points2])
    # # p1_normalized = np.matmul(np.linalg.inv(K1), points_1_hom.T).T
    # # p2_normalized = np.matmul(np.linalg.inv(K2), points_2_hom.T).T
    # # p1_normalized = np.array([[p[0], p[1]] for p in p1_normalized])
    # # p2_normalized = np.array([[p[0], p[1]] for p in p2_normalized])
    # # test_essential_matrix(E, p1_normalized, p2_normalized)

    # E = np.zeros((3, 3))
    # R = np.zeros((3, 3))
    # t = np.zeros((3, 1))
    # cv.recoverPose(
    #     E=E,
    #     points1=points1,
    #     points2=points2,
    #     cameraMatrix1=K1,
    #     cameraMatrix2=K2,
    #     distCoeffs1=None,
    #     distCoeffs2=None,
    #     R=R,
    #     t=t,
    # )
    #
    # with Divider("Estimated R and t"):
    #     Logger.log(f"R: {R}", LoggingLevel.INFO)
    #     Logger.log(f"t: {t}", LoggingLevel.INFO)
    #
    # # Test Repr Error
    # test_repr_error, test_repr_points = calculate_reprojection_error(
    #     points1,
    #     points2,
    #     cam_data2.rotation_between_cameras(cam_data1),
    #     cam_data2.translation_between_cameras(cam_data1),
    #     K1,
    #     K2,
    # )
    # Logger.log(f"Test Reprojection error: {test_repr_error}", LoggingLevel.INFO)
    #
    # #  Calc Repr Error
    # repr_err, repr_points = calculate_reprojection_error(points1, points2, R, t, K1, K2)
    # Logger.log(f"Reprojection error: {repr_err}", LoggingLevel.INFO)
    #
    # # Draw the points
    # for p1, p2, test_repr_point, repr_point in zip(points1, points2, test_repr_points, repr_points):
    #     cv.circle(img2, (int(p2[0]), int(p2[1])), 2, (255, 255, 0), -1)
    #     cv.circle(img2, (int(test_repr_point[0]), int(test_repr_point[1])), 2, (0, 255, 255), -1)
    #     cv.circle(img2, (int(repr_point[0]), int(repr_point[1])), 2, (0, 0, 255), -1)
    # cv.imshow("img2", img2)
    # cv.waitKey(0)

    # except AssertionError as e:
    #     Logger.log(f"Assertion error: {e}", LoggingLevel.WARNING)
    #     return []
    # return []

    # P, points_3d = estimate_projection(points1, points2, K1, K2)
    # if P is None:
    #     Logger.log("Could not estimate projection", LoggingLevel.DEBUG)
    #     return np.inf
    # R: ndarray = P[:, :3]
    # t: ndarray = P[:, 3]
    # if not is_rotation_matrix(R):
    #     Logger.log("Not a rotation matrix", LoggingLevel.DEBUG)
    #     return np.inf
    #
    # # Reproject points1 onto points2
    # points1_reproj: ndarray = np.matmul(R, points_3d.T) + t[:, np.newaxis]
    # points1_reproj = points1_reproj.T
    # points1_reproj = points1_reproj[:, :2] / points1_reproj[:, 2:]
    # points1_reproj = points1_reproj[:, :2]
    # points1_reproj = np.array([[p[0], p[1]] for p in points1_reproj])
    #
    # with Divider("Points 1"):
    #     Logger.log(points1, LoggingLevel.DEBUG)
    # with Divider("Points 1 reproj"):
    #     Logger.log(points1_reproj, LoggingLevel.DEBUG)
    #
    # assert points1_reproj.shape == points1.shape
    # error = np.linalg.norm(points1_reproj - points1)
    # return error


def get_points_for_pair(
    pair: Tuple[Person, Person], recorder1: PersonRecorder, recorder2: PersonRecorder
) -> Tuple[ndarray, ndarray]:
    p1, p2 = pair
    points1 = []
    points2 = []
    crs = PersonRecorder.get_all_corresponding_frame_recordings(
        p1, p2, recorder1, recorder2, (0, np.inf)
    )
    lmks1: List[Landmark] = crs[0]
    lmks2: List[Landmark] = crs[1]
    if lmks1 is None or lmks2 is None:
        return np.array(points1), np.array(points2)

    for lmk1, lmk2 in zip(lmks1, lmks2):
        visible_enough = lmk1.visibility > 0.4 and lmk2.visibility > 0.4
        if visible_enough:
            points1.append([lmk1.x, lmk1.y])
            points2.append([lmk2.x, lmk2.y])
    return np.array(points1), np.array(points2)


def calc_epipolar_error(
    points1: ndarray, points2: ndarray, cam_data1: CameraData, cam_data2: CameraData
) -> float:
    points1 = np.array(points1)
    points2 = np.array(points2)
    assert points1.shape == points2.shape and points1.shape[1] == 2

    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        Logger.log(
            "Not enough points to calculate fundamental matrix", LoggingLevel.WARNING
        )
        return np.inf

    points1_norm = np.dot(
        np.linalg.inv(K1), np.hstack((points1, np.ones((points1.shape[0], 1)))).T
    ).T
    points1_norm = points1_norm[:, :2] / points1_norm[:, 2].reshape(-1, 1)
    points2_norm = np.dot(
        np.linalg.inv(K2), np.hstack((points2, np.ones((points2.shape[0], 1)))).T
    ).T
    points2_norm = points2_norm[:, :2] / points2_norm[:, 2].reshape(-1, 1)

    E, mask = cv.findEssentialMat(
        points1_norm,
        points2_norm,
        np.eye(3),
        method=cv.RANSAC,
        prob=0.99,
        threshold=2.0,
        maxIters=100000,
    )

    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, E)
    lines1 = lines1.reshape(-1, 3)

    error = np.abs(
        lines1[:, 0] * points1_norm[:, 0]
        + lines1[:, 1] * points1_norm[:, 1]
        + lines1[:, 2]
    ) / np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)
    mean_error = float(np.mean(error))
    return mean_error


def drawlines(img1, img2, lines1: ndarray, lines2: ndarray):
    for a, b, c in lines1:
        drawline(img1, a, b, c)
    for a, b, c in lines2:
        drawline(img2, a, b, c)


def drawline(img, a, b, c):
    a = int(a)
    b = int(b)
    c = int(c)
    cv.line(
        img, (0, -c // b), (img.shape[1], -(c + a * img.shape[1]) // b), (0, 0, 255), 1
    )


def test_pairing(
    person1: Person,
    person2: Person,
    img1,
    img2,
    cam_data1: CameraData,
    cam_data2: CameraData,
) -> float:
    """
    Tests if two Person objects are the same person by calculating the reprojection error
    of the 3D points of one person onto the other person's image.

    Returns the reprojection error.
    """
    pass
    # common_indices = Person.get_common_visible_landmark_indexes(person1, person2)
    # points1 = person1.get_pose_landmarks_numpy()[common_indices]
    # points2 = person2.get_pose_landmarks_numpy()[common_indices]
    # points1 = np.array([[p[0], p[1]] for p in points1])
    # points2 = np.array([[p[0], p[1]] for p in points2])
    # K1 = cam_data1.intrinsic_matrix
    # K2 = cam_data2.intrinsic_matrix
    #
    # if points1.shape[0] < 8 or points2.shape[0] < 8:
    #     Logger.log(
    #         "Not enough points to calculate fundamental matrix", LoggingLevel.WARNING
    #     )
    #     return math.inf
    #
    # F, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 3, 0.99, 3000)
    # test_fundamental_matrix(F, points1, points2)
    #
    # # Only keep inlier points
    # points1 = points1[mask.ravel() == 1]
    # points2 = points2[mask.ravel() == 1]
    #
    # # Compute the Essential matrix
    # E = K2.T @ F @ K1
    #
    # points_1_hom = np.array([[p[0], p[1], 1] for p in points1])
    # points_2_hom = np.array([[p[0], p[1], 1] for p in points2])
    # p1_normalized = np.matmul(np.linalg.inv(K1), points_1_hom.T).T
    # p2_normalized = np.matmul(np.linalg.inv(K2), points_2_hom.T).T
    # p1_normalized = np.array([[p[0], p[1]] for p in p1_normalized])
    # p2_normalized = np.array([[p[0], p[1]] for p in p2_normalized])
    # test_essential_matrix(E, p1_normalized, p2_normalized)
    #
    # R = np.zeros((3, 3))
    # t = np.zeros((3, 1))
    # cv.recoverPose(
    #     E=E,
    #     points1=p1_normalized,
    #     points2=p2_normalized,
    #     cameraMatrix1=K1,
    #     cameraMatrix2=K2,
    #     distCoeffs1=None,
    #     distCoeffs2=None,
    #     R=R,
    #     t=t,
    #     mask=mask,
    # )
    #
    # repr_err, _ = calculate_reprojection_error(points1, points2, R, t, K1, K2)
    # return repr_err

    # points1_2d = np.array([[p1[0], p1[1]] for p1 in points1], dtype=np.float32)
    # points2_2d = np.array([[p2[0], p2[1]] for p2 in points2], dtype=np.float32)
    # K1 = np.array(K1, dtype=np.float32)
    # K2 = np.array(K2, dtype=np.float32)
    #
    # # TODO: Check if using cv2.stereoRectifyUncalibrated() would make sense here
    # P, points_3d = estimate_projection(points1_2d, points2_2d, K1, K2)
    # if P is None or points_3d is None:
    #     Logger.log("Could not estimate projection", LoggingLevel.DEBUG)
    #     return np.inf
    #
    # reprojected_points1 = project_points(points_3d, K1, P)
    # reprojected_points2 = project_points(points_3d, K2, P)
    # reprojection_error1 = np.mean(np.linalg.norm(points1_2d - reprojected_points1, axis=1))
    # reprojection_error2 = np.mean(np.linalg.norm(points2_2d - reprojected_points2, axis=1))
    #
    # mean_reprojection_error = (reprojection_error1 + reprojection_error2) / 2
    # Logger.log(mean_reprojection_error, LoggingLevel.DEBUG, label="Mean Reprojection Error")
    # return mean_reprojection_error


def test_fundamental_matrix(F: ndarray, pts1: ndarray, pts2: ndarray) -> None:
    """
    Tests if the fundamental matrix F is correct by checking the epipolar constraint and the determinant.
    """
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    # Logger.log("Testing fundamental matrix", LoggingLevel.DEBUG)
    # Logger.log(F, LoggingLevel.DEBUG, label="Fundamental Matrix")
    # Logger.log(pts1, LoggingLevel.DEBUG, label="Points 1")
    # Logger.log(pts2, LoggingLevel.DEBUG, label="Points 2")
    assert_custom(
        pts1.shape[0] == pts2.shape[0],
        f"Number of corresponding points must be equal but are {pts1.shape[0]} and {pts2.shape[0]}.",
    )
    assert_custom(
        pts1.shape[1] == 2 and pts2.shape[1] == 2,
        f"Points must have shape (n, 2) but have shape {pts1.shape} and {pts2.shape}.",
    )
    assert_custom(F.shape == (3, 3), "Fundamental matrix must have shape (3, 3).")

    for pt1, pt2 in zip(pts1, pts2):
        pt1 = np.append(pt1, 1)
        pt2 = np.append(pt2, 1)
        error = np.dot(np.dot(pt2.T, F), pt1)
        assert_custom(
            abs(error) < 0.5, f"Epipolar constraint violated. Error: {error:.6f}"
        )

    # check determinant
    det = np.linalg.det(F)
    assert_custom(
        abs(det) < 1e-4,
        f"Determinant of the fundamental matrix must be close to zero. Det: {det:.6f}",
    )


def test_essential_matrix(E: ndarray, pts1: ndarray, pts2: ndarray) -> None:
    """
    Tests if the essential matrix E is correct by checking the epipolar constraint and the determinant.
    """
    # Logger.log("Testing essential matrix", LoggingLevel.DEBUG)
    # Logger.log(f"Essential matrix: \n{E}", LoggingLevel.DEBUG)
    # Logger.log(f"Points 1: \n{pts1}", LoggingLevel.DEBUG)
    # Logger.log(f"Points 2: \n{pts2}", LoggingLevel.DEBUG)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    if pts1.shape[1] == 2:
        pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    assert_custom(
        pts1.shape[0] == pts2.shape[0],
        f"Number of corresponding points must be equal but are {pts1.shape[0]} and {pts2.shape[0]}.",
    )
    assert_custom(
        pts1.shape[1] == 3 and pts2.shape[1] == 3,
        f"Points must have shape (n, 2) but have shape {pts1.shape} and {pts2.shape}.",
    )
    assert_custom(E.shape == (3, 3), "Essential matrix must have shape (3, 3).")
    assert_custom(
        np.linalg.det(E) < 1e-4,
        f"Determinant of the essential matrix must be close to zero. Det: {np.linalg.det(E):.6f}",
    )
    _, S, _ = np.linalg.svd(E)
    assert_custom(
        abs(S[0] - S[1]) < 1,
        f"Singular values of the essential matrix must be equal. Diff: {abs(S[0] - S[1]):.6f}",
    )
    assert_custom(
        abs(S[2]) < 1e-3,
        f"Third singular value of the essential matrix must be close to zero. Singular value: {S[2]:.6f}",
    )

    for pt1, pt2 in zip(pts1, pts2):
        error = np.abs(np.dot(np.dot(pt2.T, E), pt1))
        assert_custom(error < 0.5, f"Epipolar constraint violated. Error: {error:.6f}")


def assert_custom(condition, message):
    if not condition:
        raise AssertionError(message)


def show_pair(pair: Tuple[Person, Person], img1, img2):
    copy1 = img1.copy()
    copy2 = img2.copy()
    p1, p2 = pair
    p1.draw(copy1)
    p2.draw(copy2)
    cv.imshow("img1", copy1)
    cv.imshow("img2", copy2)
    cv.waitKey(0)
    cv.destroyAllWindows()
