import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mediapipe.tasks.python.components.containers.landmark import Landmark
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from classes.camera_data import CameraData
from classes.logger import Logger, Divider
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.funcs import calc_cos_sim, normalize_points, plot_pose_2d
from functions.icp import icp


def match_pairs(
        recorder1: PersonRecorder,
        recorder2: PersonRecorder,
        frame_count: int,
        img1,
        img2,
        cam_data1: CameraData,
        cam_data2: CameraData
) -> List[Tuple[Person, Person]]:
    recent1: List[Person] = recorder1.get_recent_persons(frame_count)
    recent2: List[Person] = recorder2.get_recent_persons(frame_count)
    cost_matrix = np.zeros((len(recent1), len(recent2)))

    for i, a in enumerate(recent1):
        for j, b in enumerate(recent2):
            diff = compare_persons(a, b, img1, img2, recorder1, recorder2, frame_count, cam_data1, cam_data2)
            cost_matrix[i, j] = diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        pairs.append((recent1[i], recent2[j]))

    # pairs = test_pairs(pairs, recorder1, recorder2, img1, img2, cam_data1, cam_data2)
    return pairs


def compare_persons(
        p1: Person,
        p2: Person,
        img1,
        img2,
        recorder1: PersonRecorder,
        recorder2: PersonRecorder,
        frame_count: int,
        cam_data1: CameraData,
        cam_data2: CameraData
) -> float:
    crs = PersonRecorder.get_all_corresponding_frame_recordings(p1, p2, recorder1, recorder2)
    lmks1: List[Landmark] = crs[0]
    lmks2: List[Landmark] = crs[1]

    assert len(lmks1) == len(lmks2)

    pts1_normalized, T1 = normalize_points(np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks1], dtype=np.float32))
    pts2_normalized, T2 = normalize_points(np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks2], dtype=np.float32))
    R, distances = icp(pts1_normalized, pts2_normalized)

    if distances is None or len(distances) == 0:
        return np.inf

    # calculate cost
    cost = 0
    for i in range(len(distances)):
        cost += distances[i] * lmks1[i].visibility
    cost /= len(distances)
    return cost


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

    show_pair((pairs[0][0], pairs[0][1]), img1, img2)

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
        crs = PersonRecorder.get_all_corresponding_frame_recordings(p1, p2, recorder1, recorder2)
        lmks1: List[Landmark] = crs[0]
        lmks2: List[Landmark] = crs[1]
        if lmks1 is None or lmks2 is None:
            continue
        # TODO: Implement again
        # for pt in pts1:
        #     points1.append(pt)
        # for pt in pts2:
        #     points2.append(pt)

    points1 = np.array(points1)
    points2 = np.array(points2)

    assert len(points1) == len(points2)

    # Draw the points
    for p1, p2 in zip(points1, points2):
        cv.circle(img1, (int(p1[0]), int(p1[1])), 3, (255, 255, 0), -1)
        cv.circle(img2, (int(p2[0]), int(p2[1])), 3, (255, 255, 0), -1)
    cv.imshow("img1", img1)
    cv.imshow("img2", img2)
    cv.waitKey(0)

    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        Logger.log("Not enough points to calculate fundamental matrix", LoggingLevel.WARNING)
        return []

    try:
        F, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 3, 0.99, 3000)

        # Only keep inlier points
        points1 = points1[mask.ravel() == 1]
        points2 = points2[mask.ravel() == 1]
        test_fundamental_matrix(F, points1, points2)

        # Compute the Essential matrix
        E = K2.T @ F @ K1
        points_1_hom = np.array([[p[0], p[1], 1] for p in points1])
        points_2_hom = np.array([[p[0], p[1], 1] for p in points2])
        p1_normalized = np.matmul(np.linalg.inv(K1), points_1_hom.T).T
        p2_normalized = np.matmul(np.linalg.inv(K2), points_2_hom.T).T
        p1_normalized = np.array([[p[0], p[1]] for p in p1_normalized])
        p2_normalized = np.array([[p[0], p[1]] for p in p2_normalized])
        test_essential_matrix(E, p1_normalized, p2_normalized)

        R = np.zeros((3, 3))
        t = np.zeros((3, 1))
        cv.recoverPose(E=E, points1=p1_normalized, points2=p2_normalized, cameraMatrix1=K1, cameraMatrix2=K2,
                       distCoeffs1=None,
                       distCoeffs2=None, R=R, t=t, mask=mask)

        repr_err = calculate_reprojection_error(points1, points2, K1, R, t)
        with Divider("Reprojection error"):
            Logger.log(f"Reprojection error: {repr_err}", LoggingLevel.INFO)
    except AssertionError as e:
        Logger.log(f"Assertion error: {e}", LoggingLevel.WARNING)
        return []
    return []

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


def drawlines(img1, img2, lines1: ndarray, lines2: ndarray):
    for a, b, c in lines1:
        drawline(img1, a, b, c)
    for a, b, c in lines2:
        drawline(img2, a, b, c)


def drawline(img, a, b, c):
    a = int(a)
    b = int(b)
    c = int(c)
    cv.line(img, (0, -c // b), (img.shape[1], -(c + a * img.shape[1]) // b), (0, 0, 255), 1)


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
    common_indices = Person.get_common_visible_landmark_indexes(person1, person2)
    points1 = person1.get_pose_landmarks_numpy()[common_indices]
    points2 = person2.get_pose_landmarks_numpy()[common_indices]
    points1 = np.array([[p[0], p[1]] for p in points1])
    points2 = np.array([[p[0], p[1]] for p in points2])
    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        Logger.log("Not enough points to calculate fundamental matrix", LoggingLevel.WARNING)
        return math.inf

    F, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 3, 0.99, 3000)
    test_fundamental_matrix(F, points1, points2)

    # Only keep inlier points
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    # Compute the Essential matrix
    E = K2.T @ F @ K1

    points_1_hom = np.array([[p[0], p[1], 1] for p in points1])
    points_2_hom = np.array([[p[0], p[1], 1] for p in points2])
    p1_normalized = np.matmul(np.linalg.inv(K1), points_1_hom.T).T
    p2_normalized = np.matmul(np.linalg.inv(K2), points_2_hom.T).T
    p1_normalized = np.array([[p[0], p[1]] for p in p1_normalized])
    p2_normalized = np.array([[p[0], p[1]] for p in p2_normalized])
    test_essential_matrix(E, p1_normalized, p2_normalized)

    R = np.zeros((3, 3))
    t = np.zeros((3, 1))
    cv.recoverPose(E=E, points1=p1_normalized, points2=p2_normalized, cameraMatrix1=K1, cameraMatrix2=K2,
                   distCoeffs1=None,
                   distCoeffs2=None, R=R, t=t, mask=mask)

    repr_err = calculate_reprojection_error(points1, points2, K1, R, t)
    return repr_err

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


def calculate_reprojection_error(points1: ndarray, points2: ndarray, K: ndarray, R: ndarray, t: ndarray) -> float:
    """
    Calculate reprojection error between two sets of points.
    Arguments:
    - points1, points2: numpy arrays of shape (N, 2)
    - K: camera matrix
    - R: rotation matrix
    - t: translation vector
    Returns:
    - Average reprojection error
    """
    r = cv.Rodrigues(R)[0]
    # Add one additional dimension to the points (homogeneous coordinates)
    points1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_homogeneous = np.hstack((points2, np.ones((points2.shape[0], 1))))

    # Reproject points1 to the second image using the camera matrix, R and t
    proj_res = cv.projectPoints(points1_homogeneous, r, t, K, None)
    reprojected_points: ndarray = proj_res[0]

    # Remove the additional dimension from reprojected points
    reprojected_points = reprojected_points.squeeze()

    # Calculate the Euclidean distance between the reprojected points and points2
    error = np.sqrt(np.sum((reprojected_points - points2_homogeneous[:, :2]) ** 2, axis=1))

    # Return the average reprojection error
    return float(np.mean(error))


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
    assert_custom(pts1.shape[0] == pts2.shape[0],
                  f"Number of corresponding points must be equal but are {pts1.shape[0]} and {pts2.shape[0]}.")
    assert_custom(pts1.shape[1] == 2 and pts2.shape[1] == 2,
                  f"Points must have shape (n, 2) but have shape {pts1.shape} and {pts2.shape}.")
    assert_custom(F.shape == (3, 3), "Fundamental matrix must have shape (3, 3).")

    for pt1, pt2 in zip(pts1, pts2):
        pt1 = np.append(pt1, 1)
        pt2 = np.append(pt2, 1)
        error = np.dot(np.dot(pt2.T, F), pt1)
        assert_custom(abs(error) < 0.5, f"Epipolar constraint violated. Error: {error:.6f}")

    # check determinant
    det = np.linalg.det(F)
    assert_custom(abs(det) < 1e-4, f"Determinant of the fundamental matrix must be close to zero. Det: {det:.6f}")


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
    assert_custom(pts1.shape[0] == pts2.shape[0],
                  f"Number of corresponding points must be equal but are {pts1.shape[0]} and {pts2.shape[0]}.")
    assert_custom(pts1.shape[1] == 3 and pts2.shape[1] == 3,
                  f"Points must have shape (n, 2) but have shape {pts1.shape} and {pts2.shape}.")
    assert_custom(E.shape == (3, 3), "Essential matrix must have shape (3, 3).")
    assert_custom(np.linalg.det(E) < 1e-4,
                  f"Determinant of the essential matrix must be close to zero. Det: {np.linalg.det(E):.6f}")
    _, S, _ = np.linalg.svd(E)
    assert_custom(abs(S[0] - S[1]) < 1,
                  f"Singular values of the essential matrix must be equal. Diff: {abs(S[0] - S[1]):.6f}")
    assert_custom(abs(S[2]) < 1e-3,
                  f"Third singular value of the essential matrix must be close to zero. Singular value: {S[2]:.6f}")

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
