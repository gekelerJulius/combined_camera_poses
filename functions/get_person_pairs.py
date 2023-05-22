import json
import math
import os
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.tasks.python.components.containers import Landmark
from numpy import ndarray
from poseviz import PoseViz, ViewInfo
from scipy.optimize import linear_sum_assignment

from classes.camera_data import CameraData
from classes.logger import Logger, Divider
from classes.person import Person
from classes.person_recorder import PersonRecorder
from classes.true_person_loader import TruePersonLoader
from classes.unity_person import load_points
from consts.consts import LANDMARK_NAMES, CONNECTIONS_LIST, NAMES_LIST
from consts.mixamo_mapping import from_mixamo
from enums.logging_levels import LoggingLevel
from functions.funcs import triangulate_points, project_points, estimate_projection, get_dominant_color, \
    get_dominant_color_bbox, calc_cos_sim, normalize_points, plot_pose_2d
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
            diff = compare_persons(a.name, b.name, recorder1, recorder2, frame_count)
            cost_matrix[i, j] = diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        pairs.append((recent1[i], recent2[j]))

    # if len(pairs) > 0:
    #     error = test_pairs(pairs, img1, img2, cam_data1, cam_data2)
    #     with Divider("Pairs Error"):
    #         Logger.log(f"Error: {error}", LoggingLevel.INFO)
    return pairs


def compare_persons(
        name1: str,
        name2: str,
        recorder1: PersonRecorder,
        recorder2: PersonRecorder,
        frame_count: int
) -> Optional[float]:
    i = 0
    frame_index = 1
    diff = 0
    plot_id1 = None
    plot_id2 = None
    while frame_index < frame_count:
        p1: Person = recorder1.get_person_at_frame(name1, frame_index)
        p2: Person = recorder2.get_person_at_frame(name2, frame_index)
        frame_index += 1
        if p1 is None or p2 is None:
            continue

        points1 = p1.get_pose_landmarks_numpy()
        points2 = p2.get_pose_landmarks_numpy()

        assert len(points1) == len(points2)

        # plot_id1 = plot_pose_2d(points1, plot_id1)
        # plot_id2 = plot_pose_2d(points2, plot_id2)
        # plt.pause(3)
        # numpy1: ndarray = normalize_points(points1)
        # numpy2: ndarray = normalize_points(points2)
        # if numpy1 is None or numpy2 is None:
        #     Logger.log("One of the numpy arrays is None", LoggingLevel.WARNING)
        #     continue
        #
        # if len(numpy1) != len(numpy2):
        #     Logger.log("The numpy arrays are not the same length", LoggingLevel.WARNING)
        #     continue

        with Divider("Points"):
            Logger.log(f"Points1: {points1}", LoggingLevel.INFO)
            Logger.log(f"Points2: {points2}", LoggingLevel.INFO)

        R, distances = icp(points1, points2)
        numpy_homo = np.hstack((points1, np.ones((len(points1), 1))))
        numpy1_transformed_homo = np.matmul(R, numpy_homo.T).T
        numpy1_transformed = np.array([[n[0], n[1], n[2]] for n in numpy1_transformed_homo])

        sims = np.array([calc_cos_sim(numpy1_transformed[i], points2[i]) for i in range(len(points1))])
        diff += 1 - ((np.mean(sims) + 1) / 2)
        i += 1
    if i > 0:
        res = diff / i
    else:
        res = 1
    return res


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
        img1,
        img2,
        cam_data1: CameraData,
        cam_data2: CameraData,
) -> float:
    if len(pairs) == 0:
        return np.inf
    error = 0
    for p1, p2 in pairs:
        error += test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)
    return error / len(pairs)

    # K1: ndarray = cam_data1.intrinsic_matrix
    # K2: ndarray = cam_data2.intrinsic_matrix
    # persons1: List[Person] = [pair[0] for pair in pairs]
    # persons2: List[Person] = [pair[1] for pair in pairs]
    # common_indices: List[List[int]] = [
    #     Person.get_common_visible_landmark_indexes(persons1[i], persons2[i]) for i in
    #     range(len(persons1))
    # ]
    # print(common_indices)
    #
    # points1 = []
    # points2 = []
    #
    # for i in range(len(persons1)):
    #     lmks1: List[Tuple[float, float]] = [(lmk.x, lmk.y) for lmk in persons1[i].get_pose_landmarks()]
    #     lmks2: List[Tuple[float, float]] = [(lmk.x, lmk.y) for lmk in persons2[i].get_pose_landmarks()]
    #
    #     for j in common_indices[i]:
    #         points1.append(lmks1[j])
    #         points2.append(lmks2[j])
    #
    # points1 = np.array(points1)
    # points2 = np.array(points2)
    #
    # assert len(points1) == len(points2)
    #
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
    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        Logger.log("Not enough points to calculate fundamental matrix", LoggingLevel.WARNING)
        return math.inf

    points1_2d = np.array([[p1[0], p1[1]] for p1 in points1], dtype=np.float32)
    points2_2d = np.array([[p2[0], p2[1]] for p2 in points2], dtype=np.float32)
    K1 = np.array(K1, dtype=np.float32)
    K2 = np.array(K2, dtype=np.float32)

    # Draw points
    for p1, p2 in zip(points1_2d, points2_2d):
        cv.circle(img1, tuple(p1.astype(np.int32)), 2, (0, 0, 255), -1)
        cv.circle(img2, tuple(p2.astype(np.int32)), 2, (0, 0, 255), -1)

    cv.imshow("img1", img1)
    cv.imshow("img2", img2)
    cv.waitKey(0)

    # TODO: Check if using cv2.stereoRectifyUncalibrated() would make sense here
    P, points_3d = estimate_projection(points1_2d, points2_2d, K1, K2)
    if P is None or points_3d is None:
        Logger.log("Could not estimate projection", LoggingLevel.DEBUG)
        return np.inf
    # F, mask = cv.findFundamentalMat(
    #     points1, points2, cv.FM_RANSAC, 3, 0.99, None
    # )

    # E = K1.T @ F @ K2
    # P1 = np.dot(K1, cam_data1.extrinsic_matrix3x4)
    # P2 = np.dot(K2, cam_data2.extrinsic_matrix3x4)

    # points1_2d = points1[:, :2]
    # points2_2d = points2[:, :2]
    # points_3d = triangulate_points(points1_2d, points2_2d, P1, P2)
    # TruePersonLoader.from_3d_points(points_3d, person1.frame_count, TruePersonLoader.load(
    #     "G:\\Uni\\Bachelor\\Project\\combined_camera_poses\\simulation_data\\persons"
    # ))

    # points_3d_b = triangulate_points(points2_2d, points1_2d, P2, P1)
    # points_3d[:, 1] -= 2 * np.max(points_3d[:, 1])

    # box1 = person1.bounding_box
    # box2 = person2.bounding_box

    # viz = PoseViz(joint_names=NAMES_LIST, joint_edges=CONNECTIONS_LIST, world_up=(0, -1, 0), n_views=2)
    # view_info_1: ViewInfo = ViewInfo(
    #     frame=img1,
    #     boxes=np.array([[box1.min_x, box1.min_y, box1.get_width(), box1.get_height()]]),
    #     # poses=np.array([points_3d * 1000]),
    #     poses=np.array([org_points * 1000]),
    #     camera=cam_data1.as_cameralib_camera(),
    #     poses_alt=[],
    #     poses_true=[],
    # )
    #
    # view_info_2: ViewInfo = ViewInfo(
    #     frame=img2,
    #     boxes=np.array([[box2.min_x, box2.min_y, box2.get_width(), box2.get_height()]]),
    #     poses=np.array([]),
    #     camera=cam_data2.as_cameralib_camera(),
    #     poses_alt=[],
    #     poses_true=[],
    # )

    # viz.update(frame=img1, boxes=np.array([[box1.min_x, box1.min_y, box1.get_width(), box1.get_height()]]),
    #            poses=np.array([points_3d * 6000]),
    #            camera=cam_data1.as_cameralib_camera())
    #
    # viz.update_multiview([view_info_1, view_info_2])

    reprojected_points1 = project_points(points_3d, K1, P)
    reprojected_points2 = project_points(points_3d, K2, P)

    reprojection_error1 = np.mean(np.linalg.norm(points1_2d - reprojected_points1, axis=1))
    # Logger.log(reprojection_error1, LoggingLevel.DEBUG, label="Reprojection Error 1")

    reprojection_error2 = np.mean(np.linalg.norm(points2_2d - reprojected_points2, axis=1))
    # Logger.log(reprojection_error2, LoggingLevel.DEBUG, label="Reprojection Error 2")

    mean_reprojection_error = (reprojection_error1 + reprojection_error2) / 2
    Logger.log(mean_reprojection_error, LoggingLevel.DEBUG, label="Mean Reprojection Error")

    # for point in reprojected_points1:
    #     cv.circle(img1, tuple(point.astype(int)), 2, (0, 0, 255, 0.5), -1)
    #
    # for point in points1_2d:
    #     cv.circle(img1, tuple(point.astype(int)), 2, (0, 255, 0, 0.5), -1)
    #
    # for point in reprojected_points2:
    #     cv.circle(img2, tuple(point.astype(int)), 2, (0, 0, 255, 0.5), -1)
    #
    # for point in points2_2d:
    #     cv.circle(img2, tuple(point.astype(int)), 2, (0, 255, 0, 0.5), -1)
    return mean_reprojection_error
