import json
import math
import os
import time
from typing import List

import numpy as np
import cv2 as cv
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from poseviz import PoseViz, ViewInfo
from scipy.optimize import linear_sum_assignment

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.true_person_loader import TruePersonLoader
from classes.unity_person import load_points
from consts.consts import LANDMARK_NAMES, CONNECTIONS_LIST, NAMES_LIST
from consts.mixamo_mapping import from_mixamo
from enums.logging_levels import LoggingLevel
from functions.funcs import triangulate_points, project_points, estimate_projection


def get_person_pairs(
        a: List[Person],
        b: List[Person],
        img1,
        img2,
        cam_data1: CameraData,
        cam_data2: CameraData,
):
    """
    Returns a list of tuples of Person objects, where each tuple contains two Person objects
    that are the same person in different images.
    """
    # record = set()

    # for p1 in a:
    #     for p2 in b:
    # sim1 = p1.get_landmark_sim(p2, img1, img2)
    # sim2 = p2.get_landmark_sim(p1, img2, img1)
    #
    # if sim1 is None or sim2 is None:
    #     continue
    # record.add((p1, p2, sim1, sim2))

    # Sort the record by the smallest difference between people in a and b
    # def sort_key(record_item):
    #     return record_item[2] + record_item[3]

    # sorted_record = sorted(record, key=sort_key)

    cost_matrix = np.zeros((len(a), len(b)))
    for i, p1 in enumerate(a):
        for j, p2 in enumerate(b):
            cost_matrix[i, j] = test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = [(a[i], b[j]) for i, j in zip(row_ind, col_ind)]
    return pairs


def get_person_pairs_simple_distance(
        a: List[Person],
        b: List[Person],
        img1,
        img2,
        cam_data1: CameraData,
        cam_data2: CameraData,
):
    """
    Returns a list of tuples of Person objects, where each tuple contains two Person objects
    that are the same person in different images.
    """
    record = set()
    for p1 in a:
        for p2 in b:
            diff1 = p1.pose_distance(p2)
            diff2 = p2.pose_distance(p1)

            if diff1 is None or diff2 is None:
                continue
            record.add((p1, p2, diff1, diff2))

    # Sort the record by the smallest difference between people in a and b
    def sort_key(record_item):
        return -(record_item[2] + record_item[3])

    sorted_record = sorted(record, key=sort_key)
    pairs = []
    for p1, p2, diff1, diff2 in sorted_record:
        if p1 in [pair[0] for pair in pairs] or p2 in [pair[1] for pair in pairs]:
            continue
        pairs.append((p1, p2))

    pair_copy = pairs.copy()
    for p1, p2 in pair_copy:
        error = test_pairing(p1, p2, img1, img2, cam_data1, cam_data2)
        if error > 5:
            Logger.log(f"Pairing error: {error}", LoggingLevel.DEBUG)
            # copy1 = img1.copy()
            # copy2 = img2.copy()
            # p1.draw(copy1)
            # p2.draw(copy2)
            # cv.destroyAllWindows()
            # cv.imshow("img1", copy1)
            # cv.imshow("img2", copy2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            pairs.remove((p1, p2))
    return pairs


def is_rotation_matrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


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
    # Logger.log(cam_data1.intrinsic_matrix, LoggingLevel.DEBUG, label="cam_data1.intrinsic_matrix")
    # Logger.log(cam_data2.intrinsic_matrix, LoggingLevel.DEBUG, label="cam_data2.intrinsic_matrix")

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

    # TODO: Check if using cv2.stereoRectifyUncalibrated() would make sense here
    P, points_3d = estimate_projection(points1, points2, K1, K2)
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
