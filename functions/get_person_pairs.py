from typing import List

import numpy as np
import cv2 as cv

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from enums.logging_levels import LoggingLevel
from functions.funcs import triangulate_points, project_points


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
    # Create a Record of all matches between people in a and b and their differences between each other
    # sort the Record by the smallest difference between people in a and b and return the sorted list
    record = set()

    for p1 in a:
        for p2 in b:
            sim1 = p1.get_landmark_sim(p2, img1, img2)
            sim2 = p2.get_landmark_sim(p1, img2, img1)

            if sim1 is None or sim2 is None:
                continue
            record.add((p1, p2, sim1, sim2))

    # Sort the record by the smallest difference between people in a and b
    def sort_key(record_item):
        return record_item[2] + record_item[3]

    sorted_record = sorted(record, key=sort_key)

    if len(sorted_record) > 0:
        first_pair = sorted_record[0]

        # Show first pair for confirmation by user
        # if img1 is not None and img2 is not None:
        #     cv.imshow("First pair1", first_pair[0].draw(img1))
        #     cv.imshow("First pair2", first_pair[1].draw(img2))
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()

        proof_by_refutation(first_pair[0], first_pair[1], img1, img2, cam_data1, cam_data2)

    # Take pairs from the beginning of the sorted record until there are no more people in a or b
    pairs = []
    used_a = set()
    used_b = set()

    for p1, p2, diff1, diff2 in sorted_record:
        if p1 in used_a or p2 in used_b:
            continue
        pairs.append((p1, p2))
        used_a.add(p1)
        used_b.add(p2)

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


def proof_by_refutation(
        person1: Person,
        person2: Person,
        img1,
        img2,
        cam_data1: CameraData,
        cam_data2: CameraData,
):
    """
    Returns True if the assumption pair is correct, False otherwise.
    """
    # If the assumption pair is correct, we can use the pose points as features to compare
    # the two images and see if they are the same person

    p1_landmarks = person1.get_pose_landmarks()
    p2_landmarks = person2.get_pose_landmarks()

    # right_hand1 = p1_landmarks[PoseLandmark.RIGHT_WRIST]
    # right_hand2 = p2_landmarks[PoseLandmark.RIGHT_WRIST]
    #
    # if right_hand1 is None or right_hand2 is None:
    #     return False
    #
    # r1 = np.array([right_hand1.x, right_hand1.y, right_hand1.z])
    # r2 = np.array([right_hand2.x, right_hand2.y, right_hand2.z])

    # Define p1 and p2 as numpy arrays of shape (n, 3) where n is the number of landmarks

    points1: np.ndarray = np.array([])
    points2: np.ndarray = np.array([])

    for i, lmk1 in enumerate(p1_landmarks):
        lmk2 = p2_landmarks[i]
        if (
                lmk1 is None
                or lmk2 is None
                or lmk1.visibility < 0.5
                or lmk2.visibility < 0.5
        ):
            continue

        point1 = np.array([lmk1.x, lmk1.y, 1])
        point2 = np.array([lmk2.x, lmk2.y, 1])

        if points1.size == 0:
            points1 = point1
            points2 = point2
        else:
            points1 = np.vstack((points1, point1))
            points2 = np.vstack((points2, point2))

    if points1.shape[0] < 8:
        Logger.log("Not enough points to calculate fundamental matrix", LoggingLevel.WARNING)
        return False

    # Draw Lines between corresponding points
    # copy1 = img1.copy()
    # copy2 = img2.copy()
    # concat = np.concatenate((copy1, copy2), axis=1)
    #
    # for i in range(points1.shape[0]):
    #     x1, y1 = int(points1[i][0]), int(points1[i][1])
    #     x2, y2 = int(points2[i][0] + copy1.shape[1]), int(points2[i][1])
    #     concat = cv.line(concat, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #
    # cv.imshow("concat", concat)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    F, mask = cv.findFundamentalMat(
        points1, points2, cv.FM_RANSAC, 3, 0.99, None
    )

    # epilines1 = cv.computeCorrespondEpilines(p2, 2, fundamental_matrix)
    # epilines1 = epilines1.reshape(-1, 3)
    # epilines2 = cv.computeCorrespondEpilines(p1, 1, fundamental_matrix)
    # epilines2 = epilines2.reshape(-1, 3)

    # compute essential matrix from fundamental matrix
    K1 = cam_data1.intrinsic_matrix
    K2 = cam_data2.intrinsic_matrix
    E = K1.T @ F @ K2
    Logger.log(E, LoggingLevel.DEBUG)

    P1 = np.dot(K1, cam_data1.extrinsic_matrix3x4)
    P2 = np.dot(K2, cam_data2.extrinsic_matrix3x4)

    points1_2d = points1[:, :2]
    points2_2d = points2[:, :2]

    # Logger.log(points1_2d, LoggingLevel.DEBUG)
    # Logger.log(points2_2d, LoggingLevel.DEBUG)

    points_3d = triangulate_points(points1_2d, points2_2d, P1, P2)
    reprojected_points1 = project_points(points_3d, K1, cam_data1.extrinsic_matrix3x4)
    reprojected_points2 = project_points(points_3d, K2, cam_data2.extrinsic_matrix3x4)

    for i in range(len(points1_2d)):
        Logger.log(points1_2d[i], LoggingLevel.DEBUG)
        Logger.log(reprojected_points1[i], LoggingLevel.DEBUG)

    for i in range(len(points2_2d)):
        Logger.log(points2_2d[i], LoggingLevel.DEBUG)
        Logger.log(reprojected_points2[i], LoggingLevel.DEBUG)

    reprojection_error1 = np.mean(np.linalg.norm(points1_2d - reprojected_points1, axis=1))
    Logger.log(reprojection_error1, LoggingLevel.DEBUG)

    reprojection_error2 = np.mean(np.linalg.norm(points2_2d - reprojected_points2, axis=1))
    Logger.log(reprojection_error2, LoggingLevel.DEBUG)

    # for point in reprojected_points1:
    #     cv.circle(img1, tuple(point.astype(int)), 2, (0, 0, 255), -1)
    #
    # for point in points1_2d:
    #     cv.circle(img1, tuple(point.astype(int)), 2, (0, 255, 0), -1)
    #
    # for point in reprojected_points2:
    #     cv.circle(img2, tuple(point.astype(int)), 2, (0, 0, 255), -1)
    #
    # for point in points2_2d:
    #     cv.circle(img2, tuple(point.astype(int)), 2, (0, 255, 0), -1)
    #
    # cv.imshow("img1", img1)
    # cv.imshow("img2", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # R1, R2, t = cv.decomposeEssentialMat(essential_matrix)

    # Images are undistorted
    # dist_coeffs = np.zeros((4, 1))
    # _, R, t = cv.recoverPose(points1, points2, K1, dist_coeffs, K2, dist_coeffs, E=E)
    #
    # # Create initial camera parameters (R, t) for both cameras
    # camera_params = np.zeros((2, 6))
    # camera_params[1, :3] = cv.Rodrigues(R)[0].ravel()  # Convert rotation matrix to a rotation vector
    # camera_params[1, 3:] = t.ravel()
    #
    # # Triangulate the initial 3D points using the projection matrices for both cameras
    # P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # P2 = K2 @ np.hstack((R, t))
    # points_3d_initial = cv.triangulatePoints(P1, P2, points1.T, points2.T).T
    # points_3d_initial /= points_3d_initial[:, 3, np.newaxis]  # Convert to homogeneous coordinates
    #
    # # Set up camera_indices and point_indices for bundle adjustment
    # camera_indices = np.repeat(np.arange(2), len(points1))
    # point_indices = np.tile(np.arange(len(points1)), 2)
    #
    # # Perform bundle adjustment
    # result = bundle_adjustment(np.vstack((points1, points2)), points_3d_initial[:, :3], camera_indices, point_indices,
    #                            camera_params, points_3d_initial[:, :3])
    #
    # # Extract the optimized camera parameters and 3D points
    # optimized_params = result.x
    # optimized_camera_params = optimized_params[:2 * 6].reshape((2, 6))
    # optimized_points_3d = optimized_params[2 * 6:].reshape((-1, 3))
    #
    # print("Optimized camera parameters:")
    # print(optimized_camera_params)
    #
    # print("Optimized 3D points:")
    # print(optimized_points_3d)

    # Logger.log(LoggingLevel.INFO, "R1: " + str(R1))
    # Logger.log(LoggingLevel.INFO, "R2: " + str(R2))
    # Logger.log(LoggingLevel.INFO, "t: " + str(t))

    # check if R1 and R2 are valid rotation matrices
    # if not is_rotation_matrix(R1) or not is_rotation_matrix(R2):
    #     return False

    # projection_matrix1 = np.hstack((R1, t))
    # projection_matrix2 = np.hstack((R2, t))

    # Logger.log("p1: " + str(p1), LoggingLevel.INFO)
    # Logger.log("p2: " + str(p2), LoggingLevel.INFO)

    # img5 = img1.copy()
    # img6 = img2.copy()
    #
    #
    # for line in epilines1:
    #     x0, y0 = map(int, [0, -line[2] / line[1]])
    #     x1, y1 = map(
    #         int, [img5.shape[1], -(line[2] + line[0] * img5.shape[1]) / line[1]]
    #     )
    #     img5 = cv.line(img5, (x0, y0), (x1, y1), (0, 255, 0), 1)
    #
    # for line in epilines2:
    #     x0, y0 = map(int, [0, -line[2] / line[1]])
    #     x1, y1 = map(
    #         int, [img6.shape[1], -(line[2] + line[0] * img6.shape[1]) / line[1]]
    #     )
    #     img6 = cv.line(img6, (x0, y0), (x1, y1), (0, 255, 0), 1)
    #
    # cv.imshow("img5", img5)
    # cv.imshow("img6", img6)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # K1 = cam_data1.get_camera_matrix()
    # K2 = cam_data2.get_camera_matrix()
    # R = R1
    # T = t
    # P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    # P2 = np.dot(K2, np.hstack((R, T)))

    # triangulate points
    # Logger.log(LoggingLevel.INFO, "p1: " + str(p1))
    # Logger.log(LoggingLevel.INFO, "p2: " + str(p2))
    #
    # p1_2D = np.array([p1[0], p1[1]]).reshape(-1, 1, 2)
    # p2_2D = np.array([p2[0], p2[1]]).reshape(-1, 1, 2)
    #
    # Logger.log(LoggingLevel.INFO, "p1_2D: " + str(p1_2D))
    # Logger.log(LoggingLevel.INFO, "p2_2D: " + str(p2_2D))
    #
    # points4D = cv.triangulatePoints(P1, P2, p1_2D, p2_2D)
    # points_3D = (points4D / points4D[3])[:3].T
    # Logger.log(LoggingLevel.INFO, "3D Points: " + str(points_3D))

    # Warp image 2 to image 1
    # Define the motion model

    # Homography
    # H, _ = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
    return True
