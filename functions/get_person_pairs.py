from typing import List

import numpy as np
import cv2 as cv
import plotly

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from enums.logging_levels import LoggingLevel


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

    first_pair = sorted_record[1]

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

    p1: np.ndarray = np.array([])
    p2: np.ndarray = np.array([])

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

        if p1.size == 0:
            p1 = point1
            p2 = point2
        else:
            p1 = np.vstack((p1, point1))
            p2 = np.vstack((p2, point2))

    # Think about what can be done with the pose points to compare the two images

    # Draw Lines between corresponding points
    copy1 = img1.copy()
    copy2 = img2.copy()
    concat = np.concatenate((copy1, copy2), axis=1)

    for i in range(p1.shape[0]):
        x1, y1 = int(p1[i][0]), int(p1[i][1])
        x2, y2 = int(p2[i][0] + copy1.shape[1]), int(p2[i][1])
        concat = cv.line(concat, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv.imshow("concat", concat)
    cv.waitKey(0)
    cv.destroyAllWindows()

    fundamental_matrix, mask = cv.findFundamentalMat(
        p1, p2, cv.FM_RANSAC, 3, 0.99, None
    )

    # Draw lines on left image corresponding to points in right image
    epilines1 = cv.computeCorrespondEpilines(p2, 2, fundamental_matrix)
    epilines1 = epilines1.reshape(-1, 3)
    epilines2 = cv.computeCorrespondEpilines(p1, 1, fundamental_matrix)
    epilines2 = epilines2.reshape(-1, 3)
    img5 = img1.copy()
    img6 = img2.copy()

    for line in epilines1:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(
            int, [img5.shape[1], -(line[2] + line[0] * img5.shape[1]) / line[1]]
        )
        img5 = cv.line(img5, (x0, y0), (x1, y1), (0, 255, 0), 1)

    for line in epilines2:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(
            int, [img6.shape[1], -(line[2] + line[0] * img6.shape[1]) / line[1]]
        )
        img6 = cv.line(img6, (x0, y0), (x1, y1), (0, 255, 0), 1)

    cv.imshow("img5", img5)
    cv.imshow("img6", img6)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # compute essential matrix from fundamental matrix
    essential_matrix = np.matmul(
        np.matmul(np.transpose(cam_data2.get_camera_matrix()), fundamental_matrix),
        cam_data1.get_camera_matrix(),
    )

    Logger.log(LoggingLevel.INFO, "Essential Matrix: " + str(essential_matrix))

    # decompose essential matrix into rotation and translation
    R1, R2, t = cv.decomposeEssentialMat(essential_matrix)
    Logger.log(LoggingLevel.INFO, "R1: " + str(R1))
    Logger.log(LoggingLevel.INFO, "R2: " + str(R2))
    Logger.log(LoggingLevel.INFO, "t: " + str(t))

    # check if R1 and R2 are valid rotation matrices
    if not is_rotation_matrix(R1) or not is_rotation_matrix(R2):
        return False

    K1 = cam_data1.get_camera_matrix()
    K2 = cam_data2.get_camera_matrix()
    R = R1
    T = t
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K2, np.hstack((R, T)))

    # triangulate points
    Logger.log(LoggingLevel.INFO, "p1: " + str(p1))
    Logger.log(LoggingLevel.INFO, "p2: " + str(p2))

    p1_2D = np.array([p1[0], p1[1]]).reshape(-1, 1, 2)
    p2_2D = np.array([p2[0], p2[1]]).reshape(-1, 1, 2)

    Logger.log(LoggingLevel.INFO, "p1_2D: " + str(p1_2D))
    Logger.log(LoggingLevel.INFO, "p2_2D: " + str(p2_2D))

    points4D = cv.triangulatePoints(P1, P2, p1_2D, p2_2D)
    points_3D = (points4D / points4D[3])[:3].T
    Logger.log(LoggingLevel.INFO, "3D Points: " + str(points_3D))

    # Warp image 2 to image 1
    # Define the motion model

    # Homography
    H, _ = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
    return True
