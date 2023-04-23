import xml.etree.ElementTree as ET
from typing import Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from numpy import ndarray
from scipy.linalg import orthogonal_procrustes

from classes.custom_point import CustomPoint
from functions.get_pose import get_pose

mp_pose = mp.solutions.pose


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    avg_dist = np.mean(np.linalg.norm(points - centroid, axis=1))

    if avg_dist == 0:
        print("Points are the same")
        return points, np.eye(3)

    scale = np.sqrt(2) / avg_dist
    translation = -scale * centroid
    T = np.array([[scale, 0, translation[0]], [0, scale, translation[1]], [0, 0, 1]])

    points_norm = np.hstack((points, np.ones((points.shape[0], 1))))
    points_norm = np.dot(T, points_norm.T).T[:, :2]

    return points_norm, T


def box_to_landmarks_list(img, box):
    min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
    cropped = img[min_y:max_y, min_x:max_x]
    cropped, results = get_pose(cropped)

    landmarks = get_landmarks_as_pixel_coordinates(results, cropped)
    for landmark in landmarks:
        landmark.x += min_x
        landmark.y += min_y

    landmarks = [
        landmark for landmark in landmarks if landmark.x != 0 and landmark.y != 0
    ]
    return img, landmarks


def are_landmarks_the_same(landmarks1, landmarks2):
    avg_diff = compare_landmarks(landmarks1, landmarks2)
    print(f"Average difference: {avg_diff}")
    return avg_diff < 5


def calc_cos_sim(v1: ndarray, v2: ndarray) -> float:
    """
    Returns the cosine similarity between two vectors
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    norm = norm1 * norm2

    if norm == 0:
        return 0

    return dot / norm


def compare_landmarks(
    landmarks1, landmarks2
) -> Tuple[Union[float, None], Union[float, None]]:
    if len(landmarks1) != len(landmarks2):
        print("Landmarks are not the same length")
        return None, None

    pose1 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks1])
    pose2 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks2])

    # Use center of hips as origin
    # left_hip1 = pose1[mp_pose.PoseLandmark.LEFT_HIP]
    # right_hip1 = pose1[mp_pose.PoseLandmark.RIGHT_HIP]
    # centroid1 = (left_hip1 + right_hip1) / 2
    # left_hip2 = pose2[mp_pose.PoseLandmark.LEFT_HIP]
    # right_hip2 = pose2[mp_pose.PoseLandmark.RIGHT_HIP]
    # centroid2 = (left_hip2 + right_hip2) / 2
    #
    # pose1 -= centroid1
    # pose2 -= centroid2

    # Scale to same size
    pose1 /= np.linalg.norm(pose1)
    pose2 /= np.linalg.norm(pose2)

    # Brute force 360-degree rotation around y-axis to find best match
    best_cos_sim = 0
    best_rotation = 0
    for i in range(360):
        p1_rotated = rotate_points_y(np.copy(pose1), i)
        avg_cos_sim = 0
        for j in range(len(p1_rotated)):
            avg_cos_sim += calc_cos_sim(p1_rotated[j], pose2[j])

        # print(f"Summed cos sim: {avg_cos_sim} for rotation {i} degrees")
        cos_sim = avg_cos_sim / len(p1_rotated) if len(p1_rotated) > 0 else 0
        # print(f"Avg Cos sim: {cos_sim}")
        if cos_sim > best_cos_sim:
            best_cos_sim = cos_sim
            best_rotation = i
    return 1 - best_cos_sim, best_rotation

    # if not len(pose1) == len(pose2):
    #     # raise Exception("Landmarks are not the same length")
    #     return None, None
    #
    # Q, Scale = orthogonal_procrustes(pose1, pose2)

    # with np.printoptions(precision=3, suppress=True):
    #     print("Scale:")
    #     print(Scale)
    #     print("Q (computed by orthogonal_procrustes):")
    #     print(Q)
    # print("\nCompare Pose1 @ Q with B0.")
    # print("A0 @ Q:")
    # print(pose1 @ Q)
    # print("B0 (should be close to A0 @ Q if the noise parameter was small):")
    # print(pose2)
    # print("Difference:")
    # print(np.linalg.norm(pose1 @ Q - pose2))

    # return np.linalg.norm(pose1 @ Q - pose2)

    # diffs = []
    # if len(landmarks1) != len(landmarks2):
    #     print("Landmarks are not the same length")
    #     return False
    # for i in range(len(landmarks1)):
    #     diffs.append(landmarks1[i].distance(landmarks2[i]))
    #
    # summed = 0
    # for diff in diffs:
    #     summed += diff
    # return summed / len(diffs)


def get_landmarks_as_pixel_coordinates(results, image):
    # Check for NoneType
    if results.pose_landmarks is None:
        return []

    image_height = image.shape[0]
    image_width = image.shape[1]
    return [
        CustomPoint(landmark.x * image_width, landmark.y * image_height, landmark.z)
        for landmark in results.pose_landmarks.landmark
    ]


def draw_landmarks(results, image):
    landmarks = get_landmarks_as_pixel_coordinates(results, image)
    for landmark in landmarks:
        cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, (0, 255, 0), -1)
    return image


def draw_landmarks_list(image, landmarks, with_index=False, color=(0, 255, 0)):
    if with_index:
        for i, landmark in enumerate(landmarks):
            cv2.putText(
                image,
                str(i),
                (int(landmark.x), int(landmark.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                color,
                1,
            )
    else:
        for landmark in landmarks:
            cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, (0, 255, 0), -1)
    return image


def extract_extrinsics_from_xml(xml_file):
    # Extract the rotation vector (rvec) and translation vector (tvec) from the extrinsics file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the rotation vector (rvec)
    rvec = np.zeros((3,), dtype=np.float64)
    for i, value in enumerate(root[0].text.split()):
        rvec[i] = float(value)

    # Extract the translation vector (tvec)
    tvec = np.zeros((3,), dtype=np.float64)
    for i, value in enumerate(root[1].text.split()):
        tvec[i] = float(value)

    return rvec, tvec


def extract_intrinsics_from_xml(xml_file):
    # Extract the camera matrix (camera_matrix) and distortion coefficients (dist_coeffs) from the intrinsics file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the camera matrix (camera_matrix)
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    for i, child in enumerate(root[0]):
        if child.tag == "rows" or child.tag == "cols":
            assert int(child.text) == 3
        elif child.tag == "dt":
            assert child.text == "d"
        elif child.tag == "data":
            split = child.text.split()
            assert len(split) == 9
            camera_matrix[0, 0] = float(split[0])
            camera_matrix[0, 1] = float(split[1])
            camera_matrix[0, 2] = float(split[2])
            camera_matrix[1, 0] = float(split[3])
            camera_matrix[1, 1] = float(split[4])
            camera_matrix[1, 2] = float(split[5])
            camera_matrix[2, 0] = float(split[6])
            camera_matrix[2, 1] = float(split[7])
            camera_matrix[2, 2] = float(split[8])
        else:
            raise ValueError("Invalid tag in XML file")

    # Extract the distortion coefficients (dist_coeffs)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    for i, child in enumerate(root[1]):
        if child.tag == "rows":
            assert int(child.text) == 5
        elif child.tag == "cols":
            assert int(child.text) == 1
        elif child.tag == "dt":
            assert child.text == "d"
        elif child.tag == "data":
            split = child.text.split()
            assert len(split) == 5
            dist_coeffs[0, 0] = float(split[0])
            dist_coeffs[1, 0] = float(split[1])
            dist_coeffs[2, 0] = float(split[2])
            dist_coeffs[3, 0] = float(split[3])
            dist_coeffs[4, 0] = float(split[4])
        else:
            raise ValueError("Invalid tag in XML file")

    return camera_matrix, dist_coeffs


def match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return np.array([kp1[m.queryIdx].pt for m in good_matches]), np.array(
        [kp2[m.trainIdx].pt for m in good_matches]
    )


def points_are_close(x1, y1, x2, y2, threshold=None):
    if threshold is None:
        # Set threshold to 5% of biggest number
        threshold = max(x1, x2, y1, y2) * 0.05
    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold


def rotate_points_y(points: ndarray, angle: int, center: CustomPoint = None):
    """
    Rotate points around the y-axis by a given angle.
    Args:
        points: List of CustomPoint objects, containing x, y, z coordinates.
        angle: Angle to rotate by, in degrees (int).
        center: Center of rotation (CustomPoint object).
    """
    return [rotate_point_y(point, angle, center) for point in points]


def rotate_point_y(point: ndarray, angle: int, center: ndarray = None):
    """
    Rotate a point around the y-axis by a given angle.
    Args:
        point: CustomPoint object, containing x, y, z coordinates.
        angle: Angle to rotate by, in degrees (int).
        center: Center of rotation (CustomPoint object).
    """
    theta = np.deg2rad(angle)
    if center is None:
        center = np.array([0, 0, 0])
    point -= center
    point = np.array(
        [
            point[0] * np.cos(theta) + point[2] * np.sin(theta),
            point[1],
            -point[0] * np.sin(theta) + point[2] * np.cos(theta),
        ]
    )
    point += center
    return point


def my_procrustes(a: np.ndarray, b: np.ndarray):
    mtx1 = np.array(a, dtype=np.double, copy=True)
    mtx2 = np.array(b, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R
