import math
import xml.etree.ElementTree as ET
from typing import Union, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from numpy import ndarray
from scipy.linalg import orthogonal_procrustes
from mpl_toolkits.mplot3d import Axes3D

from classes.PlotService import PlotService
from classes.bounding_box import BoundingBox
from classes.colored_landmark import ColoredLandmark
from classes.custom_point import CustomPoint
from classes.logger import Logger, Divider
from consts.consts import CONNECTIONS_LIST
from enums.logging_levels import LoggingLevel

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# def normalize_points(points):
#     centroid = np.mean(points, axis=0)
#     avg_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
#
#     if avg_dist == 0:
#         Logger.log("All points are the same", LoggingLevel.INFO)
#         return points, np.eye(3)
#
#     scale = np.sqrt(2) / avg_dist
#     translation = -scale * centroid
#     T = np.array([[scale, 0, translation[0]], [0, scale, translation[1]], [0, 0, 1]])
#
#     points_norm = np.hstack((points, np.ones((points.shape[0], 1))))
#     points_norm = np.dot(T, points_norm.T).T[:, :2]
#
#     return points_norm, T


def normalize_points(points: ndarray) -> Tuple[ndarray, ndarray]:
    points = np.array(points)
    assert points.shape[1] == 3, "Points must be 3D"

    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)

    if np.all(std_dev == 0):
        Logger.log("All points are the same", LoggingLevel.INFO)
        return points, np.eye(3)

    points_norm = (points - mean) / std_dev
    transformation4x4 = np.eye(4)
    transformation4x4[:3, :3] = np.diag(1 / std_dev)
    transformation4x4[:3, 3] = -mean / std_dev
    return points_norm, transformation4x4


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
    landmarks1: List[ColoredLandmark], landmarks2: List[ColoredLandmark]
) -> [Union[float, None]]:
    if len(landmarks1) != len(landmarks2):
        Logger.divider()
        print(landmarks1)
        print(landmarks2)
        Logger.divider()
        return None

    if len(landmarks1) < mp_pose.PoseLandmark.RIGHT_HIP.value:
        Logger.log(f"Landmarks have length {len(landmarks1)}", LoggingLevel.WARNING)
        return None

    pose1 = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
            for landmark in landmarks1
        ]
    )

    pose1_avg_colors = np.array(
        [
            [
                landmark.avg_color,
            ]
            for landmark in landmarks1
        ]
    )

    pose1_dom_colors = np.array(
        [
            [
                landmark.dom_color,
            ]
            for landmark in landmarks1
        ]
    )

    pose2 = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
            for landmark in landmarks2
        ]
    )

    pose2_avg_colors = np.array(
        [
            [
                landmark.avg_color,
            ]
            for landmark in landmarks2
        ]
    )

    pose2_dom_colors = np.array(
        [
            [
                landmark.dom_color,
            ]
            for landmark in landmarks2
        ]
    )

    # Use center of hips as origin
    left_hip1 = pose1[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip1 = pose1[mp_pose.PoseLandmark.RIGHT_HIP]
    centroid1 = (left_hip1 + right_hip1) / 2
    pose1 -= centroid1

    left_hip2 = pose2[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip2 = pose2[mp_pose.PoseLandmark.RIGHT_HIP]
    centroid2 = (left_hip2 + right_hip2) / 2
    pose2 -= centroid2

    # Scale to same size
    pose1 /= np.linalg.norm(pose1)
    pose2 /= np.linalg.norm(pose2)

    # Brute force 360-degree rotation around y-axis to find best match
    best_cos_sim = -1
    best_rotation = 0
    plot_id = None
    for i in range(360):
        p1_rotated: ndarray = rotate_points_y(np.copy(pose1), i)
        avg_cos_sim = 0
        for j in range(len(p1_rotated)):
            avg_cos_sim += calc_cos_sim(p1_rotated[j], pose2[j])

        cos_sim = avg_cos_sim / len(p1_rotated) if len(p1_rotated) > 0 else 0
        if cos_sim > best_cos_sim:
            best_cos_sim = cos_sim
            best_rotation = i

        # Plot the rotated pose
        # plot_id = plot_pose(p1_rotated, plot_id)

    # Make the cos sim be between 0 and 1 instead of -1 and 1
    best_cos_sim = (best_cos_sim + 1) / 2

    # Check if any value in one of the dom color arrays is [None, None, None]
    if not np.any(pose1_dom_colors == [None, None, None]) and not np.any(
        pose2_dom_colors == [None, None, None]
    ):
        dom_color_sim = 1 - colors_diff(pose1_dom_colors, pose2_dom_colors)

    res = best_cos_sim
    # if dom_color_sim is not None:
    #   res -= dom_color_sim * 0.1
    #   res = dom_color_sim
    return res


def colors_diff(colors1: ndarray, colors2: ndarray) -> Union[float, None]:
    """
    Returns the average color difference between two arrays of colors
    colors1 and colors2 should be of shape (n, 3)
    """
    assert (
        colors1.shape == colors2.shape
        and len(colors1.shape) == 2
        and colors1.shape[1] == 3
    )
    if len(colors1) == 0:
        return None

    avg_color_diff = 0
    for col1, col2 in zip(colors1, colors2):
        assert col1.shape == (3,) and col2.shape == (3,)
        avg_color_diff += np.linalg.norm(col1 - col2)
    avg_color_diff /= len(colors1)

    max_possible_color_diff = np.linalg.norm(
        np.array([255, 255, 255]) - np.array([0, 0, 0])
    )
    return avg_color_diff / max_possible_color_diff


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


def draw_landmarks_list(
    image, landmarks, with_index=False, color=(0, 255, 0), title: str = "Landmarks"
):
    if with_index:
        for i, landmark in enumerate(landmarks):
            cv2.putText(
                image,
                str(i),
                (int(landmark.x), int(landmark.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                color,
                3,
            )
    else:
        # for i, landmark in enumerate(landmarks):
        #     cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, color, -1)

        for connection in POSE_CONNECTIONS:
            cv2.line(
                image,
                (int(landmarks[connection[0]].x), int(landmarks[connection[0]].y)),
                (int(landmarks[connection[1]].x), int(landmarks[connection[1]].y)),
                color,
                1,
            )

        top_right = (
            int(max([landmark.x for landmark in landmarks])),
            int(min([landmark.y for landmark in landmarks])),
        )
        cv2.putText(image, title, top_right, cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
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


def points_are_close(x1, y1, x2, y2, threshold=None) -> bool:
    """
    Check if two points are close to each other.
    Args:
        x1: x-coordinate of first point.
        y1: y-coordinate of first point.
        x2: x-coordinate of second point.
        y2: y-coordinate of second point.
        threshold: Threshold for closeness.
    """
    if threshold is None:
        threshold = max(x1, x2, y1, y2) * 0.05
    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold


def rotate_points_y(points: ndarray, angle: int, center: CustomPoint = None) -> ndarray:
    """
    Rotate points around the y-axis by a given angle.
    Args:
        points: List of CustomPoint objects, containing x, y, z coordinates.
        angle: Angle to rotate by, in degrees (int).
        center: Center of rotation (CustomPoint object).
    """
    return np.array([rotate_point_y(point, angle, center) for point in points])


def rotate_point_y(point: ndarray, angle: int, center: ndarray = None) -> ndarray:
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


def get_avg_color(image, x, y, patch_size):
    # Get Average Color of a patch of the image, ignore out of bounds values for calculating average
    colors = []
    count = 0

    for i in range(x - patch_size, x + patch_size):
        for j in range(y - patch_size, y + patch_size):
            if i < 0 or j < 0 or i >= image.shape[0] or j >= image.shape[1]:
                continue

            colors.append(image[i][j])
            count += 1

    avg_color = [None, None, None]
    if count > 0:
        avg_color = np.mean(colors, axis=0)
    return avg_color[0], avg_color[1], avg_color[2]


def get_dominant_color_patch(image, x: float, y: float, patch_size: int) -> ndarray:
    x = int(x)
    y = int(y)
    patch_size = int(np.abs(patch_size))
    min_x = x - patch_size
    max_x = x + patch_size
    min_y = y - patch_size
    max_y = y + patch_size
    return get_dominant_color(image, min_x, min_y, max_x, max_y)


def get_dominant_color_bbox(image, bbox: BoundingBox) -> ndarray:
    return get_dominant_color(image, bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)


def get_dominant_color(img, x0: int, y0: int, x1: int, y1: int) -> ndarray:
    colors = []

    max_y = img.shape[0]
    max_x = img.shape[1]

    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(max_x, x1))
    y1 = int(min(max_y, y1))

    if not (0 <= x0 <= x1 < max_x and 0 <= y0 <= y1 < max_y):
        return np.array([np.nan, np.nan, np.nan])

    for i in range(y0, y1):
        for j in range(x0, x1):
            colors.append(img[i][j])

    assert len(colors) > 0

    # Use KMeans to find the dominant color
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    pixels = np.float32(colors)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return np.array(dominant, dtype=np.uint8)


def triangulate_points(points1, points2, P1, P2) -> ndarray:
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T)
    return points_3d.reshape(-1, 3)


def project_points(points_3d, intrinsic_matrix, extrinsic_matrix):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    points3d_to_camera = np.dot(extrinsic_matrix, points_3d_homogeneous)
    points_2d_homogeneous = np.dot(intrinsic_matrix, points3d_to_camera)
    points_2d = cv2.convertPointsFromHomogeneous(points_2d_homogeneous.T)
    return points_2d.reshape(-1, 2)


def sampson_distance(x1: ndarray, x2: ndarray, F: ndarray):
    """
    Computes the Sampson distance for the given points and fundamental matrix.

    Args:
    x1 (array-like): Homogeneous coordinates of a point in the first image.
    x2 (array-like): Homogeneous coordinates of a point in the second image.
    F (array-like): The fundamental matrix (3x3).

    Returns:
    float: The Sampson distance for the given points and fundamental matrix.
    """
    constraint = x2.T @ F @ x1
    grad_x1 = (x2.T @ F).reshape(-1, 1)
    grad_x2 = (F @ x1).reshape(-1, 1)
    second_term = 1 / (grad_x1.T @ grad_x1 + grad_x2.T @ grad_x2)
    return constraint**2 * second_term


def sampson_error(P1: ndarray, P2: ndarray, points1: ndarray, points2: ndarray, F):
    error = 0
    for i in range(len(points1)):
        x1 = np.append(points1[i], [1])
        x2 = np.append(points2[i], [1])
        p1 = np.matmul(P1, x1)
        p2 = np.matmul(P2, x2)
        error += sampson_distance(p1, p2, F)
    return error


def sampson_distance_derivative(x1, x2, F):
    """
    Computes the derivative of the Sampson distance with respect to the elements of the fundamental matrix F.

    Args:
    x1 (array-like): Homogeneous coordinates of a point in the first image.
    x2 (array-like): Homogeneous coordinates of a point in the second image.
    F (array-like): The fundamental matrix (3x3).

    Returns:
    dS_dF (numpy array): The derivative of the Sampson distance with respect to the elements of F (3x3).
    """
    # Ensure inputs are numpy arrays
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    F = np.asarray(F)

    # Compute the fundamental matrix constraint: x2^T * F * x1
    constraint = x2.T @ F @ x1

    # Compute the gradients of the constraint with respect to x1 and x2
    grad_x1 = (x2.T @ F).reshape(-1, 1)
    grad_x2 = (F @ x1).reshape(-1, 1)

    # Compute the second term in the Sampson distance formula
    second_term = 1 / (grad_x1.T @ grad_x1 + grad_x2.T @ grad_x2)

    # Compute the derivative of the Sampson distance with respect to the elements of F
    outer_product = np.outer(x1, x2)
    dS_dF = 2 * constraint * second_term * outer_product
    return dS_dF


# def reprojection_error(params, points_2d, points_3d, camera_indices, point_indices, num_cameras, num_points):
#     """
#     Computes the reprojection error between the observed 2D points and the reprojected 3D points.
#
#     Args:
#     params (array-like): The camera parameters and 3D point coordinates.
#     points_2d (array-like): The observed 2D points in the image.
#     points_3d (array-like): The 3D points in the scene.
#     camera_indices (array-like): The indices of the cameras corresponding to each observation.
#     point_indices (array-like): The indices of the points corresponding to each observation.
#     num_cameras (int): The number of cameras.
#     num_points (int): The number of 3D points.
#
#     Returns:
#     error (numpy array): The reprojection error for each observation.
#     """
#     projected_points = project_points(params, points_3d, camera_indices, point_indices, num_cameras, num_points)
#     error = projected_points - points_2d
#     return error.ravel()
#
#
# def bundle_adjustment(points_2d, points_3d, camera_indices, point_indices, camera_params, points_3d_initial):
#     """
#     Performs bundle adjustment to refine the camera parameters and 3D point coordinates.
#
#     Args:
#     points_2d (array-like): The observed 2D points in the image.
#     points_3d (array-like): The initial estimates of the 3D points in the scene.
#     camera_indices (array-like): The indices of the cameras corresponding to each observation.
#     point_indices (array-like): The indices of the points corresponding to each observation.
#     camera_params (array-like): The initial estimates of the camera parameters.
#     points_3d_initial (array-like): The initial estimates of the 3D point coordinates.
#
#     Returns:
#     result (OptimizeResult): The optimization result from the SciPy least_squares function.
#     """
#     num_cameras = camera_params.shape[0]
#     num_points = points_3d_initial.shape[0]
#     points_3d = points_3d_initial.copy()
#
#     # Convert the camera parameters and 3D points into a 1D parameter array
#     params_initial = np.hstack((camera_params.ravel(), points_3d_initial.ravel()))
#
#     # Define the reprojection error function
#     error_func = lambda params: reprojection_error(params, points_2d, points_3d, camera_indices, point_indices,
#                                                    num_cameras, num_points)
#
#     # Perform the bundle adjustment using the least_squares function from SciPy
#     result = least_squares(error_func, params_initial, method='lm', verbose=2)
#     return result


def plot_pose_2d(points: ndarray, plot_id: int = None, title: str = "") -> int:
    plot_service: PlotService = PlotService.get_instance()
    if plot_id is not None:
        fig: Figure = plot_service.get_plot(plot_id)
        ax: Axes = fig.get_axes()[0]
        ax.clear()
    else:
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(111)

    fig.suptitle(title)
    # Set axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")

    pts = np.array([[pt[0], pt[1]] for pt in points])
    # Invert y-axis
    pts[:, 1] = -pts[:, 1]

    # Plot points
    ax.scatter(pts[:, 0], pts[:, 1], s=10, c="b", marker="o")

    for connection in CONNECTIONS_LIST:
        a = connection[0]
        b = connection[1]
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], c="b")

    if plot_id is None:
        plot_id = plot_service.add_plot(fig)
    plt.pause(0.001)
    return plot_id


def plot_pose_3d(points: ndarray, ax: Axes) -> None:
    assert points.shape[1] == 3
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("equal")

    # Swap y and z axes
    pts = np.array([[pt[0], pt[2], pt[1]] for pt in points])
    # Invert z-axis
    pts[:, 2] *= -1

    for pt in pts:
        ax.plot([pt[0]], [pt[1]], [pt[2]], "ro")

    for connection in CONNECTIONS_LIST:
        pt1 = pts[connection[0]]
        pt2 = pts[connection[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], "b-")


def blend_colors(
    c1: Tuple[int], c2: Tuple[int], blend_factor: float
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Combines two RGB colors by taking the average of each color channel.

    Args:
        c1 (Tuple[int]): The first color.
        c2 (Tuple[int]): The second color.
        blend_factor (float): The factor by which to blend the colors. Should be between 0 and 1.
    Returns:
        Two tuples representing the partially blended colors.
    """
    middle_color = tuple([int((c1[i] + c2[i]) / 2) for i in range(3)])
    c1 = tuple(
        [
            int(c1[i] * blend_factor + middle_color[i] * (1 - blend_factor))
            for i in range(3)
        ]
    )
    c2 = tuple(
        [
            int(c2[i] * blend_factor + middle_color[i] * (1 - blend_factor))
            for i in range(3)
        ]
    )
    return c1, c2


def compute_essential_normalized(p1: ndarray, p2: ndarray) -> ndarray:
    assert p1.shape == p2.shape, "Input points should have the same shape"
    A = np.hstack(
        (
            (p2[:, 0] * p1[:, 0])[:, np.newaxis],
            (p2[:, 0] * p1[:, 1])[:, np.newaxis],
            p2[:, 0][:, np.newaxis],
            (p2[:, 1] * p1[:, 0])[:, np.newaxis],
            (p2[:, 1] * p1[:, 1])[:, np.newaxis],
            p2[:, 1][:, np.newaxis],
            p1[:, 0][:, np.newaxis],
            p1[:, 1][:, np.newaxis],
            np.ones((p1.shape[0], 1)),
        )
    )

    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    # Enforce the constraint that the singular values are (a, a, 0)
    U, S, V = np.linalg.svd(E)
    S = (S[0] + S[1]) / 2
    E = np.dot(U, np.dot(np.diag([S, S, 0]), V))
    return E


def validate_essential_matrix(
    E: ndarray, p1: ndarray, p2: ndarray, tolerance=1e-4
) -> bool:
    """
    Validates an essential matrix by checking that the epipolar constraint is satisfied for each point pair.

    Args:
        E (ndarray): The essential matrix.
        p1 (ndarray): The first set of points.
        p2 (ndarray): The second set of points.
        tolerance (float): The tolerance for checking the epipolar constraint.

    Returns:
        True if the epipolar constraint is satisfied for each point pair, False otherwise.
    """
    assert len(p1) == len(p2), "The number of points in each set must be equal"
    assert len(p1) > 0, "At least one point pair is required"

    # Check the rank is 2
    if np.linalg.matrix_rank(E) != 2:
        print("The essential matrix is not rank 2")
        return False

    # Check the singular values are equal
    U, S, Vt = np.linalg.svd(E)
    if not np.isclose(S[0], S[1], atol=tolerance):
        print("The singular values of the essential matrix are not equal")
        return False

    # Check E*E.T has one non-zero singular value
    EEt = E @ E.T
    _, S_EEt, _ = np.linalg.svd(EEt)
    if not np.isclose(S_EEt[1:], 0, atol=tolerance).all():
        print("E*E.T does not have one non-zero singular value")
        return False

    # Check the central element of E*E.T
    if not np.isclose(EEt[1, 1], 2 * S[0] ** 2, atol=tolerance):
        print("The central element of E*E.T is not equal to 2 * S[0] ** 2")
        return False

    for i in range(len(p1)):
        x1, y1 = p1[i][0], p1[i][1]
        x2, y2 = p2[i][0], p2[i][1]
        pt1 = np.array([x1, y1, 1])
        pt2 = np.array([x2, y2, 1])
        # Check that pt2^T * E * pt1 = 0
        if np.abs(pt2.T @ E @ pt1) > tolerance:
            print(
                "The epipolar constraint is not satisfied for point pair {} and {}".format(
                    pt1, pt2
                )
            )
            return False
    return True


def decompose_essential_matrix(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Ensure rotation matrix determinant is 1
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt

    P1 = np.hstack((np.dot(U, np.dot(W, Vt)), U[:, 2].reshape(-1, 1)))
    P2 = np.hstack((np.dot(U, np.dot(W, Vt)), -U[:, 2].reshape(-1, 1)))
    P3 = np.hstack((np.dot(U, np.dot(W.T, Vt)), U[:, 2].reshape(-1, 1)))
    P4 = np.hstack((np.dot(U, np.dot(W.T, Vt)), -U[:, 2].reshape(-1, 1)))
    return [P1, P2, P3, P4]


# TODO: DEBUG
def find_correct_projection_matrix(
    P1: ndarray,
    P2: ndarray,
    P3: ndarray,
    P4: ndarray,
    points1: ndarray,
    points2: ndarray,
) -> Tuple[ndarray, ndarray]:
    P = None
    points_3D = None
    P_list = [P1, P2, P3, P4]
    M1 = np.eye(3, 4, dtype=np.float32)
    max_in_front = 0
    for i, P_temp in enumerate(P_list):
        P_temp = P_temp.astype(np.float32)
        points_3D_temp = cv2.triangulatePoints(M1, P_temp, points1.T, points2.T)
        points_3D_temp /= points_3D_temp[3]

        # Check if the points are in front of both cameras
        in_front_of_first_camera = np.dot(M1[2, :3], points_3D_temp[:3]) > 0
        in_front_of_second_camera = np.dot(P_temp[2, :3], points_3D_temp[:3]) > 0

        # If the points are in front of both cameras and there are more such points
        # than for the previous projection matrices, update the projection matrix and 3D points
        if np.sum(in_front_of_first_camera & in_front_of_second_camera) > max_in_front:
            max_in_front = np.sum(in_front_of_first_camera & in_front_of_second_camera)
            P = P_temp
            points_3D = points_3D_temp
    return P, points_3D


def estimate_projection(
    p1: ndarray, p2: ndarray, K1: ndarray, K2: ndarray
) -> [Tuple[Optional[ndarray], Optional[ndarray]]]:
    # p1 = cv2.undistortPoints(src=p1, cameraMatrix=K1, distCoeffs=None)
    # p2 = cv2.undistortPoints(src=p2, cameraMatrix=K2, distCoeffs=None)
    assert len(p1) == len(p2), "The number of points in each set must be equal"
    assert p1.shape[1] == 2, "Each point must be a 2D point"
    assert p2.shape[1] == 2, "Each point must be a 2D point"

    E, mask = cv2.findEssentialMat(p1, p2, K1, cv2.RANSAC, 0.999, 2)
    with Divider("Essential matrix"):
        Logger.log(E, LoggingLevel.DEBUG)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    with Divider("R1"):
        Logger.log(R1, LoggingLevel.DEBUG)
    with Divider("R2"):
        Logger.log(R2, LoggingLevel.DEBUG)
    with Divider("t"):
        Logger.log(t, LoggingLevel.DEBUG)

    # Combine R1, R2 with t to get the four possible projection matrices
    P1 = np.hstack((R1, t))
    P2 = np.hstack((R1, -t))
    P3 = np.hstack((R2, t))
    P4 = np.hstack((R2, -t))
    P, points_3D = find_correct_projection_matrix(P1, P2, P3, P4, p1, p2)
    with Divider("P"):
        Logger.log(P, LoggingLevel.DEBUG)
    with Divider("3D points"):
        Logger.log(points_3D, LoggingLevel.DEBUG)

    if P is None or points_3D is None:
        return None, None

    points_3D = points_3D[:3].T

    # p1, trmat1 = normalize_points(np.array(p1))
    # p2, trmat2 = normalize_points(np.array(p2))
    # E = compute_essential_normalized(p1, p2)
    # if not validate_essential_matrix(E, p1, p2):
    #     Logger.log("Invalid essential matrix", LoggingLevel.ERROR)
    #     return None
    #
    # # Find the four possible projection matrices
    # [P1, P2, P3, P4] = decompose_essential_matrix(E)
    # P, points_3D = find_correct_projection_matrix(P1, P2, P3, P4, p1, p2)
    return P, points_3D


def normalize_image(image):
    """
    Normalizes the image by subtracting the mean and dividing by the standard deviation
    :param image: The image to normalize (h,w,3)
    :return: The normalized image (h,w,3)
    """
    image = image.copy()
    image = image.astype("float32")
    channels = cv2.split(image)
    normalized_channels = []
    for channel in channels:
        mean, std_dev = cv2.meanStdDev(channel)
        normalized_channel = (channel - mean) / std_dev
        normalized_channels.append(normalized_channel)
    normalized_image = cv2.merge(normalized_channels)
    normalized_image = (normalized_image - normalized_image.min()) / (
        normalized_image.max() - normalized_image.min()
    )
    # As bgr between 0 and 255
    normalized_image = image * 255
    normalized_image = image.astype("uint8")
    return normalized_image


# TODO: Fix this or drop the changes to the original code
def unity_to_cv(point: ndarray) -> ndarray:
    # Unity (x, y, z) is equivalent to CV (x, z, -y)
    # return np.array([point[0], point[2], -point[1]])
    # return np.array([-point[1], point[2], point[0]])
    return np.array([point[0], point[1], point[2]])


def rotation_matrix_to_angles(R: ndarray) -> ndarray:
    """
    Converts a rotation matrix to a rotation vector
    :param R: The rotation matrix (3,3)
    :return: The rotation vector (3,1) in degrees
    """
    return np.rad2deg(cv2.Rodrigues(R)[0])


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculates the difference between two angles in degrees
    :param angle1: The first angle in degrees
    :param angle2: The second angle in degrees
    :return: The difference between the two angles in degrees
    """
    return min((angle1 - angle2) % 360, (angle2 - angle1) % 360)


def angles_differences(angles1: ndarray, angles2: ndarray) -> ndarray:
    """
    Calculates the difference between two rotation vectors in degrees
    :param angles1: The first rotation vector (3,1) in degrees
    :param angles2: The second rotation vector (3,1) in degrees
    :return: The difference between the two rotation vectors in degrees
    """
    return np.array([angle_difference(a, b) for a, b in zip(angles1, angles2)])


def rotation_matrices_angles_differences(R1: ndarray, R2: ndarray) -> ndarray:
    """
    Calculates the difference between two rotation matrices in degrees
    :param R1: The first rotation matrix (3,3)
    :param R2: The second rotation matrix (3,3)
    :return: The difference between the two rotation matrices in degrees
    """
    return angles_differences(
        rotation_matrix_to_angles(R1), rotation_matrix_to_angles(R2)
    )


def angles_to_rotation_matrix(angles: ndarray) -> ndarray:
    """
    Converts a rotation vector to a rotation matrix
    :param angles: The rotation vector (3,1) in degrees
    :return: The rotation matrix (3,3)
    """
    return cv2.Rodrigues(np.deg2rad(angles))[0]


def test_normalize_image():
    image_path = "G:\\Uni\\Bachelor\\Project\\combined_camera_poses\\demo\\test.jpg"
    image = cv2.imread(image_path)
    mean_before, std_dev_before = cv2.meanStdDev(image.copy())
    print("Before normalization:")
    print("Mean:", mean_before)
    print("Standard deviation:", std_dev_before)
    normalized_image = normalize_image(image)
    mean_after, std_dev_after = cv2.meanStdDev(normalized_image)
    print("After normalization:")
    print("Mean:", mean_after)
    print("Standard deviation:", std_dev_after)

    cv2.imshow("Original", image)
    cv2.imshow("Normalized", normalized_image)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0
