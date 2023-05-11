import itertools
import time
import xml.etree.ElementTree as ET
from typing import Union, List

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pyhull.simplex import Simplex
from scipy.linalg import orthogonal_procrustes
from pyhull.convex_hull import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

from classes.PlotService import PlotService
from classes.colored_landmark import ColoredLandmark
from classes.custom_point import CustomPoint
from classes.logger import Logger
from consts.consts import CONNECTIONS_LIST
from enums.logging_levels import LoggingLevel
from functions.get_pose import get_pose

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    avg_dist = np.mean(np.linalg.norm(points - centroid, axis=1))

    if avg_dist == 0:
        Logger.log("All points are the same", LoggingLevel.INFO)
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
        plot_id = plot_pose(p1_rotated, plot_id)

    # Make the cos sim be between 0 and 1 instead of -1 and 1
    best_cos_sim = (best_cos_sim + 1) / 2

    dom_color_sim = None
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


def colors_diff(colors1, colors2):
    if len(colors1) != len(colors2):
        return None

    if len(colors1) == 0:
        return None

    avg_color_diff = 0

    for i in range(len(colors1)):
        avg_color_diff += np.linalg.norm(colors1[i] - colors2[i])
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
                3,
            )
    else:
        # for i, landmark in enumerate(landmarks):
        #     cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, color, -1)
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
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


def get_dominant_color(image, x, y, patch_size):
    # First get all color values in the patch that are not out of bounds
    colors = []

    for i in range(x - patch_size, x + patch_size):
        for j in range(y - patch_size, y + patch_size):
            if i < 0 or j < 0 or i >= image.shape[0] or j >= image.shape[1]:
                continue

            colors.append(image[i][j])

    if len(colors) == 0:
        # Logger.log("No colors found in patch", LoggingLevel.WARNING)
        return [None, None, None]

    # Use KMeans to find the dominant color
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    pixels = np.float32(colors)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    # Check if dominant is a tuple of 3 numbers
    res = dominant[0], dominant[1], dominant[2]
    if len(res) != 3:
        Logger.log("Dominant color is not a tuple of 3 numbers", LoggingLevel.WARNING)
        return [None, None, None]
    return res


def triangulate_points(points1, points2, P1, P2) -> ndarray:
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T)
    return points_3d.reshape(-1, 3)


def project_points(points_3d, intrinsic_matrix, extrinsic_matrix):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    points_2d_homogeneous = np.dot(intrinsic_matrix, np.dot(extrinsic_matrix, points_3d_homogeneous))
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
    return constraint ** 2 * second_term


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

def get_simplex_normal(simplex: Simplex):
    """
    Calculates the normal vector of the plane defined by the given simplex in 3D space.

    Args:
        simplex (Simplex): A Simplex object representing the simplex.

    Returns:
        A numpy array representing the normal vector of the plane.
    """
    # Check that the simplex is a triangle (i.e., 3 vertices)
    if simplex.simplex_dim != 3:
        raise ValueError('Simplex is not a triangle')

    # Calculate the normal vector using the cross product of two edge vectors
    coords = simplex.coords
    u = coords[1] - coords[0]
    v = coords[2] - coords[0]
    normal = np.cross(u, v)
    return normal / np.linalg.norm(normal)


# Create plotter class with ids for plots
def plot_pose(points: ndarray, plot_id: int = None) -> int:
    plot_service: PlotService = PlotService.get_instance()
    if plot_id is not None:
        fig: Figure = plot_service.get_plot(plot_id)
        ax: Axes3D = fig.get_axes()[0]
        ax.clear()
    else:
        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Set axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')

    # Swap y and z axes
    pts = np.array([[pt[0], pt[2], pt[1]] for pt in points])
    # Invert z-axis
    pts[:, 2] *= -1

    for pt in pts:
        ax.plot([pt[0]], [pt[1]], [pt[2]], 'ro')

    for connection in CONNECTIONS_LIST:
        pt1 = pts[connection[0]]
        pt2 = pts[connection[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'b-')

    # conv_hull = ConvexHull(pts)
    # for j in range(len(conv_hull.simplices)):
    #     simplex: Simplex = conv_hull.simplices[j]
    #
    #     # Draw planes for the specified simplices
    #     simplex_points: ndarray = simplex.coords
    #     X = simplex_points[:, 0]
    #     Y = simplex_points[:, 1]
    #     Z = simplex_points[:, 2]
    #     ax.plot_trisurf(X, Y, Z, alpha=0.3)

    if plot_id is None:
        plot_id = plot_service.add_plot(fig)
    plt.pause(0.01)
    return plot_id
