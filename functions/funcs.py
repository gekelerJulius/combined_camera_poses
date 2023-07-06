from typing import Union, Tuple

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from numpy import ndarray

from classes.logger import Logger
from classes.plot_service import PlotService
from consts.consts import CONNECTIONS_LIST
from enums.logging_levels import LoggingLevel

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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

def get_dominant_color_patch(image, x: float, y: float, patch_size: int) -> ndarray:
    x = int(x)
    y = int(y)
    patch_size = int(np.abs(patch_size))
    min_x = x - patch_size
    max_x = x + patch_size
    min_y = y - patch_size
    max_y = y + patch_size
    return get_dominant_color(image, min_x, min_y, max_x, max_y)

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


def plot_pose_2d(points: ndarray, plot_id: str = None, title: str = "") -> str:
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
