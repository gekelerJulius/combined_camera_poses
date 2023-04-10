import cv2
import mediapipe as mp
import numpy as np

from classes.custom_point import CustomPoint

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
    landmarks = get_landmarks_as_coordinates(results, cropped)
    for landmark in landmarks:
        landmark.x += min_x
        landmark.y += min_y
    frame = draw_landmarks_list(landmarks, img)
    return frame, landmarks


def are_landmarks_the_same(landmarks1, landmarks2):
    avg_diff = compare_landmarks(landmarks1, landmarks2)
    print(f"Average difference: {avg_diff}")
    return avg_diff < 5


def compare_landmarks(landmarks1, landmarks2):
    diffs = []
    if len(landmarks1) != len(landmarks2):
        print("Landmarks are not the same length")
        return False
    for i in range(len(landmarks1)):
        diffs.append(landmarks1[i].distance(landmarks2[i]))

    if max(diffs) > 10:
        print("Landmarks are not the same")
        return False

    summed = 0
    for diff in diffs:
        summed += diff
    return summed / len(diffs)


def get_pose(image):
    # Annotate
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def get_landmarks_as_coordinates(results, image):
    # Check for NoneType
    if results.pose_landmarks is None:
        return []

    image_height = image.shape[0]
    image_width = image.shape[1]
    return [
        CustomPoint(landmark.x * image_width, landmark.y * image_height)
        for landmark in results.pose_landmarks.landmark
    ]


def draw_landmarks(results, image):
    landmarks = get_landmarks_as_coordinates(results, image)
    for landmark in landmarks:
        cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, (0, 255, 0), -1)
    return image


def draw_landmarks_list(landmarks, image):
    for landmark in landmarks:
        cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, (0, 255, 0), -1)
    return image
