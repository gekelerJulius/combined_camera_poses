import cv2
import mediapipe as mp

from classes.bounding_box import BoundingBox

mp_pose = mp.solutions.pose


def get_pose(image, box: BoundingBox):
    cropped_image = image[box.min_y : box.max_y, box.min_x : box.max_x]
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    results = pose.process(cropped_image)
    pose.close()

    if results is not None and results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            landmark.x = landmark.x * box.get_width() + box.min_x
            landmark.y = landmark.y * box.get_height() + box.min_y
    return image, results
