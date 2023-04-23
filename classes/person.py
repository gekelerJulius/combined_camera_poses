from typing import List
import mediapipe as mp
from classes.bounding_box import BoundingBox
from classes.custom_point import CustomPoint
from functions.funcs import draw_landmarks_list, compare_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class Person:
    def __init__(
            self,
            person_id: int,
            frame_count: int,
            bounding_box: BoundingBox,
            landmarks: List[CustomPoint],
    ):
        self.id = person_id
        self.frame_count = frame_count
        self.bounding_box = bounding_box
        self.landmarks = landmarks

    def draw(self, image):
        return draw_landmarks_list(image, self.landmarks, with_index=True)

    def center(self):
        return self.landmarks[0]

    def __str__(self):
        return f"ID: {self.id} | Frame: {self.frame_count} | Bounding Box: {self.bounding_box} | Landmarks: {self.landmarks}"

    def get_landmark_diff(self, person2):
        return compare_landmarks(self.landmarks, person2.landmarks)
