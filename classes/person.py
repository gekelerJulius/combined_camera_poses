from typing import NamedTuple

import mediapipe as mp

from classes.bounding_box import BoundingBox
from functions.funcs import compare_landmarks, draw_landmarks_list

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class Person:
    def __init__(
        self,
        person_id: str,
        frame_count: int,
        bounding_box: BoundingBox,
        results: NamedTuple,
    ):
        self.id = person_id
        self.frame_count = frame_count
        self.bounding_box = bounding_box
        self.results = results

    @staticmethod
    def get_landmark_ids():
        return [mp.solutions.pose.PoseLandmark(i) for i in range(33)]

    def get_pose_landmarks(self):
        if self.results is None or self.results.pose_landmarks is None:
            print("No landmarks found")
            return []
        return [lmk for lmk in self.results.pose_landmarks.landmark]

    def get_world_landmarks(self):
        if self.results is None or self.results.pose_world_landmarks is None:
            print("No world landmarks found")
            return []
        return [lmk for lmk in self.results.pose_world_landmarks.landmark]

    def draw(self, image, color=(0, 255, 0)):
        return draw_landmarks_list(
            image, self.get_pose_landmarks(), with_index=True, color=color
        )

    def __str__(self):
        return f"ID: {self.id} | Frame: {self.frame_count} | Bounding Box: {self.bounding_box} | Landmarks: {self.get_pose_landmarks()}"

    def get_landmark_diff(self, person2):
        return compare_landmarks(
            self.get_world_landmarks(), person2.get_world_landmarks()
        )
