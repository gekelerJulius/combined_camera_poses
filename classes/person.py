from typing import NamedTuple, List
from mediapipe.tasks.python.components.containers.landmark import Landmark
import mediapipe as mp

from classes.bounding_box import BoundingBox
from classes.colored_landmark import ColoredLandmark
from classes.logger import Logger
from enums.logging_levels import LoggingLevel
from functions.funcs import (
    compare_landmarks,
    draw_landmarks_list,
    get_avg_color,
    get_dominant_color,
)

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

    def get_pose_landmarks(self) -> List[Landmark]:
        if self.results is None or self.results.pose_landmarks is None:
            Logger.log(LoggingLevel.WARNING, "No pose landmarks found")
            return []
        return [lmk for lmk in self.results.pose_landmarks.landmark]

    def get_world_landmarks(self) -> List[Landmark]:
        if self.results is None or self.results.pose_world_landmarks is None:
            Logger.log(LoggingLevel.WARNING, "No world landmarks found")
            return []
        return [lmk for lmk in self.results.pose_world_landmarks.landmark]

    def get_pose_landmarks_with_color(self, image) -> List[ColoredLandmark]:
        if self.results is None or self.results.pose_landmarks is None:
            Logger.log(LoggingLevel.WARNING, "No pose landmarks found")
            return []
        if image is None:
            Logger.log(LoggingLevel.WARNING, "No image found")
            return []

        lmks = [
            ColoredLandmark(
                lmk,
                avg_color=get_avg_color(image, int(lmk.x), int(lmk.y), 3),
                dom_color=get_dominant_color(image, int(lmk.x), int(lmk.y), 3),
            )
            for lmk in self.results.pose_landmarks.landmark
        ]

        return lmks

    def draw(self, image, color=(0, 255, 0)):
        return draw_landmarks_list(
            image, self.get_pose_landmarks(), with_index=False, color=color
        )

    def __str__(self):
        return f"ID: {self.id} | Frame: {self.frame_count} | Bounding Box: {self.bounding_box} | Landmarks: {self.get_pose_landmarks()}"

    def get_landmark_sim(self, person2: "Person", img1, img2):
        if person2 is None:
            Logger.log(LoggingLevel.WARNING, "No person found")
            return 0

        if img1 is None or img2 is None:
            Logger.log(LoggingLevel.WARNING, "No image found")
            return 0

        return compare_landmarks(
            self.get_pose_landmarks_with_color(img1),
            person2.get_pose_landmarks_with_color(img2),
        )
