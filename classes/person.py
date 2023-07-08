from typing import NamedTuple, List, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from numpy import ndarray

from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.logger import Logger
from consts.consts import LANDMARK_INDICES_PER_NAME
from enums.logging_levels import LoggingLevel
from functions.funcs import (
    draw_landmarks_list,
    calc_cos_sim,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def get_real_coordinates(pt: ndarray, camera_data: CameraData) -> ndarray:
    """
    Converts a point in the image to a point in the real world
    :param pt: A point in the image
    :param camera_data: The camera data
    :return: A point in the real world
    """
    intrinsic_matrix3x3: ndarray = camera_data.intrinsic_matrix
    extrinsic_matrix4x4: ndarray = camera_data.extrinsic_matrix4x4

    pt = np.array([pt[0], pt[1], 1])
    # Multiply by the inverse of the intrinsic matrix to get the normalized coordinates
    pt = np.dot(np.linalg.inv(intrinsic_matrix3x3), pt)
    pt = np.array([pt[0], pt[1], 0, 1])
    # Multiply by the inverse of the extrinsic matrix to get the real world coordinates
    pt = np.dot(np.linalg.inv(extrinsic_matrix4x4), pt)
    pt = np.array([pt[0], pt[1], pt[2]])
    return pt


class Person:
    id: str
    frame_count: int
    name: str = None
    color: Tuple[int] = None
    bounding_box: BoundingBox
    results: NamedTuple

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
            return []
        return [lmk for lmk in self.results.pose_landmarks.landmark]

    def get_pose_landmarks_numpy(self) -> np.ndarray:
        if self.results is None or self.results.pose_landmarks is None:
            return np.array([])
        return np.array([[lmk.x, lmk.y, lmk.z] for lmk in self.results.pose_landmarks.landmark])

    def get_pose_landmarks_numpy_2d(self) -> np.ndarray:
        pts_3d = self.get_pose_landmarks_numpy()
        return np.array([[pt[0], pt[1]] for pt in pts_3d])

    def get_world_landmarks(self) -> List[Landmark]:
        if self.results is None or self.results.pose_world_landmarks is None:
            return []
        return [lmk for lmk in self.results.pose_world_landmarks.landmark]

    def get_world_landmarks_numpy(self) -> np.ndarray:
        if self.results is None or self.results.pose_world_landmarks is None:
            return np.array([])
        return np.array([np.array([lmk.x, lmk.y, lmk.z]) for lmk in self.results.pose_world_landmarks.landmark])

    def draw(self, image, color=(255, 255, 0), title=None):
        if title is None and self.name is not None:
            title = self.name
        return draw_landmarks_list(
            image,
            self.get_pose_landmarks(),
            with_index=False,
            color=color,
            title=title,
        )

    def __str__(self):
        return f"Name: {self.name if self.name is not None else 'Unnamed'} " \
               f"| Frame: {self.frame_count} "

    def __repr__(self):
        return self.__str__()

    def pose_distance(self, person2: "Person"):
        if person2 is None:
            Logger.log("No person found", LoggingLevel.WARNING)
            return 0

        p1 = self.get_pose_landmarks_numpy()
        p2 = person2.get_pose_landmarks_numpy()

        # Center the points
        p1 = p1 - np.mean(p1, axis=0)
        p2 = p2 - np.mean(p2, axis=0)

        # Find the scale
        scale = np.sum(np.linalg.norm(p2, axis=1)) / np.sum(np.linalg.norm(p1, axis=1))

        # Scale the points
        p1 *= scale

        # Find the translation
        translation = np.mean(p2, axis=0) - np.mean(p1, axis=0)

        # Transform p1 to p2's space
        p1 += translation

        # Find the error
        # error = np.mean(np.linalg.norm(p2 - p1, axis=1))

        avg_cos_sim = np.mean([calc_cos_sim(p1[i], p2[i]) for i in range(len(p1))])
        return avg_cos_sim

    def get_feet_points(self) -> ndarray:
        feet_indices = [LANDMARK_INDICES_PER_NAME[name] for name in
                        ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]]
        return self.get_pose_landmarks_numpy()[feet_indices]

    def get_feet_points_real(self, camera_data: CameraData) -> ndarray:
        feet_img = self.get_feet_points()
        return np.array([get_real_coordinates(feet_img[i], camera_data) for i in range(4)])

    @staticmethod
    def get_common_visible_landmark_indices(p1: "Person", p2: "Person", vis_tresh: float) -> List[int]:
        lmks1 = p1.get_pose_landmarks()
        lmks2 = p2.get_pose_landmarks()
        return Person.get_common_visible_landmark_indexes_landmark_list(lmks1, lmks2, vis_tresh)

    @staticmethod
    def get_common_visible_landmark_indexes_landmark_list(lmks1: List[Landmark], lmks2: List[Landmark],
                                                          vis_tresh: float) -> List[int]:
        common_lmks = []
        for i in range(len(lmks1)):
            if lmks1[i].visibility > vis_tresh and lmks2[i].visibility > vis_tresh:
                common_lmks.append(i)
        return common_lmks

    def centroid(self):
        return np.mean(self.get_pose_landmarks_numpy(), axis=0)
