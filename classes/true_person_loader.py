import os
from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.unity_person import UnityPerson
from enums.logging_levels import LoggingLevel


class TruePersonLoader:
    @staticmethod
    def load(path) -> List[UnityPerson]:
        dirs = os.listdir(path)
        res = []
        for d in dirs:
            res.append(UnityPerson(os.path.join(path, d)))
        return res

    @staticmethod
    def from_3d_points(points: ndarray, frame_count: int, unity_persons: List[UnityPerson]) -> UnityPerson:
        frame_num = frame_count
        points_dict: Dict[UnityPerson, ndarray] = {}

        for unity_person in unity_persons:
            points_dict[unity_person] = unity_person.get_frame(frame_num)

        # Sort by distance
        sorted_points = sorted(points_dict.items(), key=lambda x: np.linalg.norm(x[1] - points))
        Logger.log(f"Closest person is {sorted_points[0][0].jsonpath}", LoggingLevel.INFO, label="TruePersonLoader")
        return sorted_points[0][0]

    @staticmethod
    def confirm_pair(pair: Tuple[Person, Person], unity_persons: List[UnityPerson], frame_count: int,
                     cam_data1: CameraData, cam_data2: CameraData) -> bool:
        assert len(pair) == 2
        assert isinstance(pair[0], Person) and isinstance(pair[1], Person)
        assert frame_count >= 0

        p1: Person = pair[0]
        p2: Person = pair[1]

        # TODO: Use triangulated 3d points instead of the 2d points here

        # Match each person to a unity person using the hungarian algorithm
        l1: List[Tuple[UnityPerson, ndarray]] = [(p, p.get_image_points(frame_count, cam_data1)) for p in unity_persons]
        l2: List[Tuple[UnityPerson, ndarray]] = [(p, p.get_image_points(frame_count, cam_data2)) for p in unity_persons]

        # Sort by distance
        sorted_points1: List[Tuple[UnityPerson, ndarray]] = sorted(l1, key=lambda x: np.linalg.norm(
            x[1] - p1.get_pose_landmarks_numpy_2d()))
        sorted_points2: List[Tuple[UnityPerson, ndarray]] = sorted(l2, key=lambda x: np.linalg.norm(
            x[1] - p2.get_pose_landmarks_numpy_2d()))

        # Check if the closest person is the same
        return sorted_points1[0][0] == sorted_points2[0][0]
