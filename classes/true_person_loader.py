import os
from typing import List, Dict

import numpy as np
from numpy import ndarray

from classes.logger import Logger
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
