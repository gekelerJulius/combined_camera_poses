import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
from numpy import ndarray

from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.logger import Logger, Divider
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
    def confirm_pair(
        pair: Tuple[Person, Person],
        unity_persons: List[UnityPerson],
        frame_count: int,
        img1: ndarray,
        img2: ndarray,
    ) -> bool:
        assert len(pair) == 2
        assert isinstance(pair[0], Person) and isinstance(pair[1], Person)
        assert frame_count >= 0

        p1: Person = pair[0]
        p2: Person = pair[1]

        img_height1 = img1.shape[0]
        img_height2 = img2.shape[0]
        pts1: List[ndarray] = [
            u.get_image_points(frame_count, 0, img_height1) for u in unity_persons
        ]

        pts2: List[ndarray] = [
            u.get_image_points(frame_count, 1, img_height2) for u in unity_persons
        ]

        bounding_boxes1: List[BoundingBox] = [
            u.get_bounding_box(frame_count, 0, img_height1) for u in unity_persons
        ]

        bounding_boxes2: List[BoundingBox] = [
            u.get_bounding_box(frame_count, 1, img_height2) for u in unity_persons
        ]

        # Show Bounding Boxes
        # for bbox in bounding_boxes1:
        #     bbox.draw(img1)
        # for bbox in bounding_boxes2:
        #     bbox.draw(img2)

        # Show Points
        # for pts in pts1:
        #     for pt in pts:
        #         cv2.circle(img1, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        #     break
        #
        # for pts in pts2:
        #     for pt in pts:
        #         cv2.circle(img2, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        #     break

        costs1: List[Tuple[int, float]] = []
        costs2: List[Tuple[int, float]] = []
        for i, u in enumerate(unity_persons):
            bbox1 = bounding_boxes1[i]
            bbox2 = bounding_boxes2[i]
            overlap1 = bbox1.calculate_overlap_percentage(p1.bounding_box)
            overlap2 = bbox2.calculate_overlap_percentage(p2.bounding_box)
            costs1.append((i, overlap1))
            costs2.append((i, overlap2))

        costs1 = sorted(costs1, key=lambda x: x[1], reverse=True)
        costs2 = sorted(costs2, key=lambda x: x[1], reverse=True)
        # Logger.log(costs1, LoggingLevel.DEBUG, label="Costs1")
        # Logger.log(costs2, LoggingLevel.DEBUG, label="Costs2")
        return (
            costs1[0][0] == costs2[0][0] and costs1[0][1] > 0.5 and costs2[0][1] > 0.5
        )
