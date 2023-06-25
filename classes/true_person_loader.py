import os
from typing import List, Tuple

import cv2
from numpy import ndarray

from classes.bounding_box import BoundingBox
from classes.person import Person
from classes.unity_person import UnityPerson


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

        bounding_boxes1: List[BoundingBox] = [
            u.get_bounding_box(frame_count, 0, img_height1) for u in unity_persons
        ]

        bounding_boxes2: List[BoundingBox] = [
            u.get_bounding_box(frame_count, 1, img_height2) for u in unity_persons
        ]

        costs1: List[Tuple[int, float]] = []
        costs2: List[Tuple[int, float]] = []
        for i, u in enumerate(unity_persons):
            bbox1 = bounding_boxes1[i]
            bbox2 = bounding_boxes2[i]
            overlap1 = bbox1.calculate_overlap_percentage(p1.bounding_box)

            # if overlap1 > 0.5:
            #     p1.bounding_box.draw(img1, (180, 0, 0))
            #     bbox1.draw(img1, (0, 180, 0))
            #     cv2.imshow("img1", img1)
            #     cv2.waitKey(0)
            #     cv2.destroyWindow("img1")

            overlap2 = bbox2.calculate_overlap_percentage(p2.bounding_box)
            costs1.append((i, overlap1))
            costs2.append((i, overlap2))

        costs1 = sorted(costs1, key=lambda x: x[1], reverse=True)
        costs2 = sorted(costs2, key=lambda x: x[1], reverse=True)
        return (
            costs1[0][0] == costs2[0][0] and costs1[0][1] > 0.5 and costs2[0][1] > 0.5
        )
