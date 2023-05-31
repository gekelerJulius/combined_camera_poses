from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.get_person_pairs import compare_persons


class FrameRecord:
    frame_num: int
    persons1: List[Person]
    persons2: List[Person]
    cost_matrix: Dict[str, Dict[str, float]]

    def __init__(
            self,
            frame_num: int,
            persons1: List[Person],
            persons2: List[Person],
            cost_matrix: Dict[str, Dict[str, float]],
    ):
        self.frame_num = frame_num
        self.persons1 = persons1
        self.persons2 = persons2
        self.cost_matrix = cost_matrix


class RecordMatcher:
    rec1: PersonRecorder
    rec2: PersonRecorder
    frame_records: List[FrameRecord]

    def __init__(self, rec1: PersonRecorder, rec2: PersonRecorder):
        self.rec1 = rec1
        self.rec2 = rec2
        self.frame_records = []

    def get_alignment(self, frame_num: int) -> List[Tuple[Person, Person]]:
        look_back = 24
        recent_persons1 = self.rec1.get_recent_persons(frame_num, look_back)
        recent_persons2 = self.rec2.get_recent_persons(frame_num, look_back)

        if len(recent_persons1) == 0 or len(recent_persons2) == 0:
            return []

        cost_matrix = np.zeros((len(recent_persons1), len(recent_persons2)))
        for i, a in enumerate(recent_persons1):
            for j, b in enumerate(recent_persons2):
                relevant_records = [
                    x
                    for x in self.frame_records
                    if x.frame_num >= frame_num - look_back
                ]

                if len(relevant_records) == 0:
                    # cost_matrix[i, j] = np.inf
                    # continue
                    print("No relevant records found")
                    exit(1)

                costs = []
                for record in relevant_records:
                    if a in record.persons1 and b in record.persons2:
                        matr_a = record.cost_matrix.get(a.name)
                        if matr_a is None:
                            continue
                        matr_b = matr_a.get(b.name)
                        if matr_b is None:
                            continue
                        costs.append(matr_b)
                if len(costs) == 0:
                    cost_matrix[i, j] = 1000000
                    continue
                cost_matrix[i, j] = np.mean(costs)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs: List[Tuple[Person, Person]] = []
        for i, j in zip(row_ind, col_ind):
            pairs.append((recent_persons1[i], recent_persons2[j]))
        return pairs

    def eval_frame(
            self,
            frame_num: int,
            img1=None,
            img2=None,
            cam_data1: CameraData = None,
            cam_data2: CameraData = None,
    ) -> None:
        persons1 = self.rec1.frame_dict.get(frame_num)
        persons2 = self.rec2.frame_dict.get(frame_num)

        if (
                persons1 is None
                or len(persons1) == 0
                or persons2 is None
                or len(persons2) == 0
        ):
            persons1 = []
            persons2 = []

        max_diff = 0
        cost_matrix: Dict[str, Dict[str, float]] = {}
        for i in range(len(persons1)):
            a = persons1[i]
            cost_matrix[a.name] = {}
            matr = cost_matrix[a.name]
            for j in range(len(persons2)):
                b = persons2[j]
                diff = compare_persons(
                    a,
                    b,
                    img1,
                    img2,
                    self.rec1,
                    self.rec2,
                    frame_num,
                    cam_data1,
                    cam_data2,
                )
                if diff > max_diff:
                    max_diff = diff
                matr[b.name] = diff
        self.frame_records.append(
            FrameRecord(frame_num, persons1, persons2, cost_matrix)
        )
