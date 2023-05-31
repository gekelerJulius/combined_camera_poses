import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
import cv2 as cv

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import estimate_extrinsic
from functions.get_person_pairs import compare_persons


class FrameRecord:
    frame_num: int
    persons1: List[Person]
    persons2: List[Person]
    cost_matrix: Dict[str, Dict[str, float]]
    estimated_person_pairs: List[Tuple[Person, Person]]
    estimated_extrinsic_matrix: Optional[np.ndarray]

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
        self.estimated_person_pairs = []
        self.estimated_extrinsic_matrix = None


class RecordMatcher:
    rec1: PersonRecorder
    rec2: PersonRecorder
    frame_records: List[FrameRecord]

    def __init__(self, rec1: PersonRecorder, rec2: PersonRecorder):
        self.rec1 = rec1
        self.rec2 = rec2
        self.frame_records = []

    def get_frame_record(self, frame_num: int) -> Optional[FrameRecord]:
        for record in self.frame_records:
            if record.frame_num == frame_num:
                return record
        return None

    def get_all_previous_extrinsics(self, frame_num: int) -> List[np.ndarray]:
        res = []
        for record in self.frame_records:
            if record.frame_num < frame_num:
                res.append(record.estimated_extrinsic_matrix)
        return [x for x in res if x is not None]

    def get_alignment(self, frame_num: int) -> List[Tuple[Person, Person]]:
        look_back = 10000
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
                    cost_matrix[i, j] = 1e12
                    continue
                cost_matrix[i, j] = np.mean(costs)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs: List[Tuple[Person, Person]] = []
        for i, j in zip(row_ind, col_ind):
            pairs.append((recent_persons1[i], recent_persons2[j]))
        self.get_frame_record(frame_num).estimated_person_pairs = pairs
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

    def estimate_extrinsic_matrix(
            self, frame_num: int, cam_data1: CameraData, cam_data2: CameraData
    ) -> None:
        frame_record = self.get_frame_record(frame_num)
        if frame_record is None:
            return None
        pairs = frame_record.estimated_person_pairs
        if pairs is None or len(pairs) == 0:
            return None

        points1_img = []
        points2_img = []
        for pair in pairs:
            points1_img.extend(pair[0].get_pose_landmarks_numpy_2d())
            points2_img.extend(pair[1].get_pose_landmarks_numpy_2d())

        points1_img = np.array(points1_img)
        points2_img = np.array(points2_img)
        frame_record.estimated_extrinsic_matrix = estimate_extrinsic(
            points1_img, points2_img, cam_data1.intrinsic_matrix, cam_data2.intrinsic_matrix
        )
        R, t = (
            frame_record.estimated_extrinsic_matrix[:, :3],
            frame_record.estimated_extrinsic_matrix[:, 3],
        )
        prevs = self.get_all_previous_extrinsics(frame_num)
        if len(prevs) < 5:
            return

        # Calculate mean of previous extrinsics
        median = np.median(prevs, axis=0)
        median_R = median[:, :3]
        median_t = median[:, 3]
        median_r = cv.Rodrigues(median_R)[0]
        median_rdeg = np.rad2deg(median_r)

        r = cv.Rodrigues(R)[0]
        rdeg = np.rad2deg(r)
        rot_diff = np.linalg.norm(median_rdeg - rdeg)
        if rot_diff > 20:
            return

        K1 = cam_data1.intrinsic_matrix
        K2 = cam_data2.intrinsic_matrix
        R1 = np.eye(3)
        R2 = median_R
        t1 = np.zeros((3, 1))
        t2 = np.array([median_t]).T
        est_cam_data1 = CameraData.from_matrices(K1, R1, t1)
        est_cam_data2 = CameraData.from_matrices(K2, R2, t2)
        err1, err2 = calc_reprojection_errors(points1_img, points2_img, est_cam_data1, est_cam_data2)
        Logger.log(err1, LoggingLevel.DEBUG, "Reprojection error 1")
        Logger.log(err2, LoggingLevel.DEBUG, "Reprojection error 2")
